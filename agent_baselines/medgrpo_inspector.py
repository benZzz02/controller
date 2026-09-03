"""Transformers inference adapter for the released MedGRPO model."""

from __future__ import annotations

import sys
import json
from urllib.request import Request, urlopen
from pathlib import Path
from typing import Any, Dict, List, Optional

from .controller import ClipCandidate, InspectionResult, Inspector, VideoRequest
from .video_io import load_frame_inputs


def _default_medgrpo_root() -> Path:
    return Path(__file__).resolve().parents[2] / "MedGRPO-Code"


class MedGRPOInspector:
    """Use MedGRPO as a frozen evidence/answer tool on one retrieved clip.

    The adapter uses the repository's medical video preprocessor, then calls
    the Hugging Face Transformers model directly.  This is intentionally
    single-request inference: it fits the controller loop and avoids hiding
    tool-call accounting inside a batch server.
    """

    def __init__(
        self,
        model_path: str,
        *,
        medgrpo_root: str | Path | None = None,
        device: str = "cuda:1",
        device_map: str | Dict[str, Any] | None = None,
        quantized: bool = True,
        max_new_tokens: int = 256,
        max_pixels: int = 24 * 28 * 28,
        min_pixels: int = 8 * 28 * 28,
        prompt_prefix: str = "Answer the surgical video question using only the provided clip.",
    ) -> None:
        self.model_path = model_path
        self.medgrpo_root = Path(medgrpo_root) if medgrpo_root else _default_medgrpo_root()
        self.device = device
        self.device_map = device_map
        self.quantized = quantized
        self.max_new_tokens = max_new_tokens
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels
        self.prompt_prefix = prompt_prefix
        self.processor: Any = None
        self.model: Any = None
        self._process_vision_info: Any = None

    def _load_backend(self) -> None:
        if self.model is not None:
            return
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        inference_root = self.medgrpo_root / "inference"
        if not inference_root.exists():
            raise FileNotFoundError(
                f"MedGRPO inference code not found at {inference_root}. "
                "Pass --medgrpo-root or clone MedGRPO-Code."
            )
        if str(inference_root) not in sys.path:
            sys.path.insert(0, str(inference_root))
        from vision_process_medical import process_vision_info_medical  # type: ignore

        self._process_vision_info = process_vision_info_medical
        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            padding_side="left",
            max_pixels=self.max_pixels,
            min_pixels=self.min_pixels,
            trust_remote_code=True,
        )

        model_kwargs: Dict[str, Any] = {"trust_remote_code": True}
        # Pin the inspector to its assigned GPU by default.  `auto` would see
        # both GPUs in the controller process and may split MedGRPO across the
        # controller GPU and inspector GPU, invalidating the budget.
        model_kwargs["device_map"] = self.device_map or {"": self.device}
        if self.quantized:
            from transformers import BitsAndBytesConfig

            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
        else:
            model_kwargs["torch_dtype"] = torch.float16
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_path,
            **model_kwargs,
        ).eval()

    def _model_device(self) -> Any:
        import torch

        device = getattr(self.model, "device", None)
        if device is not None and str(device) != "meta":
            return device
        return torch.device(self.device)

    @staticmethod
    def _frames_for_candidate(request: VideoRequest, candidate: ClipCandidate) -> List[Any]:
        if candidate.frames:
            return list(candidate.frames)
        frame_paths = candidate.metadata.get("frame_paths") or request.frame_paths
        video_path = candidate.metadata.get("video_path") or request.video_path
        return load_frame_inputs(
            frame_paths=frame_paths,
            video_path=video_path,
            start_sec=candidate.start_sec,
            end_sec=candidate.end_sec,
            fps=request.fps,
            num_frames=16,
        )

    def inspect(self, request: VideoRequest, candidate: ClipCandidate) -> InspectionResult:
        self._load_backend()
        frames = self._frames_for_candidate(request, candidate)
        if not frames:
            raise ValueError(f"No frames available for candidate {candidate.clip_id}")

        question = f"{self.prompt_prefix}\nQuestion: {request.question}"
        sample_fps = float(candidate.metadata.get("fps") or request.fps or 2.0)
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": frames,
                        "sample_fps": sample_fps,
                        "max_pixels": self.max_pixels,
                        "min_pixels": self.min_pixels,
                    },
                    {"type": "text", "text": question},
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        image_inputs, video_inputs, video_kwargs = self._process_vision_info(
            messages,
            return_video_kwargs=True,
        )
        processor_kwargs: Dict[str, Any] = {
            "text": [text],
            "padding": True,
            "return_tensors": "pt",
        }
        if image_inputs is not None:
            processor_kwargs["images"] = image_inputs
        if video_inputs is not None:
            processor_kwargs["videos"] = video_inputs

        try:
            inputs = self.processor(**processor_kwargs, **video_kwargs)
        except (TypeError, ValueError):
            # Older Qwen processors do not accept the optional medical video
            # keyword arguments; the already-processed video is still valid.
            inputs = self.processor(**processor_kwargs)
        inputs = inputs.to(self._model_device())

        import torch

        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
        prompt_length = inputs.input_ids.shape[1]
        generated_ids = generated_ids[:, prompt_length:]
        answer = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()
        return InspectionResult(
            text=answer,
            metadata={
                "model": self.model_path,
                "clip_id": candidate.clip_id,
                "num_frames": len(frames),
                "start_sec": candidate.start_sec,
                "end_sec": candidate.end_sec,
            },
        )


class RemoteMedGRPOInspector:
    """Call a MedGRPOInspector hosted by the separate vllm environment."""

    def __init__(self, endpoint: str, timeout: int = 1800) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.timeout = timeout

    def inspect(self, request: VideoRequest, candidate: ClipCandidate) -> InspectionResult:
        payload = {
            "request": {"qid": request.qid, "video_id": request.video_id, "question": request.question,
                        "video_path": request.video_path, "fps": request.fps},
            "candidate": {"clip_id": candidate.clip_id, "video_id": candidate.video_id,
                          "start_sec": candidate.start_sec, "end_sec": candidate.end_sec,
                          "score": candidate.score, "metadata": candidate.metadata},
        }
        body = json.dumps(payload).encode("utf-8")
        response = urlopen(Request(self.endpoint, data=body, headers={"Content-Type": "application/json"}), timeout=self.timeout)
        result = json.loads(response.read().decode("utf-8"))
        return InspectionResult(text=result["text"], metadata=result.get("metadata", {}))
