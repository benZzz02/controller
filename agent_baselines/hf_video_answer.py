"""Frozen Hugging Face video-answer adapter used as the controller model.

For a Qwen2.5-VL-compatible checkpoint this is ready to run.  The
SurgLLaVA-Video checkpoint is built on TinyLLaVA-Video, so its native loader
may be needed if the checkpoint does not register an ``AutoModel`` class; the
``backend=auto`` path gives a clear error in that case instead of silently
loading a text-only model.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional

from .controller import AnswerModel, ClipCandidate, ModelAnswer, VideoRequest
from .video_io import load_frame_inputs


class HuggingFaceVideoAnswerModel:
    """Call a frozen video VLM through its registered Transformers class."""

    def __init__(
        self,
        model_path: str,
        *,
        backend: str = "auto",
        device: str = "cuda:0",
        device_map: str | Dict[str, Any] | None = None,
        quantized: bool = False,
        max_new_tokens: int = 256,
        num_frames: int = 16,
        vision_processor_root: str | Path | None = None,
        surgpub_root: str | Path | None = None,
    ) -> None:
        self.model_path = model_path
        self.backend = backend
        self.device = device
        self.device_map = device_map
        self.quantized = quantized
        self.max_new_tokens = max_new_tokens
        self.num_frames = num_frames
        self.vision_processor_root = Path(vision_processor_root) if vision_processor_root else None
        self.surgpub_root = Path(surgpub_root) if surgpub_root else None
        self.processor: Any = None
        self.model: Any = None
        self._process_vision_info: Any = None

    def _load_backend(self) -> None:
        if self.model is not None:
            return
        import torch
        from transformers import AutoProcessor

        if self.backend == "tinyllava":
            if self.surgpub_root and str(self.surgpub_root) not in sys.path:
                sys.path.insert(0, str(self.surgpub_root))
            from tinyllava.model.load_model import load_pretrained_model

            self.model, self.tokenizer, self.image_processor, _ = load_pretrained_model(
                self.model_path,
                load_4bit=self.quantized,
                device=self.device,
            )
            self.model = self.model.to(self.device).eval()
            from tinyllava.data.image_preprocess import ImagePreprocess
            from tinyllava.data.video_preprocess import VideoPreprocess
            from tinyllava.data.text_preprocess import TextPreprocess
            from tinyllava.utils.message import Message

            self._tiny = {
                "ImagePreprocess": ImagePreprocess,
                "VideoPreprocess": VideoPreprocess,
                "TextPreprocess": TextPreprocess,
                "Message": Message,
            }
            self._tiny_video_preprocess = VideoPreprocess(self.image_processor, self.model.config)
            self._tiny_text_preprocess = TextPreprocess(self.tokenizer, "qwen2_base")
            return

        try:
            self.processor = AutoProcessor.from_pretrained(
                self.model_path,
                trust_remote_code=True,
            )
        except TypeError:
            self.processor = AutoProcessor.from_pretrained(self.model_path)

        model_kwargs: Dict[str, Any] = {"trust_remote_code": True}
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

        if self.backend == "qwen2_5_vl":
            from transformers import Qwen2_5_VLForConditionalGeneration

            model_cls = Qwen2_5_VLForConditionalGeneration
        else:
            model_cls = None
            try:
                from transformers import AutoModelForVision2Seq

                model_cls = AutoModelForVision2Seq
            except ImportError:
                pass
            if model_cls is None:
                from transformers import AutoModelForCausalLM

                model_cls = AutoModelForCausalLM

        try:
            self.model = model_cls.from_pretrained(self.model_path, **model_kwargs).eval()
        except Exception as error:
            raise RuntimeError(
                "The controller checkpoint is not loadable through the generic "
                "Transformers video adapter. For SurgLLaVA-Video/TinyLLaVA-Video, "
                "install its native model package and replace this adapter's "
                "_load_backend/_generate implementation."
            ) from error

        if self.vision_processor_root:
            root = self.vision_processor_root / "inference"
            if str(root) not in sys.path:
                sys.path.insert(0, str(root))
            try:
                from vision_process_medical import process_vision_info_medical  # type: ignore

                self._process_vision_info = process_vision_info_medical
            except ImportError:
                self._process_vision_info = None

    def _model_device(self) -> Any:
        import torch

        device = getattr(self.model, "device", None)
        if device is not None and str(device) != "meta":
            return device
        return torch.device(self.device)

    def _frames(self, request: VideoRequest, candidate: Optional[ClipCandidate]) -> list[Any]:
        if candidate is not None and candidate.frames:
            return list(candidate.frames)
        start_sec = candidate.start_sec if candidate is not None else request.start_sec
        end_sec = candidate.end_sec if candidate is not None else request.end_sec
        return load_frame_inputs(
            frame_paths=request.frame_paths,
            video_path=request.video_path,
            start_sec=start_sec,
            end_sec=end_sec,
            fps=request.fps,
            num_frames=self.num_frames,
        )

    def _build_inputs(self, request: VideoRequest, frames: list[Any], prompt: str) -> Any:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": frames, "sample_fps": float(request.fps or 2.0)},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        if hasattr(self.processor, "apply_chat_template"):
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            text = prompt

        image_inputs = None
        video_inputs = frames
        video_kwargs: Dict[str, Any] = {}
        if self._process_vision_info is not None:
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
            # Some TinyLLaVA-style processors use raw frame lists and do not
            # understand Qwen's optional video keyword arguments.
            inputs = self.processor(**processor_kwargs)
        return inputs.to(self._model_device())

    def _answer_tinyllava(self, request: VideoRequest, frames: list[Any], prompt: str) -> ModelAnswer:
        import torch

        message = self._tiny["Message"]()
        message.add_message(f"<image>\n{prompt}")
        encoded = self._tiny_text_preprocess(message.messages, mode="eval")
        input_ids = encoded["input_ids"].unsqueeze(0).to(self.device)
        video = torch.stack([self._tiny_video_preprocess(frame) for frame in frames]).unsqueeze(0)
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=None,
                video=video,
                do_sample=False,
                temperature=0.0,
                num_beams=1,
                pad_token_id=self.tokenizer.pad_token_id,
                max_new_tokens=self.max_new_tokens,
                use_cache=True,
            )
        output = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        return ModelAnswer(text=output, metadata={"model": self.model_path, "backend": "tinyllava", "num_frames": len(frames)})

    def answer(
        self,
        request: VideoRequest,
        candidate: Optional[ClipCandidate] = None,
        evidence: Optional[str] = None,
        draft: Optional[str] = None,
    ) -> ModelAnswer:
        self._load_backend()
        frames = self._frames(request, candidate)
        if not frames:
            raise ValueError(f"No frames available for request {request.qid}")
        prompt = f"Question: {request.question}"
        choices = request.metadata.get("choices") or {}
        if choices:
            prompt += "\nOptions:\n" + "\n".join(f"{key}: {choices[key]}" for key in ("A", "B", "C", "D") if key in choices)
            prompt += "\nReturn only the letter of the correct option (A, B, C, or D)."
        if evidence:
            prompt += f"\nIndependent medical inspection evidence:\n{evidence}\nRe-evaluate the answer."
        if draft:
            prompt += f"\nPrevious draft:\n{draft}\nReturn a corrected final answer."
        if self.backend == "tinyllava":
            return self._answer_tinyllava(request, frames, prompt)
        inputs = self._build_inputs(request, frames, prompt)

        import torch

        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
        prompt_length = inputs.input_ids.shape[1] if hasattr(inputs, "input_ids") else 0
        if prompt_length:
            generated_ids = generated_ids[:, prompt_length:]
        text = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()
        return ModelAnswer(
            text=text,
            metadata={
                "model": self.model_path,
                "backend": self.backend,
                "num_frames": len(frames),
                "candidate": candidate.clip_id if candidate is not None else None,
            },
        )


class SurgPubVideoAnswerModel(HuggingFaceVideoAnswerModel):
    """Named entry point for a SurgLLaVA-Video/SurgPub controller checkpoint."""

    pass
