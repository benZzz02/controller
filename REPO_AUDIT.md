# Surgical Video-Agent Repository Audit

Date: 2026-09-03

## Repositories cloned or already available

| Repository | Local path | Intended role | Assessment |
|---|---|---|---|
| VideoAgent | `../VideoAgent` | ReAct/LangGraph orchestration and tool interface examples | Reuse the interface and logging ideas; replace generic tools with surgical retrieval and inspection adapters. |
| LongVT | `../LongVT` | Native video tool-calling, SFT/RL data format, and `verl`-based RL scaffold | Most useful RL reference. The released recipe is tied to Qwen2.5-VL-7B, so it needs a model/data adapter for a 3B SurgLLaVA-Video controller. |
| ParaVT | `../ParaVT` | Parallel tool calling and multi-agent RL reference | Useful as a later ablation or extension; not needed for the first sequential agent. |
| MedGRPO-Code | `../MedGRPO-Code` | Frozen medical video-language inspector inference | Directly usable as a separate inference service. Its JSON input format can be generated from a retrieved surgical clip. |
| orena-focus | `../orena-focus` | Official FOCUS loaders, preprocessing, answer parsing, and evaluator | Directly usable for FOCUS evaluation. Prefer its `Evaluator` over forcing FOCUS into a generic evaluator. |
| SurgLaVi | `../SurgLaVi` | Public surgical CLIP-style retrieval baseline and feature extraction | Use its released SurgCLIP as the primary frozen retriever for reproducibility; keep SurgAlign as a retrieval ablation and do not download the large SurgLaVi video corpus initially. |
| lmms-engine | `../lmms-engine` | General multimodal SFT engine | Keep as an optional SFT backend. It does not directly support the SurgPub-Video 3B model without a custom model adapter. |
| lmms-eval | `../lmms-eval` | General VLM benchmark runner | Useful for generic baselines, but it has no native FOCUS task adapter in this checkout. |

The existing SurgAlign code under `/Users/ben/SurgAlign/clip_tfnc` is retained only as an optional retrieval ablation and should not be duplicated. No official standalone SurgPub-Video/SurgLLaVA-Video repository was found in the current repository search; obtain its released checkpoint/data separately if needed.

## Recommended implementation order

1. Use the released SurgCLIP model from SurgLaVi as the primary frozen
   retriever; keep the existing SurgAlign implementation only as an optional
   retrieval ablation.
2. Implemented in `agent_baselines/`: map `question + video_path/frame_paths +
   start/end` to the common controller schema and the MedGRPO JSON format,
   while returning structured text evidence.
3. Reuse the `VideoAgent` tool schema for `retrieve_clip` and `inspect_clip`, but keep the first policy loop sequential and bounded.
4. Convert SurgPub-Video VQA records into LongVT-style tool-call trajectories for cold-start SFT.
5. Port only the relevant LongVT `verl`/agent-loop pieces for RL. Start with the Qwen2.5-VL-compatible path; add a custom SurgLLaVA-Video model adapter only after the tool policy is validated.
6. Evaluate FOCUS through `orena-focus` directly, and use the same response records for cost/latency analysis.

## First experiment matrix

- Controller only: SurgCLIP retrieval + 3B controller, no MedGRPO.
- Tool always on: retrieve, then inspect for every question.
- Adaptive tool: inspect only when confidence is low or retrieved evidence conflicts.
- RL ablation: cold-start SFT versus agent RL with the same model, data split, visual-token budget, and maximum number of tool calls.

The first implementation should use one retrieved clip and at most one MedGRPO call per trajectory. Parallel tool calling from ParaVT can be evaluated later only if sequential tool use gives a clear gain.

## Important compatibility boundary

The repositories do not form one drop-in stack. The main engineering task is the adapter layer:

```text
FOCUS/SurgPub record
        -> common VideoRequest
        -> SurgCLIP retrieve_clip
        -> MedGRPO inspect_clip (optional)
        -> controller response + structured trace
        -> native dataset evaluator
```

The source-only audit completed successfully. Model checkpoints, datasets, and Python environments were intentionally not downloaded.

## No-training baseline protocol

All models remain frozen. Only deterministic preprocessing, prompt templates, retrieval, and tool routing are used.

| Variant | Pipeline | Purpose |
|---|---|---|
| B0 Direct | Uniform/global video frames -> 3B Controller -> answer | Measures the base VLM without tools. |
| B1 Retrieve | SurgCLIP Top-1 clip -> 3B Controller -> answer | Measures retrieval benefit at a matched visual budget. |
| B2 Inspect | SurgCLIP Top-1 clip -> frozen MedGRPO -> answer | Measures the value of the medical inspector, without an agent controller. |
| B3 Rule-Agent | Top-1 clip -> Controller; call MedGRPO only on a fixed low-confidence, invalid-format, or temporal-evidence rule; Controller produces the final answer | The no-training approximation of the proposed loop. |
| B4 Oracle clip | Ground-truth/annotated temporal window -> Controller | Upper bound for retrieval quality when temporal labels exist. |

For every sample, save `qid`, video ID, question, prediction, retrieved windows and scores, tool-call flag, frames/tokens, latency, and peak memory. Evaluate the answer with the native dataset evaluator. On FOCUS, use `orena-focus`'s `Evaluator`; do not replace its format-aware scoring with a generic LLM judge.

The decision rule is simple: B1 must beat B0 before RL is justified; B3 must beat B1 under the same visual-token budget before claiming an Agent/tool-policy gain. If only B2 wins, the gain is from the stronger inspector model rather than from agentic control.

Because SurgLaVi is itself trained on public surgical videos, audit video/source overlap with every SurgPub-Video and FOCUS split before reporting cross-dataset results. If overlap cannot be ruled out, report SurgCLIP as a potentially contaminated retrieval baseline and include a non-surgical OpenAI-CLIP or SurgAlign comparison.
