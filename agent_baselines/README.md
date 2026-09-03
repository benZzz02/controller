# No-training controller runtime

`LangGraphSurgicalController` is the recommended orchestration layer for the
frozen-model baselines. It does not learn a policy: LangGraph provides the
explicit state machine, conditional routing, bounded tool call, checkpoint
identity, and node-level streaming needed for reproducible agent experiments.
`SurgicalController` remains as a dependency-light reference implementation for
unit tests and exact lightweight ablations.

Inject three frozen adapters:

```text
Retriever   = released SurgCLIP-B from ../SurgLaVi
AnswerModel = SurgLLaVA-Video-3B or another HF-compatible video VLM
Inspector   = released MedGRPO 7B inference wrapper
```

The four modes correspond to the baseline matrix:

```python
LangGraphSurgicalController(answer_model, mode="direct")
LangGraphSurgicalController(answer_model, retriever, mode="retrieve")
LangGraphSurgicalController(answer_model, retriever, inspector, mode="inspect")
LangGraphSurgicalController(answer_model, retriever, inspector, mode="adaptive")
```

The adaptive graph is deliberately constrained:

```text
retrieve -> draft -> rule gate -> [stop | inspect once -> final rethink]
```

The actual model-serving code belongs in adapters, not in the runtime. The
controller records retrieval results, gate reasons, tool calls, answer stages,
checkpoint thread id, and latency in a common trace. Raw video frames are kept
outside graph state; checkpoints contain only serializable requests and clip
metadata.

## Installation

On a GPU machine, install the runtime dependencies from:

```bash
pip install -r requirements-baseline.txt
```

The default CLI runtime is LangGraph. Use `--runtime lightweight` when you
need the pure-Python reference path or when checking the controller without
installing the graph runtime.

The adapter adds the local `SurgLaVi/src/surgclip` package to `sys.path` and
therefore does not require installing the whole SurgLaVi repository. The
released SurgCLIP weights are downloaded on the first real retrieval call.

## SurgPub-Video smoke test

First validate dataset paths and fields without loading any model:

```bash
python -m agent_baselines.run_baseline \
  --data /path/to/surgpub_test.json \
  --data-root /path/to/surgpub_media \
  --output /tmp/surgpub.normalized.jsonl \
  --dry-run
```

For the two-GPU frozen loop (GPU 0: SurgCLIP + controller; GPU 1: MedGRPO):

```bash
python -m agent_baselines.run_baseline \
  --data /path/to/surgpub_test.json \
  --data-root /path/to/surgpub_media \
  --output /path/to/results/adaptive.jsonl \
  --mode adaptive \
  --runtime langgraph \
  --controller-model /path/to/SurgLLaVA-Video-3B \
  --inspector-model /path/to/uAI-NEXUS-MedVLM-1.0a-7B-RL \
  --device-controller cuda:0 \
  --device-inspector cuda:1 \
  --inspector-4bit \
  --num-frames 16 \
  --window-sec 8 \
  --stride-sec 4 \
  --limit 20
```

`--mode inspect` only needs the MedGRPO checkpoint and answers from the
retrieved Top-1 clip. `--mode retrieve` measures the controller after
SurgCLIP retrieval without MedGRPO. `--mode adaptive` uses the fixed rule gate
and makes at most one inspector call per question. The output trace includes
the selected graph path and per-node latency, so the same JSONL format can be
used for lightweight-vs-LangGraph equivalence checks.

## Why LangGraph instead of the neighboring VideoAgent repository?

The neighboring `VideoAgent` code is useful as a generic ReAct/LangGraph
reference, but it assumes API-backed agents and a broader tool manager. This
baseline needs local frozen checkpoints, two explicit GPU placements, fixed
retrieval windows, and a matched one-call inspection budget. The local
LangGraph runtime keeps those scientific constraints explicit while leaving a
future learned policy free to replace only the gate/router.

The generic controller model adapter is ready for a checkpoint registered by
Transformers. The published SurgLLaVA-Video model is based on TinyLLaVA-Video;
if its checkpoint is not registered with `AutoModel`, only the model-specific
loader in `hf_video_answer.py` needs to be replaced. Dataset normalization,
SurgCLIP retrieval, MedGRPO inspection, traces, and evaluation inputs remain
unchanged.
