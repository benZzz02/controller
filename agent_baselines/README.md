# No-training controller

`SurgicalController` is the orchestration layer for frozen-model baselines. It
does not learn a policy and does not require LangGraph, `verl`, or an external
agent library.

Inject three frozen adapters:

```text
Retriever   = released SurgCLIP-B from ../SurgLaVi
AnswerModel = SurgLLaVA-Video-3B or another HF-compatible video VLM
Inspector   = released MedGRPO 7B inference wrapper
```

The four modes correspond to the baseline matrix:

```python
SurgicalController(answer_model, mode="direct")
SurgicalController(answer_model, retriever, mode="retrieve")
SurgicalController(answer_model, retriever, inspector, mode="inspect")
SurgicalController(answer_model, retriever, inspector, mode="adaptive")
```

The actual model-serving code belongs in adapters, not in the controller. The
controller records retrieval results, gate reasons, tool calls, answer stages,
and latency in a common trace.

## Installation

The controller itself has no heavyweight import-time dependency. On a GPU
machine, install the runtime dependencies from:

```bash
pip install -r requirements-baseline.txt
```

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
and makes at most one inspector call per question.

The generic controller model adapter is ready for a checkpoint registered by
Transformers. The published SurgLLaVA-Video model is based on TinyLLaVA-Video;
if its checkpoint is not registered with `AutoModel`, only the model-specific
loader in `hf_video_answer.py` needs to be replaced. Dataset normalization,
SurgCLIP retrieval, MedGRPO inspection, traces, and evaluation inputs remain
unchanged.
