# SurgVerify: CVPR 2027 working draft

This is a provisional CVPR-style paper draft for the surgical-video Agent
direction discussed in the surgical-video agent planning task. The current
research question is whether a minimal Agent can decide when retrieved surgical
evidence is enough and whether one targeted second look is worth its visual
cost under a fixed budget.

The current implementation is intentionally small: released SurgCLIP retrieval,
a frozen SurgLLaVA-Video/SurgPub-compatible controller, a frozen MedGRPO
inspector, and a validation-calibrated deterministic gate. It makes one first
look, then either stops or verifies once using the retrieved temporal window.
The controller is model-agnostic so the same traces can later support learned
agent-RL policies.

The draft intentionally uses `TODO` markers for every unverified experiment
number, exact split count, and implementation choice. No result in this
directory should be presented as measured until it is replaced by a logged
run and its evaluation manifest.

The formatting files were copied from the latest official CVPR author kit
available on 2026-09-02. A CVPR 2027-specific kit was not yet published; see
`../cvpr2027_template/CVPR2027_STATUS.md`.

## Build

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

The current file is in review mode and uses anonymous authors.

## Required evidence before submission

- Implement the common video interface and fixed visual-token accounting.
- Run the frozen SurgCLIP + SurgPub-Video + MedGRPO controller in
  `agent_baselines/`.
- Lock the first-look thresholds and verify-once policy before test evaluation.
- Run the SurgPub-Video smoke test and the full controlled comparison.
- Audit train/test media overlap before making any cross-dataset claim.
- Replace every `TODO` result with a saved prediction file, metric log, and
  configuration manifest.
- Re-check the official CVPR 2027 author kit and the final double-blind rules.
