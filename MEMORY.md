# Project Memory – Read This First

Snapshot: **2026-08-14 14:46 IST**. The repository HEAD is `d7127be` on
`main`. The DVC/MLflow run and the development-notebook run have completed.
The worktree remains intentionally dirty with fresh model, metric, notebook,
documentation, and evidence artefacts. Verify volatile state before acting.

## Project

- BITS Pilani WILP M.Tech AIML, AIMLCZG523 MLOps Assignment 2.
- Student: Hamza Aziz, ID `2024AC05133`, semester `S2-25`.
- Use case: Cats vs Dogs binary image classification using Kaggle dataset
  `tongpython/cat-and-dog`.
- Remote: `git@github.com:hamza-aziz-ai/MLOPs-Cats-Dogs-Pipeline.git`.
- Milestones M1-M5 cover experiment development, DVC/MLflow, CI, deployment,
  monitoring, and submission evidence.
- Video upload and LMS submission are the student's responsibility and remain
  out of scope for the agent.

## Verified current state

### Git

- Inspected HEAD: `d7127be0816664aaeac0f12fce20e7e22deb8a57`
  (`docs: refresh cross-agent project handoff`).
- Do not push until the user explicitly asks for that specific message. A push
  starts the full CI/CD workflow.
- Preserve the user's existing `Pipfile`, `Pipfile.lock`, `scripts/train.py`,
  `AGENTS.md`, and `CLAUDE.md` changes plus all fresh execution artefacts.
- `dev/` remains local export material and must not be committed unless the
  user explicitly changes that policy.
- `notebooks/01_model_development.pdf` is a verified 57-page A4 export and is
  currently untracked.
- `README.md`, `SUBMISSION_REPORT.md`, `SUBMISSION_CHECKLIST.md`,
  `VIDEO_DEMO_SCRIPT.md`, `MEMORY.md`, `AGENTS.md`, and `CLAUDE.md` contain the
  current local documentation refresh and remain uncommitted.
- The requested Graphify update and cleanup completed after the documentation
  refresh. The final graph has 549 nodes, 674 links, and 62 named communities;
  `graph.json`, `GRAPH_REPORT.md`, `graph.html`, and the existing Obsidian vault
  were regenerated. `scripts/graphify_clean.py` removed one source-less built-in
  exception shadow node during the refresh and reported the corrected final
  graph clean on the last pass.
- The semantic update also refreshed the Markdown structural tier, removing the
  obsolete `MEMORY.md` heading node that claimed a DVC run was still active.
  The final benchmark reports 43.6x average query-context reduction.
- Installed Graphify is 0.9.41 while the local skill text is 0.9.42. The update
  completed successfully with the warning recorded; no package upgrade was
  attempted.

### Completed DVC/MLflow run

- Experiment ID: `778441259214907561`
- Run ID: `6b712432dc584f9d9bdadd33d0196536`
- Status: finished (`status: 3`)
- Architecture: `CatDogResNet50`
- Device: CUDA
- Epochs completed: 70
- Training duration: `12576.0644841` seconds (`03:29:36.064`)
- Test accuracy: `0.9450549451`
- Weighted precision: `0.9450706068`
- Weighted recall: `0.9450549451`
- Weighted F1: `0.9450542870`
- Recorded checkpoint SHA-256 at run completion:
  `9f118d1f51bffda2079313dd1b4cef0b9f2b8c9e67108e2582d2ea40ae78a128`
- The legacy run name remains `baseline-cnn-1.0.0`; the logged architecture
  parameter is authoritative.

### Completed notebook run

- Notebook: `notebooks/01_model_development.ipynb`
- Structure: 87 cells, 46 code cells, all 46 executed, 260 outputs, zero error
  outputs, valid nbformat 4.5.
- Epochs completed: 74; early stopping triggered at epoch 74.
- Best validation accuracy: `0.9394697349`.
- Test accuracy: `0.9390609391`.
- Weighted precision/recall/F1:
  `0.9392018668` / `0.9390609391` / `0.9390554650`.
- Mean average precision: `0.9867129639`; mean ROC-AUC: `0.9865239521`.
- Confusion matrix: `[[465, 35], [26, 475]]` for true cats/dogs by predicted
  cats/dogs.
- The notebook wrote its best raw state dict to the shared checkpoint after
  the DVC run, producing SHA-256 `292fa56f8a0660c2decb32601bb5ca292abb1957436605230161daac233713e7`.
  Because a raw `state_dict()` save carries no `metrics`/`class_names`
  wrapper, `load_checkpoint()` fell back to an empty `metrics: {}`, so
  `/model/info` reported no evaluation numbers while this checkpoint was live.
- **Resolved 2026-08-14**: the notebook checkpoint has been superseded. The
  DVC run's checkpoint was recovered byte-for-byte from
  `mlruns/778441259214907561/6b712432dc584f9d9bdadd33d0196536/artifacts/model/resnet50_baseline.pt`
  (verified SHA-256 match to `9f118d1f...` below) and copied back to
  `models/resnet50_baseline.pt`. `dvc status` now reports "Data and pipelines
  are up to date." Local `uvicorn` restart confirmed: `/model/info` sha256
  `9f118d1f...`, populated `evaluation_metrics` (0.945 acc), and correct
  high-confidence predictions on cat/dog samples.
- **Decision: DVC is the sole production checkpoint owner going forward.**
  The notebook is experimental only and must never write
  `models/resnet50_baseline.pt`. Fixed in `notebooks/01_model_development.ipynb`
  cell `ea667b8a`: `model_checkpoint['filepath']` now points to
  `./artifacts/resnet50/resnet50_notebook_checkpoint.pt` instead of
  `./models/resnet50_baseline.pt`. Both the per-epoch best-checkpoint save and
  the post-training reload-for-eval route through this one config value, so
  the notebook can no longer touch the production checkpoint.

### Current tests

- Verified on 2026-08-14: 9 tests passed in 12.22 seconds.
- Coverage: 86% (355 statements, 51 missed).

### Current configuration and prepared data

`params.yaml` is the source of truth for both DVC training and notebook values:

| Setting                      | Value                    |
|------------------------------|--------------------------|
| Image size                   | 224                      |
| Train/validation/test ratios | 0.70 / 0.20 / 0.10       |
| Maximum images per class     | 5000                     |
| Batch size                   | 32                       |
| Epoch budget                 | 100                      |
| Learning rate                | 0.001                    |
| Weight decay                 | 0.0001                   |
| Workers                      | 0                        |
| Early-stopping patience      | 20 epochs                |
| Early-stopping minimum delta | 0.01 validation accuracy |
| Seed                         | 42                       |

Verified prepared split counts:

- Train: 6,999 (`cats=3500`, `dogs=3499`)
- Validation: 1,999 (`cats=1000`, `dogs=999`)
- Test: 1,001 (`cats=500`, `dogs=501`)
- Total: 9,999 images. The preparation report records 29 duplicate-content
  images dropped and zero corrupt/skipped images before the capped split was
  finalized.

### Architecture and data flow

The original small `CatDogCNN` has been fully replaced by a randomly
initialized, from-scratch `CatDogResNet50`. It is not pretrained and does not
use transfer learning.

```text
Kaggle download
  -> scripts/prepare_data.py (canonicalize, content-deduplicate, split)
  -> data/processed/{train,validation,test}/{cats,dogs}
  -> scripts/train.py + src/cats_dogs_mlops/model.py
  -> models/resnet50_baseline.pt + metrics + MLflow + DVC artefacts
  -> src/cats_dogs_mlops/api.py
  -> Docker Compose deployment + Prometheus monitoring
```

`scripts/train.py` tracks every epoch in MLflow, stops after 20 epochs without
an improvement greater than 0.01, restores the best in-memory state, evaluates
that state on the test split, then writes the checkpoint, metrics, plots, and
metadata. Those final files do not exist during most of the training loop.

### Notebook

`notebooks/01_model_development.ipynb` is the single M1 model-development
notebook and implements the same from-scratch ResNet-50. It is hand-authored,
uses `params.yaml`, and deliberately has `RUN_TRAINING = True`.

Notebook-generated outputs are separate from DVC training outputs:

- Notebook metrics, JSON/text reports, plots, and prediction images:
  `artifacts/resnet50/`
- DVC training plots and metadata: root `artifacts/`
- Shared checkpoint path: `models/resnet50_baseline.pt`

Do not execute the notebook and DVC training concurrently because both can
write the shared checkpoint. A full notebook execution can train up to 100
epochs. When editing the notebook, preserve its outputs unless the user asks to
clear them, validate with `nbformat`, compile every code cell, and stage only
the intended source hunks if the live execution output is also dirty.

### Evidence and deployment

- `README.md`, `SUBMISSION_REPORT.md`, `SUBMISSION_CHECKLIST.md`, and
  `VIDEO_DEMO_SCRIPT.md` now distinguish the completed DVC run from the
  completed notebook run and use fresh local M1/test values.
- `AGENTS.md` and `CLAUDE.md` are synchronized and now require separate
  DVC/notebook evidence lineages, explicit final-checkpoint selection, and
  Graphify cleanup after each successful refresh.
- Existing screenshots, CI run URL, GHCR digest, remote deployment URL,
  post-deployment metrics, final commit/tag, and submission ZIP belong to the
  retired evidence set and are not current-checkpoint proof.
- The previous self-hosted GitHub Actions runner was removed. A fresh WSL2
  Ubuntu runner is required before the deployment job can run again.
- The public remote contains an older state than local `main`; do not infer the
  local architecture or evidence from GitHub until the user authorizes a push.

## Next steps

1. ~~Decide checkpoint lineage~~ — **done**: DVC checkpoint (`9f118d1f...`,
   0.945 test accuracy) is final; restored from MLflow, verified live via
   local `uvicorn`. Notebook write-protected against
   `models/resnet50_baseline.pt` (see above).
2. Review and commit only intended source/execution/evidence files; exclude
   `dev/`, temporary files, and unrelated changes.
3. Recapture M1 screenshots for the 9,999-image split, current DVC status,
   87/46-cell notebook execution, MLflow run, and 9-test/86%-coverage result.
4. ~~Rebuild and test Docker locally~~ — **done**: Compose containers healthy
   (`cats-dogs-mlops-api-1`, `cats-dogs-mlops-prometheus-1`), `/model/info`
   sha256 confirmed `9f118d1f...`, `post_deployment_evaluation.py` against the
   local deployment: 20/20 correct, F1 1.0, mean latency 76.15ms, p95
   121.26ms.
5. Push only after an explicit user request. Before the push, register a fresh
   trusted self-hosted Linux runner.
6. Capture fresh CI/GHCR digest, deployment, health/predict/model-info,
   Compose, Prometheus, smoke, and post-deployment-evaluation evidence.
7. Rebuild the submission ZIP and record its new file count, size, SHA-256, and
   contents screenshot.
8. Record the final video under five minutes and complete LMS submission.

## Durable decisions and lessons

- The full pipeline uses the student's from-scratch ResNet-50, not a small
  custom CNN and not pretrained torchvision weight.
- `Pipfile`/`Pipfile.lock` serve local CUDA development on Python 3.14;
  `requirements.txt`/`requirements-api.txt` serve Python 3.11 CI/deployment.
  They are intentionally different dependency surfaces.
- Prepared splits are content-hash deduplicated to prevent leakage across
  train, validation, and test.
- DVC/MLflow and notebook executions are separate evidence lineages. DVC is
  the sole production checkpoint owner; the notebook is experimental only and
  is now write-protected (writes to `artifacts/resnet50/`, not
  `models/resnet50_baseline.pt`). Never mix one run's metrics or hash with the
  other's checkpoint.
- A raw `torch.save(model.state_dict(), path)` checkpoint (no `metrics`/
  `class_names` wrapper) loads fine via `load_checkpoint()`'s notebook-format
  branch, but `/model/info` reports empty `evaluation_metrics` — by design,
  not a bug. Only `save_checkpoint()`-written checkpoints (DVC's
  `scripts/train.py`) embed metrics.
- Git-tracked DVC outputs use `cache: false`; otherwise DVC rejects files also
  tracked by SCM.
- Notebook prose and comments describe the current PyTorch implementation only.
  Do not reintroduce Keras migration commentary, bird-dataset references, or
  stale API names.
- Verify facts with commands and artefacts. Never fabricate metrics, hashes,
  run IDs, digests, screenshots, or deployment status.
- After a requested Graphify refresh, run `scripts/graphify_clean.py` to remove
  source-less built-in exception shadow nodes before treating the graph as
  authoritative.
- Use new commits only. Never amend or push without the user's instruction.
