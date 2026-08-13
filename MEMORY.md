# Project Memory - Read This First

Snapshot: **2026-08-14 01:52 IST**. The code state was inspected at `f7757cf`
on `main`; this handoff-documentation update is the next local commit. Verify
the resulting HEAD and all volatile state with read-only commands before acting
because training and Git state can change.

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

- The inspected code commit is `f7757cf`
  (`fix(notebook): resolve remaining inspections`); the handoff commit follows it.
- Local `origin/main` is at `c27a189`.
- Before this handoff commit, local `main` was 28 commits ahead of the locally
  recorded remote ref; after it, the count is 29.
- Do not push until the user explicitly asks in that specific message. A push
  starts the full CI/CD workflow and can take about 20 minutes.
- The current worktree is intentionally dirty because a real DVC training run
  is active. Do not clean, reset, checkout, stage, or commit its outputs while
  the process is running.

### A real DVC training run is active

At this snapshot, PID `113364` (started `2026-08-14 01:45:05 IST`) is running
`dvc repro train`. DVC reports that it owns the write lock for
`metrics/training_metrics.json`. Do not start another DVC stage, training
script, or full notebook execution until this process exits.

Active MLflow run:

- Experiment ID: `778441259214907561`
- Run ID: `6b712432dc584f9d9bdadd33d0196536`
- Status: running (`end_time: null`)
- Architecture parameter: `CatDogResNet50`
- Device: CUDA
- Git commit recorded by MLflow: `f7757cfb62bc431d21918757c4b82e8671dfe0f1`
- Latest completed epoch observed at the snapshot: epoch 3
  (`train_accuracy=0.5835`, `validation_accuracy=0.6108`). These are progress
  values, **not final model metrics**.
- The run name is still `baseline-cnn-1.0.0`; the logged `architecture`
  parameter is authoritative. Do not rename a live run.

DVC removes stage outputs before rebuilding them. Therefore these Git status
entries are expected while training is active, not evidence of accidental data
loss:

```text
 M Pipfile
 M Pipfile.lock
 D artifacts/confusion_matrix.png
 D artifacts/loss_curves.png
 D artifacts/model_metadata.json
 M dvc.lock
 D metrics/training_metrics.json
 D models/resnet50_baseline.pt
 M notebooks/01_model_development.ipynb
 M scripts/train.py
?? artifacts/resnet50/dataset_samples.png
?? dev/
?? mlruns/778441259214907561/6b712432dc584f9d9bdadd33d0196536/
```

`dev/null/` is an unrelated local hook byproduct and must not be committed.
The notebook modification contains execution state/output and must be reviewed
separately from source edits. `dataset_samples.png` is a notebook artefact.
The unstaged `Pipfile`, `Pipfile.lock`, and `scripts/train.py` changes add
`tqdm` per-batch progress bars. They were not created or staged by this handoff
update. The active run is using those uncommitted presentation-only changes,
although MLflow records the nearest Git commit (`f7757cf`); review, test, and
commit them separately after training finishes.

### Current configuration and prepared data

`params.yaml` is the source of truth for both DVC training and notebook values:

| Setting | Value |
|---|---:|
| Image size | 224 |
| Train/validation/test ratios | 0.70 / 0.20 / 0.10 |
| Maximum images per class | 5000 |
| Batch size | 32 |
| Epoch budget | 100 |
| Learning rate | 0.001 |
| Weight decay | 0.0001 |
| Workers | 0 |
| Early-stopping patience | 20 epochs |
| Early-stopping minimum delta | 0.01 validation accuracy |
| Seed | 42 |

Verified prepared split counts:

- Train: 6,999 (`cats=3500`, `dogs=3499`)
- Validation: 1,999 (`cats=1000`, `dogs=999`)
- Test: 1,001 (`cats=500`, `dogs=501`)
- Total: 9,999 images. Content-hash deduplication intentionally removed a
  duplicate, so the total is one below 10,000.

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
the intended source hunks if live execution output is also dirty.

### Evidence and deployment

- `README.md`, `SUBMISSION_REPORT.md`, and `SUBMISSION_CHECKLIST.md` still have
  explicit stale-evidence banners. Their metrics, digests, screenshots, and
  deployment claims must not be treated as current until regenerated from this
  ResNet-50 run.
- The previous self-hosted GitHub Actions runner was removed. A new WSL2 Ubuntu
  runner is required before the deploy job can run again.
- The public remote contains an older state than local `main`; do not infer the
  local architecture or evidence from GitHub until the user authorizes a push.
- Last known local tests: 8 passed. They were not rerun for this snapshot to
  avoid disturbing the active training process.

## Next steps after the active training process exits

1. Confirm the DVC command exited successfully; do not infer success merely
   because the PID disappeared.
2. Inspect `metrics/training_metrics.json`, `artifacts/model_metadata.json`,
   the MLflow run status, checkpoint size/hash, and final epoch/early-stopping
   values.
3. Run `dvc status` and confirm whether `dvc.lock` and outputs are coherent.
4. Run `pytest` in the already-active Pipenv environment, or
   `pipenv run pytest` from a normal shell.
5. Review Git status carefully. Do not commit `dev/`, notebook checkpoints, or
   unrelated notebook execution output. Commit DVC outputs only after they are
   verified as the final successful run.
6. Regenerate README/report/checklist numbers and all submission screenshots
   from the ResNet-50 artifacts. Remove stale-evidence notices only when every
   cited value has fresh evidence.
7. Rebuild and test Docker locally. Push only after an explicit user request.
8. If deploying through CI, register a fresh self-hosted runner first, then
   repeat deployed smoke tests, Prometheus checks, and post-deployment
   evaluation.

## Durable decisions and lessons

- The full pipeline uses the student's from-scratch ResNet-50, not a small
  custom CNN and not pretrained torchvision weights.
- `Pipfile`/`Pipfile.lock` serve local CUDA development on Python 3.14;
  `requirements.txt`/`requirements-api.txt` serve Python 3.11 CI/deployment.
  They are intentionally different dependency surfaces.
- Prepared splits are content-hash deduplicated to prevent leakage across
  train, validation, and test.
- Git-tracked DVC outputs use `cache: false`; otherwise DVC rejects files also
  tracked by SCM.
- Notebook prose and comments describe the current PyTorch implementation only.
  Do not reintroduce Keras migration commentary, bird-dataset references, or
  stale API names.
- Verify facts with commands and artifacts. Never fabricate metrics, hashes,
  run IDs, digests, screenshots, or deployment status.
- Use new commits only. Never amend or push without the user's instruction.
