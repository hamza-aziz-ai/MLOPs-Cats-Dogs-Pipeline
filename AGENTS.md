# Repository Instructions

Keep this file and `CLAUDE.md` synchronized. They provide stable operating
rules; `MEMORY.md` contains the volatile Git, training, evidence, and deployment
state.

## Start here

1. Read `MEMORY.md` completely.
2. Verify its snapshot with `git log -1`, `git status`, and process inspection.
3. If DVC training is active, do not start another DVC stage, training script,
   or full notebook execution. Wait for the existing process to finish.

## Architecture

The production model is `CatDogResNet50`, a randomly initialized ResNet-50
implemented from scratch in `src/cats_dogs_mlops/model.py`. It is not pretrained
and does not use transfer learning.

```text
scripts/download_data.py
  -> scripts/prepare_data.py
  -> data/processed/{train,validation,test}/{cats,dogs}
  -> scripts/train.py + src/cats_dogs_mlops/model.py
  -> models/resnet50_baseline.pt
  -> src/cats_dogs_mlops/inference.py
  -> src/cats_dogs_mlops/api.py
  -> Docker Compose + Prometheus
```

Key files:

- `params.yaml`: DVC and notebook configuration source of truth.
- `dvc.yaml`: `download -> prepare -> train` reproducible pipeline.
- `notebooks/01_model_development.ipynb`: single M1 development notebook and
  theoretical/code walkthrough for the same ResNet-50.
- `src/cats_dogs_mlops/preprocessing.py`: shared train/evaluation transforms.
- `scripts/post_deployment_evaluation.py`: deployed-service evaluation.
- `.github/workflows/ci-cd.yml`: test, image publish, and self-hosted deploy.

## Environment and commands

From a normal shell, use Pipenv:

```powershell
pipenv run pytest
pipenv run dvc repro
pipenv run dvc status
pipenv run python scripts/train.py --help
pipenv run jupyter nbconvert --to notebook --execute --inplace notebooks/01_model_development.ipynb
```

If the prompt already begins with `(MLOps-Cats-Dogs-Pipeline)`, the environment
is active. Run the inner command directly:

```powershell
pytest
dvc repro
dvc status
python scripts/train.py --help
```

Do not nest `pipenv run` inside the active environment; that produces Pipenv's
Courtesy Notice. Bare system Python outside the environment lacks project
dependencies.

Useful targeted DVC commands:

```powershell
dvc repro prepare   # rebuild only processed splits when data parameters change
dvc repro train     # train from prepared data
dvc status
```

Never run these concurrently with an existing DVC/training/notebook process.

## Training configuration and early stopping

Read values from `params.yaml`; do not hardcode notebook-only alternatives.
The current configuration uses 224x224 images, a 70/20/10 split, up to 5,000
images per class, batch size 32, and a 100-epoch budget.

Both `scripts/train.py` and the notebook use early stopping with
`training.patience` and `training.min_delta` (currently 20 and 0.01). The script
restores the best validation-accuracy state before test evaluation and final
checkpoint creation.

## Output ownership and concurrency

- Shared model checkpoint: `models/resnet50_baseline.pt`.
- DVC train outputs: `metrics/training_metrics.json` and root
  `artifacts/{loss_curves.png,confusion_matrix.png,model_metadata.json}`.
- Notebook-only reports, metrics, plots, and prediction images:
  `artifacts/resnet50/`.
- MLflow file store: `mlruns/`.

DVC may delete its declared outputs before rebuilding them. While a train stage
is running, missing checkpoint/metric/plot files and a modified `dvc.lock` are
transient. Do not restore or commit them mid-run.

The notebook and DVC training share the checkpoint path. Never execute them in
parallel.

## Dependency files are intentionally separate

- `Pipfile`/`Pipfile.lock`: local CUDA development on Python 3.14, including a
  dedicated PyTorch package index.
- `requirements.txt`/`requirements-api.txt`: Python 3.11 CI and Docker builds
  from plain PyPI.

Do not mechanically unify these files. Verify changed CI/deployment pins have
Python 3.11 manylinux wheels on plain PyPI.

## DVC and Git-tracked outputs

Model checkpoints, plots, metrics, and selected MLflow evidence are committed
to Git for assignment visibility. Every corresponding `dvc.yaml` output must
use `cache: false`; otherwise DVC rejects the file as already tracked by SCM.

After a run completes:

1. Confirm the process exit code.
2. Inspect final metrics, MLflow status, checkpoint hash, and artefact timestamps.
3. Run `dvc status`.
4. Run `pytest`.
5. Review `git status` before staging anything.

Never commit `dev/`, `.ipynb_checkpoints/`, temporary lock files, partial MLflow
runs, or unrelated notebook execution state.

## Notebook editing

`notebooks/01_model_development.ipynb` is hand-authored and deliberately has
`RUN_TRAINING = True`. A top-to-bottom execution can therefore train for up to
100 epochs. Do not execute it merely to validate a source-only edit.

For scripted changes, use `nbformat`, preserve cell order/outputs unless asked
otherwise, run `nbformat.validate()`, and compile every code cell. Search the
whole notebook after removing or renaming a concept. Do not leave stale Keras,
bird-dataset, retired-CNN, or migration-diary prose behind.

Notebook source and execution output can be dirty in the same JSON file. When
only source should be committed, stage the intended source hunks rather than
committing every notebook output change.

## Tests and static warnings

The suite contains API, inference, preprocessing, and split/deduplication tests.
Run `pytest` after training completes or after ordinary code changes. Keep both
runtime warnings and IDE type inspections clean through real type narrowing or
API corrections, not blanket warning suppression.

## CI/CD and self-hosted deployment

The deployment job uses `runs-on: [self-hosted, linux]`. The previous runner was
removed. Before the next deployment, register a fresh Linux runner (WSL2 Ubuntu
on this machine is the established option) using the current commands shown by
GitHub under **Settings -> Actions -> Runners -> New self-hosted runner**.

On this Windows host, include `--cd ~` in every `wsl` invocation; otherwise
`wsl.exe` can fail while translating the Windows repository path. Docker
Desktop supplies the shared Docker daemon. Remove the runner again after final
deployment/evidence capture.

## Git discipline

- Make new commits only; never amend or rewrite previous work.
- Preserve unrelated user changes in a dirty worktree.
- Commit only the files/hunks belonging to the current request.
- **Never push unless the user explicitly asks in that specific message.**
- A push triggers the full CI/CD workflow, so do not infer push authorization
  from an earlier request.

## graphify

The repository knowledge graph lives under `graphify-out/`.

- For codebase questions, start with `graphify query "<question>"` when
  `graphify-out/graph.json` exists.
- Use `graphify path "<A>" "<B>"` for relationships and
  `graphify explain "<concept>"` for focused context.
- Prefer `graphify-out/wiki/index.md` for broad navigation.
- Read `GRAPH_REPORT.md` only when scoped queries are insufficient.
- After code changes, run `graphify update .`. Documentation-only state updates
  do not require a topology refresh.
