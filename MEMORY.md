# Project Memory — Read This First

Snapshot as of **2026-08-13 21:41 IST**, HEAD `5b31fbe`. Written so any agent
(Claude, ChatGPT, Codex, whatever) picking this repo up cold can act correctly
without re-deriving the last several hours of decisions. If you're that agent:
read this whole file before touching anything, then check "Current state"
below against reality (`git log`, `git status`, `dvc status`) since it drifts.

## What this is

BITS Pilani WILP M.Tech AIML, course AIMLCZG523 (MLOps), Assignment 2.
Student: Hamza Aziz, ID 2024AC05133, semester **S2-25** (confirmed by the student;
the PDF header itself says S1-25 — known, deliberate, documented discrepancy,
not a mistake to "fix"). End-to-end MLOps pipeline: Cats vs Dogs binary image
classification, Kaggle `tongpython/cat-and-dog`, milestones M1 – M5 per
`Problem-Statement.pdf`.

GitHub: `https://github.com/hamza-aziz-ai/MLOPs-Cats-Dogs-Pipeline` (public).

## Current state (the important part)

**The model architecture was swapped mid-session.** The originally graded M1
baseline was a small custom CNN (`CatDogCNN`). It has been **fully replaced**
by a from-scratch ResNet-50 (`CatDogResNet50`, `src/cats_dogs_mlops/model.py`)
— not a pretrained/transfer-learning model, randomly initialized, matching the
PDF's "baseline CNN" scope with a deeper architecture. This was an explicit,
deliberate user decision (not a mistake), made after the original CatDogCNN
pipeline was already fully verified live (see "History" below).

**What's committed locally but NOT pushed:** everything from `c27a189` through
`5b31fbe` (13 commits) is local-only. The last commit actually on
`origin/main` is `5b8e043` (still the old CatDogCNN, fully green CI/CD). **Do
not `git push` without the user explicitly asking** — this has been an
explicit standing instruction for the whole session. Check with the user
before pushing even if they haven't repeated it recently.

**Retraining is genuinely in progress / incomplete**, done informally (the
notebook itself, `RUN_TRAINING = True`, run cell-by-cell), *not* yet via
`dvc repro`:
- `models/resnet50_baseline.pt` exists on disk (~128 MB) but is **untracked**
  and reflects an early/partial run (the best checkpoint saved as of epoch 7,
  `val_accuracy ≈ 0.53` — near-random, expected this early for a from-scratch
  ResNet-50 on only 2,000 images with no pretraining).
- `dvc status` shows the `train` stage as changed (deps + command changed,
  output changed) — `dvc.lock` has **not** been updated to match. Do not trust
  `dvc status`/`dvc repro` output until this is reconciled.
- `notebooks/01_model_development.ipynb` has uncommitted execution-output
  changes (real epoch logs) from this same run.

**Every number in the docs is stale.** `README.md`, `SUBMISSION_REPORT.md`,
and `SUBMISSION_CHECKLIST.md` all carry a **STALE EVIDENCE NOTICE** banner at
the top — MLflow run ID, model SHA-256, GHCR digest, deployed accuracy, and
every screenshot in `docs/screenshots/` still reflect the *retired* CatDogCNN
and its now-torn-down deployment. Do not cite any specific number from those
files as current without regenerating it first.

## What still needs to happen (in order)

1. Let the ResNet-50 training run finish (or decide it's finished / re-run
   deliberately). Confirm a real, final checkpoint exists.
2. Run `pipenv run dvc repro` for real, so `dvc.lock` matches what's on disk.
   Confirm `pipenv run dvc status` reports clean.
3. Run `pipenv run pytest` — last known-good state was 8/8 passing against the
   new architecture (untrained forward pass through the full API), but that
   was before training finished; re-verify.
4. Rebuild the Docker image, `git push`, let CI build+publish to GHCR (**ask
   the user first** — this is the push they've been holding back).
5. Provision a self-hosted runner for the deployment job — the previous one (WSL2
   Ubuntu, Docker Desktop shared daemon) was deliberately deregistered and
   removed after the last deployment demo. See `AGENTS.md` for how to stand
   one up again; it's quick (~5 min) on this machine.
6. Re-run `scripts/post_deployment_evaluation.py` against the redeployed
   service for fresh 20-request numbers.
7. Rebuild the submission ZIP (`git ls-files` → zip, sha256) and retake all 16
   evidence screenshots (`docs/screenshots/E1`–`E16`). `scripts/_terminal_screenshot.py`
   renders CLI output to PNG; GitHub Actions/GHCR/MLflow/Prometheus screenshots
   need a browser tool (Playwright was used this session).
8. Replace the STALE EVIDENCE banners and every number in
   README/SUBMISSION_REPORT/SUBMISSION_CHECKLIST/VIDEO_DEMO_SCRIPT.md with the
   real regenerated values. Don't fabricate numbers — everyone used this
   session came from an actual command run, and was re-verified after each
   change (see "Working style" below).
9. Still outstanding, not automatable: final video recording (<5:00), LMS
   upload. Explicitly out of scope for the agent — the student does these.

## History (why things look the way they do)

- Started as a fully working CatDogCNN pipeline, submitted-once-before state
  (commits up to `bc52b5f`/`54594f6`).
- This session: found and fixed a real train/test split-leakage bug in
  `scripts/prepare_data.py` (Kaggle mirrors ship duplicate photos under
  different filenames; splitting without content-hash dedup let duplicates
  land in both train and test).
- Fixed a pipenv/CI dependency mess: `Pipfile` is the source of truth for
  *local* dev (CUDA torch via a separate `downloadpytorch` index), while
  `requirements.txt`/`requirements-api.txt` are the source of truth for
  *CI/deployment* (must resolve on plain PyPI + Python 3.11 — verified
  directly, not assumed). These legitimately diverge; don't try to unify them.
- Built a from-scratch ResNet-50 PyTorch port from the student's own original
  Keras/TensorFlow notebook (`ResNet-Implementation.ipynb`, since deleted),
  fixing real bugs in the student's own migration attempt along the way
  (missing imports, double-softmax, model never moved to `.to(device)`, Keras
  generators fed to a manual PyTorch loop that would hang forever).
- Got a full live CI/CD cycle green end-to-end against the *old* CatDogCNN:
  pushed to GitHub, registered a self-hosted runner (WSL2 Ubuntu), verified
  real GHCR publish + really remotely deploy + real `/health`/`/predict` against
  the deployed container + real Prometheus target `up`. That state is what's
  still on `origin/main` (`5b8e043`).
- User then decided to make ResNet-50 the actual served model (not a
  companion notebook) — this triggered the architecture swap described above,
  which invalidated all the just-verified live evidence. That's intentional
  and accepted; the docs' STALE EVIDENCE banners reflect it honestly rather
  than either lying or reverting the decision.
- Repo clean-up pass: removed a superseded MLflow run, empty default
  experiment, unused notebook images (loaded from external GitHub URLs, not
  local), a redundant architecture-only demo notebook, and CI byproducts that
  got committed by accident. Renamed `model/` → `artifacts/resnet50/` (was a
  one-letter collision with `models/`, the real checkpoint dir).
- Extensive notebook-content pass: the ResNet-50 notebook was originally a
  half-migrated Keras→PyTorch mess (see git log for the specific bugs found
  and fixed). After becoming the primary M1 notebook, went through several
  rounds of removing leftover Keras-API references in Markdown *and* code
  comments (`ImageDataGenerator`, `flow_from_directory`, `model.compile`,
  `model.fit_generator`, `model.load_weights`, `model.predict_generator`,
  `model.predict()`, "the original migration," "used to hang," etc.) — the
  rules that emerged: **describe what the current PyTorch code does, on its
  own terms, with zero reference to the Keras method it replaced.** Apply
  that rule to any future edits in this notebook.

## Working style established this session (follow it)

- **Verify, don't assume it.** Every claim in the docs was backed by an actual
  command run (real `curl`, real `pytest`, real `dvc status`), not inferred.
  When something couldn't be verified live (e.g. Python-3.11 install
  resolution without a local 3.11 interpreter), that limitation was stated
  explicitly rather than glossed over.
- **Root-cause fixes, not workarounds.** E.g. the split-leakage bug was fixed
  in `prepare_data.py` itself, not patched around; the DVC/git double-tracking
  conflict was fixed with `cache: false` in `dvc.yaml`, not by abandoning DVC.
- **No stale claims left standing.** Every time a change invalidated a
  previously stated fact (a metric, a digest, a comment describing old
  behaviour), it got corrected in the same pass, not left to rot — including
  going back through old notebook comments the user had to point out one by
  one. Do the same: if you change something that a comment/doc describes,
  update the comment/doc too, don't leave it for someone else to notice.
- **Commit locally, push only when asked.** Standing instruction, still in
  effect. Local commits are affordable and reversible; pushes are not (they
  trigger a real, ~20-minute, resource-consuming CI/CD cycle every time,
  since the workflow has no path filters).
- **Ask before large-blast-radius pivots.** The CatDogCNN→ResNet-50 swap only
  happened after an explicit clarifying question about scope depth. Don't
  assume "the user mentioned X once" means "do the biggest version of X."
