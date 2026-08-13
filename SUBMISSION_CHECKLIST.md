# Assignment 2 Submission Checklist

Checked items are supported by verified local evidence. Unchecked items require student identity, instructor confirmation, live GitHub/GHCR/remote deployment evidence, final packaging, screenshots, or recording.

## M1 — Model Development & Experiment Tracking

PDF scope: **Git/DVC, baseline model, MLflow**.

- [x] Official source identified as `Problem-Statement.pdf`.
- [x] S2-25 folder/request versus S1-25 PDF-header discrepancy documented.
- [x] Git-ready project structure and ignore rules are present.
- [x] Kaggle handle is `tongpython/cat-and-dog`.
- [x] DVC download, prepare, and train stages are defined.
- [x] `params.yaml` controls data/training configuration.
- [x] Processed dataset contains 2,000 images.
- [x] Split is 1,600 train / 200 validation / 200 test.
- [x] Images are 224 × 224 RGB; corrupt count is zero.
- [x] Manifest/fingerprint and leakage-check evidence is present.
- [x] `dvc status` is clean.
- [x] Baseline CNN trained for five epochs.
- [x] Test accuracy is `0.62`.
- [x] Weighted precision is `0.6428571429`.
- [x] Weighted recall is `0.62`.
- [x] Weighted F1 is `0.6041666667`.
- [x] MLflow run ID is `e040fb0dbc64492f96eea1affa576c90`.
- [x] Model SHA-256 is `8070fc046c4e247d30fa0ec469cd2fd22c1205070c9ddde4121ba540bcc0a793`.
- [x] Notebook has 34 cells, 14 executed code cells, and zero errors.
- [x] Capture DVC clean-status screenshot. (`docs/screenshots/E2_dvc_status.png`)
- [x] Capture MLflow run/metrics/artifacts screenshot. (`docs/screenshots/E5_mlflow_run.png`, `E6_mlflow_artifacts.png`)

## M2 — Model Packaging & Containerization

PDF scope: **FastAPI health and predict endpoints, pinned requirements, Docker/local verification**.

- [x] Training and inference share preprocessing constants/logic.
- [x] Checkpoint stores and validates the inference contract.
- [x] FastAPI `/health` is implemented.
- [x] FastAPI `/predict` is implemented with multipart form field `image`.
- [x] Invalid input behavior is handled.
- [x] `requirements-api.txt` pins serving dependencies.
- [x] `Dockerfile` packages the API runtime.
- [x] Local API/package behavior is verified.
- [x] Capture local `/health` response. (`docs/screenshots/E7_health.png`)
- [x] Capture one genuine `/predict` response (`docs/screenshots/E7_predict.png`) using:

```powershell
curl.exe -X POST -F "image=@C:\path\to\sample.jpg" http://localhost:8000/predict
```

## M3 — CI Pipeline for Build, Test & Image Creation

PDF scope: **unit tests, GitHub Actions, Docker build, GHCR publishing**.

- [x] Eight automated tests pass.
- [x] Coverage is 83%.
- [x] GitHub Actions workflow exists at `.github/workflows/ci-cd.yml`.
- [x] Workflow contains automated test/build gates.
- [x] Docker image builds locally.
- [x] GHCR publishing path is configured.
- [x] Documentation does not claim an unverified GHCR push.
- [x] Create/select GitHub repository: **https://github.com/hamza-aziz-ai/MLOPs-Cats-Dogs-Pipeline**.
- [x] Push final commit. Tag pending confirmed semester (see Identity section).
- [x] Confirm Actions permissions include `contents: read` and `packages: write`.
- [x] Run M3 successfully: **https://github.com/hamza-aziz-ai/MLOPs-Cats-Dogs-Pipeline/actions/runs/31689507647**.
- [x] Confirm all unit-test and image-build jobs are green.
- [x] Confirm package exists in GHCR: **https://github.com/hamza-aziz-ai/MLOPs-Cats-Dogs-Pipeline/pkgs/container/mlops-cats-dogs-pipeline**.
- [x] Record immutable digest: **`sha256:02029143e1931a74b0deb3ac92fae06f2446b54ec00b4b0ba1319f20341ba697`**.
- [x] Verify displayed digest matches the image intended for M4 deployment (same tag/digest pulled by the deploy job).
- [x] Capture 8-passed/83%-coverage screenshot. (`docs/screenshots/E8_pytest_coverage.png`)
- [x] Capture successful GitHub Actions test/build screenshot. (`docs/screenshots/E10_github_actions_run.png`)
- [x] Capture GHCR tag/digest screenshot. (`docs/screenshots/E10_ghcr_package.png`)

## M4 — CD Pipeline & Deployment

PDF scope: **Docker Compose target; main-branch self-hosted Linux runner pulls/deploys image; post-deploy smoke**.

- [x] `docker-compose.yml` defines the deployment target.
- [x] Main-branch CD path is configured for a Linux x64 self-hosted runner.
- [x] Deployment logic pulls/deploys the image and runs a post-deploy smoke check.
- [x] Final local Compose stack was verified with both API and Prometheus containers healthy.
- [x] Final local Prometheus target `http://api:8000/metrics` reported `up`.
- [x] Local post-deployment smoke check passed.
- [x] Documentation does not claim an unverified remote self-hosted deployment.
- [x] Provision a trusted Linux x64 host (WSL2 Ubuntu 24.04, x86_64, Docker Engine + Compose v2 v5.3.1).
- [x] Install Docker Engine and Docker Compose v2 on the host.
- [x] Register the runner only to the trusted repository/organization (repo-scoped, not org-wide).
- [x] Confirm runner labels match the workflow (`self-hosted`, `linux` — matches `runs-on: [self-hosted, linux]`).
- [x] Confirm runner is online before the main-branch deployment.
- [x] Complete remote pull/deploy/smoke: **https://github.com/hamza-aziz-ai/MLOPs-Cats-Dogs-Pipeline/actions/runs/31689507647/job/94416274227**.
- [x] Confirm deployed image digest: **`sha256:02029143e1931a74b0deb3ac92fae06f2446b54ec00b4b0ba1319f20341ba697`**.
- [x] Verify remote Compose containers are healthy (`api` and `prometheus`, both `Up`/`healthy`).
- [x] Verify remote `/health` and one cat/dog request (`/health` → `{"status":"ready","model_loaded":true}`; `/predict` on a real cat image → `{"label":"cats","confidence":0.582}`).
- [x] Record endpoint/private-host note: **private host — student's own WSL2 Ubuntu self-hosted runner, `http://localhost:8000`, not publicly routable.**
- [x] Capture runner-online, deploy, Compose-health, and smoke screenshots. (`docs/screenshots/E12_runner_online.png`, `E11_compose_healthy.png`)
- [ ] Disable/remove the runner after demonstration if it will not be securely maintained.

Why this remains manual: the Linux self-hosted runner operates the target host's Docker daemon and persists beyond an ephemeral CI VM. Do not expose it to untrusted pull requests or secrets.

## M5 — Monitoring, Logs & Final Submission

PDF scope: **request/response logs, Prometheus counters/latency, feedback and 20-request performance tracking, source/config/model ZIP, video under five minutes**.

- [x] Request/response activity is logged.
- [x] Prometheus-compatible counters and latency observations are exposed.
- [x] `monitoring/prometheus.yml` targets `http://api:8000/metrics`.
- [x] Both local Compose containers were healthy and Prometheus target was `up`.
- [x] Feedback persistence path is configurable.
- [x] Post-deployment evaluator is present.
- [x] Evaluated batch contains 20 labelled requests.
- [x] Deployed accuracy is `0.7`.
- [x] Deployed weighted F1 is `0.6969696970`.
- [x] Mean latency is `58.9655750 ms`.
- [x] P95 latency is `85.8087000 ms`.
- [x] README.md, SUBMISSION_REPORT.md, SUBMISSION_CHECKLIST.md, and VIDEO_DEMO_SCRIPT.md are present.
- [x] Video script target is 4:40, below five minutes.
- [x] Baseline limitations and transfer-learning improvement are honest.
- [x] Capture request/response log screenshot without private data. (`docs/screenshots/E13a_request_logs.png`, `E13b_feedback_predictions.png`)
- [x] Capture `/metrics` output and Prometheus target `up`. (`docs/screenshots/E14_prometheus_target_up.png`)
- [x] Capture 20-request accuracy/F1/latency output. (`docs/screenshots/E15_post_deployment.png`)
- [x] Build source/config/model ZIP: **`cats-dogs-mlops-submission.zip`** (172 git-tracked files, 16.6 MB).
- [x] Record ZIP SHA-256: **`53a583c3950458d7223fda43871b17818af3b8a1dcb24deac472b6ecafbd9eb1`**.
- [x] Verify ZIP contains required source, configuration, and model artifact (`src/`, `scripts/`, `tests/`, `params.yaml`, `dvc.yaml`, `Dockerfile`, `models/cat_dog_cnn.pt`, `mlruns/`).
- [x] Verify ZIP excludes secrets, `.env`, `.venv`, unnecessary raw data, private logs, and caches (built from `git ls-files` — no untracked/ignored content, no `.env`/`.idea`/build caches).
- [ ] Record final video: **[VIDEO URL OR FILENAME]**.
- [ ] Confirm final video duration is below 5:00.

## Identity and academic confirmation

- [x] Replace **[ENTER STUDENT NAME]** with **Hamza Aziz**.
- [x] Replace **[ENTER BITS ID]** with **2024AC05133**.
- [x] Confirm semester with LMS/instructor: **S2-25**.
- [x] Use the confirmed semester consistently in report, README, repository, tag, ZIP, video, and LMS filename.
- [ ] Confirm LMS submission format/naming.
- [ ] Review rendered report for broken links, diagrams, and page layout.

## Screenshot matrix

- [x] **E1 M1:** repository root and final commit/tag. (`docs/screenshots/E1_repo_commit.png`)
- [x] **E2 M1:** DVC stages and clean status. (`docs/screenshots/E2_dvc_status.png`)
- [x] **E3 M1:** 2,000-image summary and 1,600/200/200 split. (`docs/screenshots/E3_dataset_split.png`)
- [x] **E4 M1:** executed notebook and output shape. (`docs/screenshots/E4_notebook_executed.png`)
- [x] **E5 M1:** MLflow run `e040fb0dbc64492f96eea1affa576c90`. (`docs/screenshots/E5_mlflow_run.png`)
- [x] **E6 M1:** test metrics, learning curves, confusion matrix. (`docs/screenshots/E6_mlflow_artifacts.png`, `artifacts/confusion_matrix.png`, `artifacts/loss_curves.png`)
- [x] **E7 M2:** FastAPI health and prediction using field `image`. (`docs/screenshots/E7_health.png`, `E7_predict.png`)
- [x] **E8 M3:** 8 tests passed and 83% coverage. (`docs/screenshots/E8_pytest_coverage.png`)
- [x] **E9 M3:** Docker image build. (`docs/screenshots/E9_docker_images.png`)
- [x] **E10 M3:** live Actions/GHCR evidence. (`docs/screenshots/E10_github_actions_run.png`, `E10_ghcr_package.png`)
- [x] **E11 M4:** both Compose containers healthy. (`docs/screenshots/E11_compose_healthy.png`)
- [x] **E12 M4:** local smoke pass and remote runner/deploy. (`docs/screenshots/E12_runner_online.png`)
- [x] **E13 M5:** request/response logs and application metrics. (`docs/screenshots/E13a_request_logs.png`, `E13b_feedback_predictions.png`)
- [x] **E14 M5:** Prometheus target `http://api:8000/metrics` reporting `up`. (`docs/screenshots/E14_prometheus_target_up.png`)
- [x] **E15 M5:** 20-request performance results. (`docs/screenshots/E15_post_deployment.png`)
- [x] **E16 M5:** source/config/model ZIP contents/checksum. (`docs/screenshots/E16_zip_contents.png`) Video still pending.

## Final reproducibility check

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
dvc repro
dvc status
pytest --cov=cats_dogs_mlops --cov-report=term-missing
jupyter nbconvert --execute --to notebook --inplace notebooks\01_model_development.ipynb
docker build -t cats-dogs-mlops:submission .
docker compose up -d
docker compose ps
Invoke-RestMethod http://localhost:8000/health
curl.exe -X POST -F "image=@C:\path\to\sample.jpg" http://localhost:8000/predict
python scripts/post_deployment_evaluation.py --base-url http://localhost:8000
docker compose down
```

- [x] Fresh Python 3.11 environment has no undeclared dependency — verified every `requirements*.txt` pin resolves to a real, installable manylinux cp311 wheel (not literally re-run in a fresh venv; CI's own `ubuntu-latest`/Python 3.11 job installing and passing is the stronger live proof of this).
- [x] `dvc repro` succeeds or correctly reports unchanged stages — ran for real; `download`/`prepare` unchanged, `train` reran cleanly.
- [x] `dvc status` remains clean — verified after the repro above.
- [x] Tests remain 8 passed with at least 83% coverage — reran locally: 8 passed, 83% coverage.
- [x] Notebook executes with zero errors — verified: 34 cells, 14 code cells executed, 0 error outputs.
- [x] Image builds — same image already live on GHCR (see M3).
- [x] Both Compose containers become healthy — verified live (`api`, `prometheus` both healthy).
- [x] Prometheus target is `up` — verified live (`cats-dogs-api`, state `up`).
- [x] Post-deploy smoke/evaluation succeeds — reran for real: 20/20 requests, accuracy `0.7`.
- [x] Metrics/checksum match the report — the report was updated to this new canonical run (`e040fb0dbc64492f96eea1affa576c90`, accuracy `0.62`) rather than left pointing at a stale one.

## Final integrity and sign-off

- [x] Replace every required bracketed placeholder (video URL/filename is the one remaining item, since the video itself doesn't exist yet).
- [x] Confirm no state-of-the-art or production-ready claim was introduced.
- [x] Confirm live M3/M4 claims match genuine URLs, digest, and screenshots.
- [x] Confirm model checksum and MLflow run ID match artifacts/screenshots.
- [x] Confirm final Git commit/tag is the submitted one (`896c4cd`, tag `s2-25-aimlczg523-assignment2`).
- [x] Confirm all evaluator links open (GitHub run, GHCR package, deploy job all verified reachable).
- [ ] Watch final video once at normal speed; audio/text are readable.
- [ ] Upload before the LMS deadline and retain receipt.

| Check | Value |
|---|---|
| Student name | Hamza Aziz |
| BITS ID | 2024AC05133 |
| Confirmed semester | S2-25 |
| Final Git commit | **`896c4cd`** |
| Release/tag | **`s2-25-aimlczg523-assignment2`** |
| CI run | **https://github.com/hamza-aziz-ai/MLOPs-Cats-Dogs-Pipeline/actions/runs/31689507647** |
| GHCR digest | **`sha256:02029143e1931a74b0deb3ac92fae06f2446b54ec00b4b0ba1319f20341ba697`** |
| Deployment job | **https://github.com/hamza-aziz-ai/MLOPs-Cats-Dogs-Pipeline/actions/runs/31689507647/job/94416274227** |
| ZIP | **`cats-dogs-mlops-submission.zip`**, sha256 `53a583c3950458d7223fda43871b17818af3b8a1dcb24deac472b6ecafbd9eb1` |
| Video | **[URL/FILENAME]** |
| Submission time | **[DATE AND TIME WITH TIME ZONE]** |

- [ ] I confirm that the evidence is genuine, limitations are retained, and every manual/live item has been verified.
