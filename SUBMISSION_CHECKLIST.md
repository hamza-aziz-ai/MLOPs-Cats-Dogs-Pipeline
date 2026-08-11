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
- [x] Test accuracy is `0.595`.
- [x] Weighted precision is `0.6283957292`.
- [x] Weighted recall is `0.595`.
- [x] Weighted F1 is `0.5668333378`.
- [x] MLflow run ID is `4045b2ae755640799354dab50621441b`.
- [x] Model SHA-256 is `da70dbd561c481de897fc11b20c8d07d776aafa343074cacc27a9b421116f4af`.
- [x] Notebook has 34 cells, 14 executed code cells, and zero errors.
- [ ] Capture DVC clean-status screenshot.
- [ ] Capture MLflow run/metrics/artifacts screenshot.

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
- [ ] Capture local `/health` response.
- [ ] Capture one genuine `/predict` response using:

```powershell
curl.exe -X POST -F "image=@C:\path\to\sample.jpg" http://localhost:8000/predict
```

## M3 — CI Pipeline for Build, Test & Image Creation

PDF scope: **unit tests, GitHub Actions, Docker build, GHCR publishing**.

- [x] Six automated tests pass.
- [x] Coverage is 83%.
- [x] GitHub Actions workflow exists at `.github/workflows/ci-cd.yml`.
- [x] Workflow contains automated test/build gates.
- [x] Docker image builds locally.
- [x] GHCR publishing path is configured.
- [x] Documentation does not claim an unverified GHCR push.
- [ ] Create/select GitHub repository: **[GITHUB REPOSITORY URL]**.
- [ ] Push final commit and tag.
- [ ] Confirm Actions permissions include `contents: read` and `packages: write`.
- [ ] Run M3 successfully: **[GITHUB ACTIONS RUN URL]**.
- [ ] Confirm all unit-test and image-build jobs are green.
- [ ] Confirm package exists in GHCR: **[GHCR IMAGE URL]**.
- [ ] Record immutable digest: **[GHCR SHA256 DIGEST]**.
- [ ] Verify displayed digest matches the image intended for M4 deployment.
- [ ] Capture 6-passed/83%-coverage screenshot.
- [ ] Capture successful GitHub Actions test/build screenshot.
- [ ] Capture GHCR tag/digest screenshot.

## M4 — CD Pipeline & Deployment

PDF scope: **Docker Compose target; main-branch self-hosted Linux runner pulls/deploys image; post-deploy smoke**.

- [x] `docker-compose.yml` defines the deployment target.
- [x] Main-branch CD path is configured for a Linux x64 self-hosted runner.
- [x] Deployment logic pulls/deploys the image and runs a post-deploy smoke check.
- [x] Final local Compose stack was verified with both API and Prometheus containers healthy.
- [x] Final local Prometheus target `http://api:8000/metrics` reported `up`.
- [x] Local post-deployment smoke check passed.
- [x] Documentation does not claim an unverified remote self-hosted deployment.
- [ ] Provision a trusted Linux x64 host.
- [ ] Install Docker Engine and Docker Compose v2 on the host.
- [ ] Register the runner only to the trusted repository/organization.
- [ ] Confirm runner labels match the workflow.
- [ ] Confirm runner is online before the main-branch deployment.
- [ ] Complete remote pull/deploy/smoke: **[DEPLOYMENT JOB URL]**.
- [ ] Confirm deployed image digest: **[DEPLOYED SHA256 DIGEST]**.
- [ ] Verify remote Compose containers are healthy.
- [ ] Verify remote `/health` and one cat/dog request.
- [ ] Record endpoint/private-host note: **[DEPLOYMENT URL / PRIVATE HOST NOTE]**.
- [ ] Capture runner-online, deploy, Compose-health, and smoke screenshots.
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
- [x] Deployed accuracy is `0.6`.
- [x] Deployed weighted F1 is `0.5238095238`.
- [x] Mean latency is `22.34358 ms`.
- [x] P95 latency is `30.4336 ms`.
- [x] README.md, SUBMISSION_REPORT.md, SUBMISSION_CHECKLIST.md, and VIDEO_DEMO_SCRIPT.md are present.
- [x] Video script target is 4:40, below five minutes.
- [x] Baseline limitations and transfer-learning improvement are honest.
- [ ] Capture request/response log screenshot without private data.
- [ ] Capture `/metrics` output and Prometheus target `up`.
- [ ] Capture 20-request accuracy/F1/latency output.
- [ ] Build source/config/model ZIP: **[ZIP FILENAME]**.
- [ ] Record ZIP SHA-256: **[ZIP SHA256]**.
- [ ] Verify ZIP contains required source, configuration, and model artifact.
- [ ] Verify ZIP excludes secrets, `.env`, `.venv`, unnecessary raw data, private logs, and caches.
- [ ] Record final video: **[VIDEO URL OR FILENAME]**.
- [ ] Confirm final video duration is below 5:00.

## Identity and academic confirmation

- [ ] Replace **[ENTER STUDENT NAME]** with **[STUDENT NAME]**.
- [ ] Replace **[ENTER BITS ID]** with **[BITS ID]**.
- [ ] Confirm semester with LMS/instructor: **[S1-25 OR S2-25]**.
- [ ] Use the confirmed semester consistently in report, README, repository, tag, ZIP, video, and LMS filename.
- [ ] Confirm LMS submission format/naming.
- [ ] Review rendered report for broken links, diagrams, and page layout.

## Screenshot matrix

- [ ] **E1 M1:** repository root and final commit/tag.
- [ ] **E2 M1:** DVC stages and clean status.
- [ ] **E3 M1:** 2,000-image summary and 1,600/200/200 split.
- [ ] **E4 M1:** executed notebook and output shape.
- [ ] **E5 M1:** MLflow run `4045b2ae755640799354dab50621441b`.
- [ ] **E6 M1:** test metrics, learning curves, confusion matrix.
- [ ] **E7 M2:** FastAPI health and prediction using field `image`.
- [ ] **E8 M3:** 6 tests passed and 83% coverage.
- [ ] **E9 M3:** Docker image build.
- [ ] **E10 M3:** live Actions/GHCR evidence only after completion.
- [ ] **E11 M4:** both Compose containers healthy.
- [ ] **E12 M4:** local smoke pass; remote runner/deploy only after completion.
- [ ] **E13 M5:** request/response logs and application metrics.
- [ ] **E14 M5:** Prometheus target `http://api:8000/metrics` reporting `up`.
- [ ] **E15 M5:** 20-request performance results.
- [ ] **E16 M5:** source/config/model ZIP contents/checksum and final video file/link.

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

- [ ] Fresh Python 3.11/3.12 environment has no undeclared dependency.
- [ ] `dvc repro` succeeds or correctly reports unchanged stages.
- [ ] `dvc status` remains clean.
- [ ] Tests remain 6 passed with at least 83% coverage.
- [ ] Notebook executes with zero errors.
- [ ] Image builds.
- [ ] Both Compose containers become healthy.
- [ ] Prometheus target is `up`.
- [ ] Post-deploy smoke/evaluation succeeds.
- [ ] Metrics/checksum match the report, or the report is updated to the new canonical run.

## Final integrity and sign-off

- [ ] Replace every required bracketed placeholder.
- [ ] Confirm no state-of-the-art or production-ready claim was introduced.
- [ ] Confirm live M3/M4 claims match genuine URLs, digest, and screenshots.
- [ ] Confirm model checksum and MLflow run ID match artifacts/screenshots.
- [ ] Confirm final Git commit/tag is the submitted one.
- [ ] Confirm all evaluator links open.
- [ ] Watch final video once at normal speed; audio/text are readable.
- [ ] Upload before the LMS deadline and retain receipt.

| Check | Value |
|---|---|
| Student name | **[STUDENT NAME]** |
| BITS ID | **[BITS ID]** |
| Confirmed semester | **[S1-25 OR S2-25]** |
| Final Git commit | **[COMMIT SHA]** |
| Release/tag | **[TAG]** |
| CI run | **[URL]** |
| GHCR digest | **[SHA256]** |
| Deployment job | **[URL]** |
| ZIP | **[FILENAME AND SHA256]** |
| Video | **[URL/FILENAME]** |
| Submission time | **[DATE AND TIME WITH TIME ZONE]** |

- [ ] I confirm that the evidence is genuine, limitations are retained, and every manual/live item has been verified.
