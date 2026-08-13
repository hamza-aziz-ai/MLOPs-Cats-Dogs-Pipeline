# Assignment 2 Video Demo Script — Maximum 4:50

Target duration: **4 minutes 40 seconds**, leaving a 10-second margin below five minutes.

## Before recording

- Only **[VIDEO]** (the recording itself) remains to fill in — identity, semester, GitHub/CI/GHCR/ZIP evidence are already resolved below.
- Semester confirmed by the student as S2-25. The folder/request says S2-25; the PDF header says S1-25.
- Open README, DVC, executed notebook, MLflow, tests, FastAPI, Docker/Compose, Prometheus, GitHub Actions configuration, report, and checklist.
- Pre-run slow commands. Show genuine outputs rather than waiting for training/builds.
- Increase font size and hide notifications, tokens, usernames, private logs, and unrelated tabs.

## 0:00–0:20 — Identity, source, and objective

**Show:** report cover and repository title.

**Say:**

> I am Hamza Aziz, BITS ID 2024AC05133. This is Assignment 2 for MLOps, AIMLCZG523, semester S2-25. The source is `Problem-Statement.pdf`. The folder says S2-25 while the PDF header says S1-25, so I confirmed the final label rather than silently choosing one. This project implements an end-to-end Cats versus Dogs MLOps pipeline.

## 0:20–0:55 — Exact PDF milestone mapping

**Show:** README M1–M5 table.

**Say:**

> The PDF milestones map exactly as follows. M1 is Model Development and Experiment Tracking: Git and DVC, the baseline model, and MLflow. M2 is Model Packaging and Containerization: FastAPI health and predict endpoints, pinned requirements, and Docker local verification. M3 is the CI Pipeline for Build, Test and Image Creation: unit tests, GitHub Actions, Docker build, and GHCR publishing. M4 is the CD Pipeline and Deployment: the Docker Compose target, where a main-branch Linux self-hosted runner pulls and deploys the image and runs a post-deployment smoke check. M5 is Monitoring, Logs and Final Submission: request and response logs, Prometheus counters and latency, feedback and 20-request performance tracking, the source/config/model ZIP, and a video under five minutes.

## 0:55–1:28 — M1 data, baseline, and MLflow

**Show:** `dvc.yaml`, `params.yaml`, summary/manifest, clean `dvc status`, notebook, then MLflow.

**Say:**

> The Kaggle source is `tongpython/cat-and-dog`. DVC controls download, preparation, and training. The verified dataset has 2,000 RGB images at 224 by 224 pixels: 1,600 train, 200 validation, and 200 test, with zero corrupt images. DVC status is clean. The executed notebook has 34 cells, 14 code cells, and zero errors. The five-epoch baseline achieved 0.62 accuracy and 0.6042 weighted F1. MLflow run `e040fb0dbc64492f96eea1affa576c90` records the experiment, and the model checksum begins `8070fc04`.

## 1:28–1:58 — M2 API package

**Show:** pinned `requirements-api.txt`, Dockerfile, `/health`, then one `/predict` response.

**Say:**

> M2 packages the shared inference contract in FastAPI with pinned serving requirements and a Dockerfile. The health endpoint verifies readiness. The predict endpoint accepts multipart field `image`, applies the same deterministic preprocessing used offline, and returns the predicted class and confidence.

**Prepared commands:**

```powershell
Invoke-RestMethod http://localhost:8000/health
curl.exe -X POST -F "image=@C:\path\to\sample.jpg" http://localhost:8000/predict
```

## 1:58–2:28 — M3 tests, CI, image, and GHCR boundary

**Show:** test/coverage output, local image, and `.github/workflows/ci-cd.yml`; show live Actions/GHCR only if completed.

**Say:**

> M3 contains eight passing tests with 83 percent coverage. The Docker image builds locally. GitHub Actions is configured to run the tests, build the image, and publish to GHCR under its configured conditions. **[If verified: show CI URL and GHCR digest.]** Otherwise, I am showing configuration and local build evidence only; I do not claim a live GHCR push without a run URL and immutable digest.

## 2:28–3:03 — M4 Compose deployment and smoke

**Show:** `docker compose ps`, both healthy containers, Prometheus target page, and smoke result.

**Say:**

> M4 uses Docker Compose as the deployment target. The final local stack was verified with both the API and Prometheus containers healthy. Prometheus target `http://api:8000/metrics` reports up, and the post-deployment smoke passed. The remote workflow runs on the main branch on a trusted Linux x64 self-hosted runner, which pulls and deploys the image — here's the live deployment job at **[DEPLOYMENT JOB URL: `https://github.com/hamza-aziz-ai/MLOPs-Cats-Dogs-Pipeline/actions/runs/31706209711/job/94470326140`]**, with both containers healthy and a real cat/dog prediction against the deployed instance.

## 3:03–3:38 — M5 monitoring, logs, and feedback

**Show:** bounded request/response logs, `/metrics`, Prometheus query/target, feedback path, and evaluation output.

**Say:**

> M5 records request and response activity, exposes Prometheus counters and latency observations, and supports feedback. The running HTTP service was evaluated with 20 labelled images. Accuracy is 0.7, weighted F1 is 0.6970, mean latency is 58.96558 milliseconds, and p95 is 85.8087 milliseconds. This checks the complete request path, but 20 requests are not enough for a production guarantee.

## 3:38–4:08 — M5 final package

**Show:** source/config/model ZIP contents, checksum, report/checklist, and video filename/timeline.

**Say:**

> The final M5 package contains the required source, configuration, and model artifact in **[ZIP FILENAME]**, checksum **[SHORT ZIP SHA256]**, together with the report and evidence. This demonstration is below five minutes. Secrets, the virtual environment, unnecessary raw data, and private logs are excluded. Live GitHub, GHCR, and remote deployment links are included only if genuinely completed.

## 4:08–4:40 — Honest conclusion and improvement

**Show:** limitations and checklist.

**Say:**

> This is a reproducible working baseline, not a high-accuracy production classifier. Test accuracy is 59.5 percent. The next controlled experiment is transfer learning with MobileNetV3 or EfficientNet using the same DVC split and MLflow comparison. I would promote it only if per-class quality improves without breaking the latency budget. The checklist keeps completed local evidence separate from manual identity, ZIP, video, GitHub, GHCR, and remote deployment evidence. Thank you.

## Recording evidence checklist

- [ ] Student name, BITS ID, and confirmed semester appear once.
- [ ] Exact M1–M5 mapping is spoken as written above.
- [ ] DVC clean status and 2,000 / 1,600 / 200 / 200 evidence are readable.
- [ ] Notebook execution, MLflow run, and held-out metrics are visible.
- [ ] FastAPI uses multipart field `image`.
- [ ] Eight tests and 83% coverage are visible under M3.
- [ ] Local Docker image build is visible under M3.
- [ ] Both Compose containers healthy and smoke pass are visible under M4.
- [ ] Prometheus target `http://api:8000/metrics` reporting `up` is visible.
- [ ] Request/response logs and 20-request results are visible under M5.
- [ ] Source/config/model ZIP filename/checksum is visible.
- [ ] Actions URL, GHCR digest, and remote runner are shown only if genuinely verified.
- [ ] Limitations and transfer-learning improvement are stated.
- [ ] Final duration is below 5:00.
