# Assignment 2 Submission Report — End-to-End MLOps Pipeline

## Cover information

| Field | Submission value |
|---|---|
| Student name | **[ENTER STUDENT NAME]** |
| BITS ID | **[ENTER BITS ID]** |
| Programme | BITS Pilani WILP M.Tech. Artificial Intelligence and Machine Learning |
| Course | MLOps — AIMLCZG523 |
| Assignment | Assignment 2 |
| Semester tag | **[CONFIRM S1-25 OR S2-25]** |
| GitHub repository | **[ENTER GITHUB REPOSITORY URL]** |
| CI run | **[ENTER SUCCESSFUL GITHUB ACTIONS RUN URL]** |
| Container image | **[ENTER GHCR IMAGE URL AND DIGEST]** |
| Submission ZIP | **[ENTER SOURCE/CONFIG/MODEL ZIP FILENAME AND SHA-256]** |

## 1. Source and scope

The authoritative task source used for this implementation is:

`Semester-3/MLOps (S2-25_AIMLCZG523)/Assignments/Assignment-2/Problem-Statement.pdf`

There is a semester-label discrepancy that must not be silently resolved: the course folder and request identify **S2-25**, while the PDF header identifies **S1-25**. The technical work follows the supplied problem statement. The final cover page, Git tag, and LMS upload name should use the label confirmed by the instructor/LMS: **[CONFIRM S1-25 OR S2-25]**.

This report documents an end-to-end Cats vs Dogs MLOps baseline and follows the PDF's milestone structure exactly:

1. **M1 — Model Development & Experiment Tracking**
2. **M2 — Model Packaging & Containerization**
3. **M3 — CI Pipeline for Build, Test & Image Creation**
4. **M4 — CD Pipeline & Deployment**
5. **M5 — Monitoring, Logs & Final Submission**

## 2. Executive summary

The completed local pipeline processed 2,000 public images into balanced class partitions at 224 × 224 RGB resolution. A compact CNN was trained for five epochs. Its held-out test accuracy was `0.595` and weighted F1 was `0.5668333378`. The checkpoint was tied to MLflow run `4045b2ae755640799354dab50621441b` and SHA-256 `da70dbd561c481de897fc11b20c8d07d776aafa343074cacc27a9b421116f4af`.

The same preprocessing contract and class ordering are used for inference. Six automated tests passed with 83% coverage. The Docker image built successfully. The final local Docker Compose stack was verified with both API and Prometheus containers healthy, Prometheus target `http://api:8000/metrics` reporting `up`, and the post-deployment smoke passing. A 20-image request batch against the deployed service produced accuracy `0.6`, weighted F1 `0.5238095238`, mean latency `22.34358 ms`, and p95 latency `30.4336 ms`.

The outcome is a functioning and reproducible local MLOps baseline. The predictive performance is moderate and is not overstated. Transfer learning is the primary proposed model improvement. Live GitHub Actions publication, GHCR evidence, and a remote self-hosted deployment are not claimed without their required URLs and digest.

## 3. Problem definition

The machine-learning task is closed-set binary image classification:

\[
f(x;\theta) \rightarrow \{\text{cats},\text{dogs}\}.
\]

For output logits \(\mathbf{z}\), softmax gives:

\[
p(y=c\mid x)=\frac{e^{z_c}}{\sum_j e^{z_j}},
\]

and training minimizes cross-entropy:

\[
\mathcal{L}(x,y)=-\log p(y\mid x).
\]

The engineering objective is broader than one score: data, code, configuration, experiments, model artifacts, packaging, CI, deployment, monitoring, logs, and final evidence must be traceable and repeatable.

## 4. Architecture and milestone flow

```mermaid
flowchart TD
    A["M1: Git/DVC data and baseline"] --> B["M1: MLflow run and checkpoint"]
    B --> C["M2: FastAPI package and pinned requirements"]
    C --> D["M2/M3: Docker definition and image build"]
    E["M3: Unit tests and GitHub Actions"] --> D
    D --> F["M3: GHCR publication configuration"]
    F --> G["M4: Main-branch Linux self-hosted runner"]
    G --> H["M4: Docker Compose pull/deploy/smoke"]
    H --> I["M5: Request/response logs"]
    H --> J["M5: Prometheus counters and latency"]
    H --> K["M5: Feedback and 20-request tracking"]
    I --> L["M5: Source/config/model ZIP and <5 min video"]
    J --> L
    K --> L
```

Key decision: preprocessing and checkpoint metadata are shared between offline and online paths. This reduces training-serving skew and makes the artifact self-describing through class names, image size, model version, metrics, and checksum.

## 5. M1 — Model Development & Experiment Tracking

M1 covers Git/DVC, the baseline model, and MLflow.

### 5.1 Dataset source and provenance

- Dataset handle: `tongpython/cat-and-dog`
- Download implementation: `scripts/download_data.py`
- Reproducibility: `dvc.yaml` and `params.yaml`
- DVC state after reproduction: clean

The preparation stage verifies decodability, applies orientation correction, converts to RGB, center-crops without stretching, resizes to 224 × 224, and stores a SHA-256 content fingerprint in the manifest.

| Property | Value |
|---|---:|
| Total images | 2,000 |
| Training | 1,600 |
| Validation | 200 |
| Test | 200 |
| Image format | RGB |
| Resolution | 224 × 224 |
| Corrupt images skipped | 0 |

The split is deterministic and stratified by class. The manifest supports source-path and content-hash leakage checks.

### 5.2 Transform contract

Training applies random resized crop, horizontal flip, and mild color jitter. Validation, test, and inference use deterministic resizing. All paths use the same normalization:

\[
x'=\frac{x-0.5}{0.5}.
\]

The notebook asserts the `(3, 224, 224)` tensor contract.

### 5.3 Baseline CNN and training

Three convolutional blocks use convolution, batch normalization, ReLU, and max pooling. Adaptive average pooling, dropout, and a two-unit linear layer produce raw logits for cross-entropy loss.

| Parameter | Value |
|---|---:|
| Epochs | 5 |
| Image size | 224 |
| Batch size | 32 |
| Learning rate | 0.001 |
| Weight decay | 0.0001 |
| Optimizer | AdamW |
| Loss | Cross-entropy |
| Seed | 42 |

### 5.4 MLflow and held-out results

| Evidence | Value |
|---|---|
| Experiment | `cats-vs-dogs-baseline` |
| Run ID | `4045b2ae755640799354dab50621441b` |
| Model version | `1.0.0` |
| Model SHA-256 | `da70dbd561c481de897fc11b20c8d07d776aafa343074cacc27a9b421116f4af` |

| Metric | Value |
|---|---:|
| Accuracy | 0.595 |
| Weighted precision | 0.6283957292 |
| Weighted recall | 0.595 |
| Weighted F1 | 0.5668333378 |

MLflow records parameters, dataset sizes, device, class ordering, code revision where available, epoch metrics, final metrics, training duration, plots, checkpoint, metrics JSON, and model metadata.

### 5.5 Notebook evidence

`notebooks/01_model_development.ipynb` covers problem understanding, mathematics, provenance, split/leakage checks, preprocessing visualization, model/output checks, training invocation, artifact inspection, evaluation, and limitations. Verified status: **34 cells, 14 code cells executed, zero error outputs, valid nbformat 4.5**. Its checked default `RUN_TRAINING=False` inspects canonical DVC artifacts without launching a duplicate run.

## 6. M2 — Model Packaging & Containerization

M2 covers FastAPI health and predict endpoints, pinned requirements, and Docker/local package verification.

The service loads the trusted checkpoint, uses the shared deterministic preprocessing path, and exposes:

- `/health` for readiness;
- `/predict` for multipart image classification using form field `image`;
- `/metrics` for M5 Prometheus scraping;
- feedback behavior for M5 performance tracking.

The checkpoint loader validates required metadata and uses a weights-only load path. `requirements-api.txt` pins the serving dependencies, while `Dockerfile` defines the repeatable application image.

Local API verification commands:

```powershell
uvicorn cats_dogs_mlops.api:app --host 0.0.0.0 --port 8000
Invoke-RestMethod http://localhost:8000/health
curl.exe -X POST -F "image=@C:\path\to\sample.jpg" http://localhost:8000/predict
```

Do not upload sensitive images.

## 7. M3 — CI Pipeline for Build, Test & Image Creation

M3 covers unit tests, GitHub Actions, Docker image creation, and GHCR publishing.

Verified local evidence:

- Tests passed: **6**
- Coverage: **83%**
- Docker image build: passed locally
- GitHub Actions workflow: present at `.github/workflows/ci-cd.yml`
- GHCR publishing path: configured, but no live push is claimed

The workflow defines automated quality gates and image creation/publication under its configured events and permissions. Required live M3 evidence remains:

- **[GITHUB REPOSITORY URL]**
- **[SUCCESSFUL GITHUB ACTIONS RUN URL]**
- **[GHCR IMAGE URL]**
- **[GHCR IMMUTABLE SHA256 DIGEST]**
- **[SCREENSHOT: TEST/BUILD/PUBLISH JOBS PASSING]**

Coverage is a useful gate but not proof of correctness. Future tests should add corrupt input, load failure, concurrency, resource limits, and rollback behavior.

## 8. M4 — CD Pipeline & Deployment

M4 covers the Docker Compose target and the main-branch Linux self-hosted runner that pulls/deploys the image and performs a post-deployment smoke check.

Verified local M4 evidence:

- `docker-compose.yml` defines the deployment target.
- The final local Compose stack ran with both API and Prometheus containers healthy.
- Prometheus target `http://api:8000/metrics` reported `up`.
- The post-deployment smoke check passed.

The workflow's remote M4 path is configured for a trusted Linux x64 self-hosted runner on the main branch. That runner must pull the published image, deploy it with Docker Compose, and run the smoke check. A GitHub-hosted runner is ephemeral and cannot be treated as the persistent target host.

Remote M4 evidence is **not claimed** until these are supplied:

- **[SELF-HOSTED RUNNER SCREENSHOT]**
- **[SUCCESSFUL MAIN-BRANCH DEPLOYMENT JOB URL]**
- **[DEPLOYED IMAGE DIGEST]**
- **[REMOTE COMPOSE STATUS AND HEALTH/SMOKE SCREENSHOT]**
- **[DEPLOYMENT URL OR PRIVATE HOST NOTE]**

The runner must be trusted, patched, restricted to the intended repository, and protected from untrusted pull-request code and secret leakage.

## 9. M5 — Monitoring, Logs & Final Submission

M5 covers request/response logs, Prometheus counters/latency, feedback and 20-request performance tracking, the source/config/model ZIP, and a video shorter than five minutes.

### 9.1 Monitoring and logs

The service records request/response activity, exports Prometheus-compatible counters and latency observations, exposes health state, and supports feedback persistence. `monitoring/prometheus.yml` scrapes `http://api:8000/metrics` inside the Compose network. In the final local verification, both containers were healthy and this target reported `up`.

Recommended panels/alerts include request rate, error rate, mean/p95/p99 latency, confidence/class distribution, invalid-image rate, active model version/checksum, feedback coverage, delayed accuracy, container restarts, CPU, and memory.

### 9.2 Post-deployment performance tracking

| Metric | Value |
|---|---:|
| Labelled requests | 20 |
| Accuracy | 0.6 |
| Weighted F1 | 0.5238095238 |
| Mean latency | 22.34358 ms |
| P95 latency | 30.4336 ms |

The deployed accuracy is close to the offline accuracy, which supports basic train/serve consistency, but 20 requests are too few for a production guarantee. The lower deployed weighted F1 reinforces the need for per-class and larger-sample analysis.

### 9.3 Final submission package

The documentation and a 4:40 video script are ready. The final manual package must include:

- source, configuration, and model artifact ZIP: **[ZIP FILENAME AND SHA-256]**;
- final video under five minutes: **[VIDEO URL OR FILENAME]**;
- identity and semester confirmation;
- required local screenshots;
- live M3/M4 URLs and digest only if actually completed.

Do not include secrets, `.env`, `.venv`, unnecessary raw data, private logs, or cached dependencies in the ZIP.

## 10. Exact M1–M5 evidence matrix

| PDF milestone | Evidence delivered | Verified status | Manual/live evidence still required |
|---|---|---|---|
| **M1 — Model Development & Experiment Tracking** | Git-ready source; DVC pipeline/clean status; dataset provenance; five-epoch baseline; notebook; MLflow run; metrics/plots/checkpoint | Complete locally | MLflow/DVC screenshots and student metadata |
| **M2 — Model Packaging & Containerization** | FastAPI `/health` and `/predict`; shared inference contract; pinned API requirements; Dockerfile/local package verification | Complete locally | Final API/package screenshot |
| **M3 — CI Pipeline for Build, Test & Image Creation** | Six tests/83% coverage; GitHub Actions workflow; local Docker image build; GHCR publishing configuration | Local tests/build complete | Repository URL, successful CI URL, GHCR URL/digest |
| **M4 — CD Pipeline & Deployment** | Compose target; main-branch Linux self-hosted pull/deploy/smoke configuration; local two-container healthy stack and smoke | Local Compose deployment complete | Live runner/deployment job and remote smoke evidence |
| **M5 — Monitoring, Logs & Final Submission** | Request/response logging; Prometheus counters/latency and `up` target; feedback; 20-request evaluation; report/checklist; 4:40 script | Local monitoring/evaluation/docs complete | Source/config/model ZIP, final video, screenshots, semester confirmation |

## 11. Limitations and improvement plan

### Current limitations

1. Test accuracy `0.595` and weighted F1 `0.5668333378` are moderate.
2. The network is trained from scratch on a capped 2,000-image dataset.
3. Aggregate weighted metrics may hide class-specific weaknesses.
4. The closed-set classifier must choose cat or dog for unrelated images.
5. The deployment evaluation has only 20 labelled requests.
6. Local MLflow/Prometheus/Compose evidence is appropriate for the assignment baseline, not a multi-user production platform.
7. Reproducibility seeds do not guarantee bitwise equality across GPU drivers and hardware.

### Primary improvement: transfer learning

The next controlled experiment should use MobileNetV3 or EfficientNet pretrained on ImageNet. Train a new head with the backbone frozen, then fine-tune final blocks at a lower learning rate. Keep the identical DVC split and log both runs to MLflow. Compare accuracy, per-class precision/recall/F1, calibration, latency, model size, and CPU memory. Promote only if a predeclared quality gate is met without violating the latency budget.

Further improvements include confidence calibration, out-of-distribution rejection, data-slice evaluation, larger external test data, drift analysis, a model registry, signed images/SBOM, vulnerability scanning, and automated rollback.

## 12. Reproduction commands

Use Python 3.11 or 3.12.

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

mlflow ui --backend-store-uri file:./mlruns --port 5000
uvicorn cats_dogs_mlops.api:app --host 0.0.0.0 --port 8000
curl.exe -X POST -F "image=@C:\path\to\sample.jpg" http://localhost:8000/predict

docker build -t cats-dogs-mlops:local .
docker compose up --build -d
docker compose ps
python scripts/post_deployment_evaluation.py --base-url http://localhost:8000
docker compose down
```

## 13. Final evidence placeholders

- **[STUDENT NAME]**
- **[BITS ID]**
- **[CONFIRMED SEMESTER TAG]**
- **[GITHUB REPOSITORY URL]**
- **[GITHUB ACTIONS RUN URL]**
- **[GHCR IMAGE URL]**
- **[GHCR IMMUTABLE DIGEST]**
- **[SELF-HOSTED DEPLOYMENT JOB URL]**
- **[SOURCE/CONFIG/MODEL ZIP FILENAME AND SHA-256]**
- **[VIDEO URL OR FILENAME]**
- **[SCREENSHOT: DVC REPRO AND CLEAN STATUS]**
- **[SCREENSHOT: MLFLOW RUN 4045b2ae755640799354dab50621441b]**
- **[SCREENSHOT: TEST METRICS AND PLOTS]**
- **[SCREENSHOT: PYTEST 6 PASSED / 83% COVERAGE]**
- **[SCREENSHOT: DOCKER IMAGE BUILD]**
- **[SCREENSHOT: API HEALTH AND PREDICTION]**
- **[SCREENSHOT: BOTH COMPOSE CONTAINERS HEALTHY]**
- **[SCREENSHOT: PROMETHEUS TARGET http://api:8000/metrics UP]**
- **[SCREENSHOT: POST-DEPLOYMENT RESULTS]**
- **[SCREENSHOT: ACTIONS, GHCR, AND REMOTE RUNNER — ONLY AFTER LIVE VERIFICATION]**

## 14. Conclusion

The assignment delivers a coherent local lifecycle aligned with the PDF milestones: M1 makes model development traceable, M2 packages the inference service, M3 defines and locally verifies the test/build path, M4 verifies the Compose target locally and configures—but does not falsely claim—the remote main-branch deployment, and M5 provides monitoring, logs, performance tracking, and final-submission materials. The baseline's moderate quality is visible and leads to an academically defensible next step: controlled transfer learning followed by evidence-based promotion.
