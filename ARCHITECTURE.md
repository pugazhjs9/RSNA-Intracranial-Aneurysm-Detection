# Production Architecture: RSNA Intracranial Aneurysm Detection System

> **Author**: Staff+ Principal Architect  
> **Domain**: Medical AI / Clinical Decision Support  
> **Stack**: Python · PyTorch · FastAPI · Kubernetes · Kafka · PostgreSQL · Redis · S3-compatible object store  
> **Constraint**: HIPAA-compliant, GPU-inference, ensemble of 6 heterogeneous models (~4 GB VRAM each loaded concurrently), DICOM ingestion

---

## 0. Context & Problem Framing

The system takes a raw MRI/TOF-MRA DICOM series as input and returns:
1. Per-location aneurysm probability scores across 14 anatomical classes
2. A localizer image showing the highest-confidence slice with bounding box
3. An audit trail suitable for clinician review

The inference pipeline is a **6-model GPU ensemble**:
- `exp0`: YOLOv11 — aneurysm detection/localization (1280px)
- `exp1`: YOLOv5-v7.0 — brain bounding box detection (640px)
- `exp2/exp4`: ViT-Large 384, EVA-Large 384 — classification
- `exp3/exp5`: MiT-B4 FPN 384 — multi-task classification + segmentation

Final ensemble: `0.25×exp3 + 0.25×exp5 + 0.125×exp2_vit + 0.125×exp4_vit + 0.125×exp2_eva + 0.125×exp4_eva`

This is not a toy: ~0.89 AUC test score, clinical-grade sensitivity requirements, and PHI data obligations.

---

## 1. Architecture Style

**Decision: Modular Event-Driven Microservices on Kubernetes**

### Rejected alternatives

| Alternative | Why Rejected |
|---|---|
| Monolith | GPU model loading is 4–8 GB; single process failure == full outage. Zero blast radius control. |
| Pure Serverless | GPU cold start latency is 30–90s. Unacceptable for clinical workflows. AWS Lambda has no A10G support at reasonable cost. |
| Pure Microservices (fine-grained) | Network overhead between inference steps (pre-process → brain-det → classify) would kill throughput. Inference stages are tightly coupled on GPU memory. |
| Ray Serve only | Valid layer, but not a full architecture. Used within the inference cluster. |

### Chosen: Modular Event-Driven Microservices

- **Coarse-grained services** aligned to bounded contexts (Upload, Inference, Result, Audit)
- **Kafka** as the central event bus — decouples ingestion rate from inference capacity
- **GPU Inference cluster** designed around Ray Serve for model multiplexing on shared GPU memory
- **Synchronous REST/gRPC** only at user-facing boundaries; everything internal is async

---

## 2. System Design

### 2.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              CLINICAL USERS                                     │
│         (Radiologists · Clinician Web UI · PACS Systems · HL7/FHIR Clients)    │
└────────────────────────────┬───────────────────────────────────────────────────┘
                             │  HTTPS / REST / FHIR R4
                ┌────────────▼────────────┐
                │       API GATEWAY        │   ← Kong / AWS API GW
                │  (Auth · RL · Routing)   │   ← mTLS for PACS integrations
                └─┬──────────┬────────────┘
                  │          │
        ┌─────────▼──┐  ┌────▼───────────┐
        │  Upload    │  │  Result Query  │
        │  Service   │  │  Service (REST)│
        └─────┬──────┘  └────▲───────────┘
              │ Multipart DICOM upload       │ Series predictions
              │              │
      ┌───────▼──────────────┴──────────────┐
      │          DICOM Object Store          │   ← MinIO / S3 (PHI encrypted)
      └───────┬──────────────────────────────┘
              │ ObjectCreated event
      ┌───────▼──────────────────────────────┐
      │           EVENT BUS (Kafka)           │
      │  Topics: series.uploaded             │
      │          series.preprocessed         │
      │          series.inference.complete   │
      │          series.result.ready         │
      │          audit.log                   │
      └───┬──────────────┬────────────────────┘
          │              │
  ┌───────▼──┐    ┌──────▼──────────────────────┐
  │ Preproc  │    │     INFERENCE CLUSTER        │
  │ Service  │    │  ┌────────────────────────┐  │
  │ (CPU)    │    │  │    Ray Serve           │  │
  │          │    │  │  ┌──────┐ ┌─────────┐ │  │
  │ • DICOM  │    │  │  │Brain │ │Aneurysm │ │  │
  │   parse  │    │  │  │ Det  │ │  Det    │ │  │
  │ • Window │    │  │  │YOLOv5│ │YOLOv11  │ │  │
  │ • 3-ch   │    │  │  └──────┘ └─────────┘ │  │
  │   stack  │    │  │  ┌──────┐ ┌──────────┐│  │
  └──────────┘    │  │  │ViT/  │ │ MiT-B4   ││  │
                  │  │  │EVA   │ │ FPN x2   ││  │
                  │  │  │x2 cls│ │ aux-cls  ││  │
                  │  │  └──────┘ └──────────┘│  │
                  │  │     Ensemble Layer     │  │
                  │  └────────────────────────┘  │
                  │  GPU Node Pool (A10G/A100)   │
                  └──────┬───────────────────────┘
                         │ Kafka: series.inference.complete
                  ┌──────▼───────────────────────┐
                  │       Result Service          │
                  │  (Write predictions to DB)    │
                  └──────┬───────────────────────┘
                         │
               ┌─────────▼──────────────┐
               │  PostgreSQL (RDS/CRDB)  │   ← Predictions, audit, metadata
               │  Redis (ElastiCache)    │   ← Hot predictions, session cache
               │  S3 (result images)     │   ← Localizer PNGs, overlays
               └────────────────────────┘
                         │
               ┌─────────▼──────────────┐
               │   Audit Service         │   ← Kafka consumer
               │  (HIPAA access logs)   │   ← Immutable append-only store
               └────────────────────────┘
```

### 2.2 Data Flow — Critical Path

```
DICOM Upload → S3 PUT → Kafka(series.uploaded)
  → Preprocessing Service (CPU)
      → DICOM parse, window clipping, z-sort, 3-channel stack
      → Write processed tensors to S3
  → Kafka(series.preprocessed)
  → Ray Serve Inference Pipeline (GPU)
      Stage 1: Brain Detection (YOLOv5, 640px)     — ~30ms
      Stage 2: Crop + Resize (384px)               — CPU, 10ms
      Stage 3: Classification ensemble (batch=8)   — ~200ms
      Stage 4: Aneurysm Localizer (YOLOv11, 1280px)— ~50ms (conditional)
      Stage 5: Weighted ensemble merge             — 1ms
  → Kafka(series.inference.complete)
  → Result Service → PostgreSQL + Redis + S3(localizer PNG)
  → Kafka(series.result.ready)
  → Notification Service → WebSocket push / webhook callback

Total E2E P95 target: < 90 seconds for a full TOF-MRA series
```

---

## 3. Backend Design

### 3.1 Service Breakdown

| Service | Language | Framework | Responsibility |
|---|---|---|---|
| **API Gateway** | — | Kong | Auth, rate-limit, routing, mTLS |
| **Upload Service** | Python | FastAPI | Multipart DICOM ingest, chunked upload, S3 multipart |
| **Preprocessing Service** | Python | FastAPI + Celery worker | DICOM→image, windowing, z-sort, tensor serialization |
| **Inference Service** | Python | Ray Serve | 6-model GPU ensemble, batching, result aggregation |
| **Result Service** | Python | FastAPI | Write/read predictions, serve localizer images |
| **Notification Service** | Python | FastAPI + WebSocket | Push results to connected clients, webhook delivery |
| **Audit Service** | Python | Kafka Consumer | HIPAA-compliant immutable access log |
| **Admin Service** | Python | FastAPI | Model versioning, configuration, feature flags |

### 3.2 API Design — REST (primary) + gRPC (internal GPU calls)

**External REST API** (OpenAPI 3.1, JSON)

```
POST   /v1/series                          → Upload DICOM series
GET    /v1/series/{series_id}              → Poll status
GET    /v1/series/{series_id}/predictions  → Get prediction scores
GET    /v1/series/{series_id}/localizer    → Get annotated slice image
DELETE /v1/series/{series_id}             → PHI deletion (GDPR/HIPAA)

POST   /v1/webhooks                        → Register result callback
GET    /v1/audit/{series_id}              → Access audit trail (admin)
```

**Internal gRPC** (between Preprocessing → Inference)
```protobuf
service InferenceService {
  rpc RunInference (InferenceRequest) returns (InferenceResponse);
  rpc GetModelStatus (Empty) returns (ModelStatusResponse);
}
message InferenceRequest {
  string series_id = 1;
  string s3_tensor_path = 2;
  InferenceConfig config = 3;  // batch_size, crop_rat, thresholds
}
```

### 3.3 Business Logic Organization

Strict Hexagonal (Ports & Adapters) within each service:
```
service/
├── domain/          # Pure business logic, no framework deps
│   ├── entities.py  # Series, Prediction, AuditEvent
│   ├── services.py  # InferencePipeline, EnsembleAggregator
│   └── events.py    # Domain events
├── adapters/
│   ├── inbound/     # FastAPI routes, Kafka consumers
│   └── outbound/    # S3 client, DB repos, Kafka producer
├── config/          # Settings, feature flags
└── main.py
```

The `RSNA_IAD` class from `core.py` becomes a **domain service** `InferencePipeline`, injected via DI. Model checkpoint paths are config-driven, not hardcoded.

---

## 4. Database & Storage

### 4.1 Database Selection

| Concern | Technology | Rationale |
|---|---|---|
| **Primary datastore** | PostgreSQL 16 (CockroachDB for multi-region) | ACID for PHI, rich JSON support for prediction blobs, proven at scale |
| **Session & hot-cache** | Redis 7 Cluster | Sub-millisecond prediction lookups, distributed locks for dedup |
| **Object storage** | S3-compatible (MinIO self-hosted / AWS S3) | DICOM files 50–500 MB each, processed tensors, localizer PNGs |
| **Event store** | Kafka with Tiered Storage (S3) | Infinite retention for compliance, event replay for retraining |
| **Audit log** | PostgreSQL + WORM S3 policy | Immutable, append-only, cryptographically signed rows |

### 4.2 Schema Design

```sql
-- Core tables (PostgreSQL, UUID PKs, all PHI encrypted at rest)

CREATE TABLE series (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    patient_id      UUID NOT NULL,              -- references de-identified patient
    upload_time     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    status          TEXT NOT NULL CHECK (status IN ('uploaded','preprocessing','inferencing','complete','failed')),
    modality        TEXT NOT NULL,              -- MR, CT
    dicom_s3_key    TEXT NOT NULL,              -- encrypted column
    processed_s3_key TEXT,
    series_uid      TEXT UNIQUE NOT NULL,       -- DICOM SeriesInstanceUID
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE predictions (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    series_id       UUID NOT NULL REFERENCES series(id),
    model_version   TEXT NOT NULL,              -- 'ensemble-v1.0.0'
    aneurysm_present FLOAT4 NOT NULL,
    location_scores JSONB NOT NULL,             -- {"Left MCA": 0.72, ...}
    localizer_s3_key TEXT,
    ensemble_weights JSONB NOT NULL,
    inference_time_ms INT NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE audit_log (
    id              BIGSERIAL PRIMARY KEY,
    event_time      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    actor_id        UUID NOT NULL,
    actor_type      TEXT NOT NULL,              -- 'clinician', 'system', 'admin'
    action          TEXT NOT NULL,              -- 'view', 'upload', 'delete'
    resource_type   TEXT NOT NULL,
    resource_id     UUID NOT NULL,
    ip_address      INET,
    signature       TEXT NOT NULL               -- HMAC-SHA256 of row
) PARTITION BY RANGE (event_time);
```

### 4.3 Indexing Strategy

```sql
-- Hot path: status polling
CREATE INDEX CONCURRENTLY idx_series_status ON series(status) WHERE status != 'complete';
-- Prediction lookup by series
CREATE INDEX CONCURRENTLY idx_predictions_series_id ON predictions(series_id);
-- Audit queries (HIPAA audit requests)
CREATE INDEX CONCURRENTLY idx_audit_resource ON audit_log(resource_id, event_time DESC);
CREATE INDEX CONCURRENTLY idx_audit_actor ON audit_log(actor_id, event_time DESC);
-- JSONB GIN for location score queries across anatomy classes
CREATE INDEX CONCURRENTLY idx_predictions_location_gin ON predictions USING GIN(location_scores);
```

### 4.4 Partitioning

- `audit_log`: Range-partitioned monthly by `event_time` → auto-archived to S3 Glacier after 7 years (HIPAA requirement)
- `series`: Partition by `upload_time` (quarterly) — queries are always recency-biased
- S3 lifecycle: DICOM raw → S3 Standard (30d) → S3 IA (1y) → Glacier (7y) → Delete

### 4.5 Object Storage Layout

```
s3://rsna-iad-medical/
├── dicom/raw/{series_id}/           ← Encrypted, KMS-managed key per series
├── processed/{series_id}/           ← Tensor files (.npy), 384x384x3
├── results/{series_id}/
│   ├── localizer.png                ← Annotated slice
│   └── scores.json
└── models/{version}/                ← Checkpoint registry
    ├── exp0_yolov11.pt
    ├── exp1_yolov5.pt
    ├── exp2_vit_large.pt
    └── ensemble_config.json
```

---

## 5. Scalability & Performance

### 5.1 Horizontal Scaling Strategy

```
                  Inference Cluster (GPU-bound, scale by queue depth)
                  ┌──────────────────────────────────────┐
                  │  HPA: target queue lag > 5 jobs      │
  kafka lag ──→   │  Node pool: A10G (24GB) or A100      │
                  │  Ray Serve: handle model reuse        │
                  │  Min: 2 replicas, Max: 20             │
                  └──────────────────────────────────────┘

                  Upload/Result Services (CPU-bound, scale by RPS)
                  ┌──────────────────────────────────────┐
                  │  HPA: target CPU 60%, P95 lat < 200ms│
                  │  Min: 3 replicas, Max: 50             │
                  └──────────────────────────────────────┘
```

**Key scaling invariant**: The GPU inference deployment is the bottleneck and pricing constraint. Scale it on Kafka consumer lag (KEDA), not CPU. Each A10G pod can load all 6 models (≈11 GB total VRAM) and process 1 series/60s with batching.

### 5.2 Load Balancing

- **External**: AWS ALB / Cloudflare → API Gateway (Kong)
- **Internal**: Kubernetes Service (iptables round-robin for stateless services)
- **Inference**: Ray Serve handles intra-cluster load balancing with max_ongoing_requests per replica
- **DB**: PgBouncer connection pooling (transaction mode), read replicas for result queries

### 5.3 Caching (Redis)

```python
# Cache prediction results — read-heavy after inference completes
PREDICTION_CACHE_KEY = "pred:{series_id}"
PREDICTION_TTL = 3600 * 24  # 24h — results don't change post-inference

# Cache model status for health UI
MODEL_STATUS_KEY = "model:status"
MODEL_STATUS_TTL = 30  # 30s refresh

# Distributed lock to prevent duplicate inference jobs
INFERENCE_LOCK_KEY = "lock:inference:{series_id}"
INFERENCE_LOCK_TTL = 300  # 5 min fence
```

CDN (CloudFront): Cache localizer PNG images (signed URLs, 1h TTL). These are immutable once generated.

### 5.4 Rate Limiting

Via Kong:
- Per-user: 100 series uploads/day, 1000 GET /predictions/day
- Per-IP: 20 concurrent connections
- PACS system integrations: dedicated API keys with higher quotas
- Burst: token bucket, 10 req/s sustained, 30 req/s burst

---

## 6. Async & Event Processing

### 6.1 Kafka Topic Design

```
Topic                         | Partitions | Retention | Key
------------------------------|------------|-----------|-----
series.uploaded               | 24         | 7d        | series_id
series.preprocessed           | 24         | 7d        | series_id
series.inference.complete     | 24         | 7d        | series_id
series.result.ready           | 24         | 72h       | series_id
series.failed                 | 12         | 30d       | series_id (DLQ)
audit.events                  | 12         | 365d      | actor_id
model.retraining.trigger      | 4          | 30d       | model_version
```

Partitioning by `series_id` (hash) guarantees ordering per series across the pipeline.

### 6.2 Consumer Groups

```
consumer-group: preproc-workers      → series.uploaded
consumer-group: inference-workers    → series.preprocessed
consumer-group: result-writers       → series.inference.complete
consumer-group: notifiers            → series.result.ready
consumer-group: audit-writers        → audit.events (all topics fan-out)
consumer-group: retraining-trigger   → series.result.ready (ML Ops feedback loop)
```

### 6.3 Background Jobs (Celery + Redis broker, separate from Kafka)

- **Celery Beat**: Scheduled DICOM cleanup (PHI deletion after retention period)
- **Celery Beat**: Daily model performance report generation
- **Celery Beat**: Audit log cryptographic integrity check (nightly)
- **Celery Worker**: Webhook delivery with exponential backoff + DLQ

### 6.4 Idempotency

Every Kafka consumer is idempotent:
- Check Redis lock before processing: `SETNX lock:inference:{series_id} 1 EX 300`
- Upsert on database write (INSERT ... ON CONFLICT DO NOTHING)
- At-least-once delivery + idempotent consumers = exactly-once semantics

---

## 7. Security

### 7.1 Authentication & Authorization

```
┌──────────────────────────────────────────────────────────────────┐
│  OIDC Provider (Keycloak self-hosted or Auth0)                    │
│  Tokens: short-lived JWT (15 min), refresh (7d), scope-scoped    │
└────────────────────────┬─────────────────────────────────────────┘
                         │
                      Kong (JWT validation at gateway, no service trusts raw tokens)
                         │
        ┌────────────────▼─────────────────────────────┐
        │         RBAC Roles                            │
        │  clinician    → upload, view own series       │
        │  radiologist  → view all series in org        │
        │  admin        → all + audit + model mgmt      │
        │  system       → internal service-to-svc (mTLS)│
        └───────────────────────────────────────────────┘
```

Service-to-service auth: **mTLS with SPIFFE/SPIRE** (Istio service mesh). No service trusts another without a verified SPIFFE ID. Zero trust network.

### 7.2 PHI Data Protection

| Layer | Control |
|---|---|
| Transit | TLS 1.3 everywhere; mTLS on internal mesh |
| DICOM at rest | AES-256, KMS key per series, envelope encryption |
| Database columns | pgcrypto for patient_id, dicom_s3_key (searchable encryption with deterministic mode) |
| Audit immutability | HMAC-SHA256 signature of each audit row with a rotating HMAC key stored in HSM |
| De-identification | DICOM tags stripped/replaced at Upload Service before storage (keep only SeriesInstanceUID) |
| Deletion | Crypto-shredding: destroy the KMS key → DICOM becomes permanently unrecoverable |

### 7.3 API Security

- Kong: OWASP Top-10 WAF (ModSecurity rules)
- Input validation: Pydantic v2 strict mode on all request models
- File upload: Magic byte validation for DICOM (`.dcm` + `DICM` at offset 128)
- SSRF prevention: No user-controlled URLs processed server-side
- Content-Security-Policy, HSTS, X-Frame-Options on all responses
- Secrets management: HashiCorp Vault (dynamic DB credentials, S3 presigned URL signing keys)

---

## 8. Reliability

### 8.1 Fault Tolerance Architecture

```
                        Failure Mode Map

Series Upload Service fails:
  → Upload retried client-side (SDK with exponential backoff)
  → S3 multipart upload is resumable

Kafka broker fails:
  → Kafka is multi-broker (RF=3, min.insync.replicas=2)
  → No data loss; producers retry with acks=all

Preprocessing Service crashes mid-job:
  → Job in Kafka is un-acked, re-consumed by another worker
  → S3 write is transactional (atomic PUT via presigned)

GPU Pod crashes during inference:
  → Ray Serve detects replica failure, restarts (< 10s)
  → Kafka offset not committed until result written to S3
  → Automatic retry via Kafka re-delivery

Database primary fails:
  → CockroachDB/PG HA: automatic failover < 30s
  → PgBouncer reconnects transparently

All 3 GPU nodes fail (AZ outage):
  → Spot interruption handler saves state to Kafka
  → Auto-provisioned in another AZ within 5 min
```

### 8.2 Retry Strategy

```python
# Per-service retry config (tenacity)
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception_type(TransientError),
    before_sleep=log_retry_attempt
)
async def call_inference_service(request): ...

# Kafka consumer retry: exponential backoff per partition
# After 3 retries → DLQ topic + alert
```

### 8.3 Circuit Breakers

Implemented via **Istio traffic policy** (no app-level library needed):
```yaml
trafficPolicy:
  outlierDetection:
    consecutiveGatewayErrors: 5
    interval: 30s
    baseEjectionTime: 30s
    maxEjectionPercent: 50
```

For external dependencies (S3, KMS): `tenacity` + custom `CircuitBreaker` class with half-open probing.

### 8.4 Disaster Recovery

| Metric | Target |
|---|---|
| **RTO** (Recovery Time Objective) | < 15 minutes |
| **RPO** (Recovery Point Objective) | < 1 minute (Kafka replication) |

- Multi-AZ deployment by default
- Cross-region S3 replication (active-passive) for DICOM data
- Kafka MirrorMaker 2 for cross-region event replication
- Daily PostgreSQL PITR snapshots (automated via RDS/CockroachDB)
- Runbook automation: `kubectl apply -f dr/restore.yaml` brings full stack in < 15 min
- Chaos engineering: monthly Chaos Monkey runs in staging

---

## 9. Observability

### 9.1 Three Pillars Stack

```
┌────────────────────────────────────────────────────────────────────┐
│  LOGGING       │ Structured JSON → Fluentd → OpenSearch (7d hot)  │
│                │ Long-term: S3 (1y) + Athena for audit queries     │
├────────────────────────────────────────────────────────────────────┤
│  METRICS       │ Prometheus + Grafana                              │
│                │ Business: inference_latency_p99, auc_drift        │
│                │ Infra: GPU_utilization, kafka_consumer_lag        │
│                │ SRE: error_rate, availability, saturation         │
├────────────────────────────────────────────────────────────────────┤
│  TRACING       │ OpenTelemetry → Jaeger / Tempo                    │
│                │ Trace ID propagated: upload → kafka → inference   │
│                │ GPU kernel time captured via NVTX annotations     │
└────────────────────────────────────────────────────────────────────┘
```

### 9.2 Key Metrics & Alerts

```yaml
# SLOs
inference_latency_p95: < 90s  # alert at 75s
inference_error_rate:  < 0.1% # alert at 0.01%
upload_success_rate:   > 99.9%
model_auc_drift:       alert if 7d rolling AUC drops > 2% vs baseline

# Infrastructure
gpu_memory_utilization: alert > 90%
kafka_consumer_lag:     alert > 50 messages (inference topic)
db_connection_pool:     alert > 80% pool saturation

# HIPAA-specific
unauthorized_access_attempts: alert any
audit_signature_mismatch:    alert any (critical)
phi_deletion_failures:        alert any (critical)
```

### 9.3 ML Model Monitoring

- **Prediction drift**: Track distribution of `aneurysm_present` scores daily
- **Confidence calibration**: Monitor ECE (Expected Calibration Error) on new data
- **Data drift**: PSI (Population Stability Index) on preprocessed image pixel statistics
- **Retraining trigger**: Kafka event `model.retraining.trigger` when drift threshold crossed
- Dashboard: Grafana + custom ML metrics panel showing per-anatomical-class AUC trends

---

## 10. DevOps & Deployment

### 10.1 CI/CD Pipeline

```
Developer pushes → GitHub Actions

┌─────────────────────────────────────────────────────────────────┐
│  PR Checks (< 5 min)                                            │
│  ├── pre-commit: ruff, mypy, bandit (security), hadolint        │
│  ├── unit tests: pytest -x --cov=90%                            │
│  └── security scan: trivy (container), semgrep (SAST)           │
├─────────────────────────────────────────────────────────────────┤
│  Merge to main → Integration CI (< 20 min)                      │
│  ├── Build Docker images (multi-stage, layer cache in ECR)      │
│  ├── Integration tests (testcontainers: Kafka+PG+MinIO in Docker)│
│  ├── Model smoke test: run inference on synthetic DICOM         │
│  └── Push to ECR with sha256 digest tag                         │
├─────────────────────────────────────────────────────────────────┤
│  Auto-deploy to Staging (Argo CD, GitOps)                       │
│  ├── Helm chart diff + approval gate                            │
│  ├── Smoke test suite runs against staging                      │
│  └── Performance regression: p95 latency must not regress > 10% │
├─────────────────────────────────────────────────────────────────┤
│  Production Deployment (manual approval OR off-hours auto)      │
│  ├── Blue-Green via Argo Rollouts                               │
│  ├── 10% canary → 50% → 100% (based on error rate SLO)         │
│  └── Automatic rollback if error rate > 0.5% in 5 min window   │
└─────────────────────────────────────────────────────────────────┘
```

### 10.2 Containerization

**Multi-stage Dockerfile (Inference Service)**:
```dockerfile
# Stage 1: Build dependencies
FROM python:3.11-slim AS builder
WORKDIR /build
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Stage 2: CUDA runtime (GPU inference)
FROM nvcr.io/nvidia/pytorch:24.04-py3 AS inference
WORKDIR /app
COPY --from=builder /root/.local /root/.local
COPY src/demo-test/ ./

# Non-root user (security hardening)
RUN useradd -r -u 1001 appuser && chown -R appuser:appuser /app
USER appuser

CMD ["python", "-m", "ray.dashboard", "--serve"]
```

CPU services use `python:3.11-slim` base — no CUDA bloat.

### 10.3 Kubernetes Architecture

```yaml
# GPU inference nodes
nodeSelector:
  cloud.google.com/gke-accelerator: nvidia-a10g
resources:
  limits:
    nvidia.com/gpu: "1"
    memory: "32Gi"
  requests:
    nvidia.com/gpu: "1"
    memory: "24Gi"

# KEDA autoscaler on Kafka lag
apiVersion: keda.sh/v1alpha1
kind: ScaledObject
spec:
  triggers:
  - type: kafka
    metadata:
      topic: series.preprocessed
      lagThreshold: "5"
```

### 10.4 Environment Strategy

| Env | Purpose | Infra |
|---|---|---|
| **local** | Developer iteration | docker-compose (CPU mode, fake GPU via mocked Ray) |
| **dev** | Shared integration | Small K8s cluster (1x T4 GPU), ephemeral per-PR branches |
| **staging** | Pre-prod gate | Production-parity (1x A10G), anonymized DICOM test cases |
| **production** | Live clinical | Multi-AZ, 2+ A10G workers, RDS Multi-AZ, Kafka RF=3 |

---

## 11. Codebase Structure

```
rsna-iad/
├── services/
│   ├── upload/
│   │   ├── domain/
│   │   │   ├── entities.py          # Series, DicomFile
│   │   │   └── services.py          # UploadOrchestrator
│   │   ├── adapters/
│   │   │   ├── inbound/
│   │   │   │   └── routes.py        # FastAPI /v1/series POST
│   │   │   └── outbound/
│   │   │       ├── s3_repo.py       # S3 multipart upload
│   │   │       └── kafka_producer.py
│   │   ├── Dockerfile
│   │   └── pyproject.toml
│   │
│   ├── preprocessing/
│   │   ├── domain/
│   │   │   └── pipeline.py          # dicom2image, windowing, z-sort
│   │   └── ...
│   │
│   ├── inference/
│   │   ├── domain/
│   │   │   ├── models/
│   │   │   │   ├── aneurysm_det.py  # YOLOv11 wrapper
│   │   │   │   ├── brain_det.py     # YOLOv5 wrapper
│   │   │   │   ├── cls_model.py     # ViT/EVA classifier
│   │   │   │   └── aux_model.py     # MiT-B4 FPN
│   │   │   ├── pipeline.py          # RSNA_IAD (refactored from core.py)
│   │   │   └── ensemble.py          # Weighted ensemble aggregation
│   │   ├── adapters/
│   │   │   ├── inbound/
│   │   │   │   ├── kafka_consumer.py
│   │   │   │   └── grpc_server.py
│   │   │   └── outbound/
│   │   │       ├── s3_repo.py       # Load tensors, write results
│   │   │       └── kafka_producer.py
│   │   ├── ray_serve_app.py         # Ray Serve deployment
│   │   ├── Dockerfile.gpu
│   │   └── pyproject.toml
│   │
│   ├── result/
│   ├── notification/
│   ├── audit/
│   └── admin/
│
├── libs/                            # Shared internal libraries (published as wheel)
│   ├── rsna_common/
│   │   ├── dicom.py                 # DICOM parse utilities (refactored from prepare/)
│   │   ├── auth.py                  # JWT validation, RBAC decorators
│   │   ├── observability.py         # OTel setup, structured logging
│   │   └── schemas.py               # Shared Pydantic models (Prediction, Series)
│   └── rsna_ml/
│       ├── preprocessing.py         # Image windowing, 3-channel stack
│       └── augmentations.py         # Albumentations pipelines
│
├── infrastructure/
│   ├── helm/
│   │   ├── inference/
│   │   ├── upload/
│   │   └── kafka/
│   ├── terraform/
│   │   ├── eks/                     # EKS cluster, node groups
│   │   ├── rds/                     # PostgreSQL Multi-AZ
│   │   └── s3/                      # Buckets, lifecycle, replication
│   └── k8s/
│       ├── keda/                    # ScaledObjects
│       ├── istio/                   # VirtualServices, mTLS policies
│       └── monitoring/              # Prometheus rules, Grafana dashboards
│
├── tests/
│   ├── unit/                        # Fast, no I/O
│   ├── integration/                 # testcontainers-based
│   └── e2e/                         # Full DICOM → prediction flow
│
├── training/                        # ML training code (not deployed to prod)
│   ├── exp0_aneurysm_det/           # <- current src/exp0_aneurysm_det
│   ├── exp1_brain_det/
│   ├── exp2_cls/
│   ├── exp3_aux/
│   ├── exp4_cls_pseudo/
│   ├── exp5_aux_pseudo/
│   └── prepare/
│
├── .github/
│   └── workflows/
│       ├── ci.yml
│       ├── cd-staging.yml
│       └── cd-production.yml
│
├── docker-compose.yml               # Local dev stack (no GPU required)
└── Makefile                         # dev, test, lint, deploy shortcuts
```

**Key principle**: `training/` is completely isolated from `services/`. Model weights are artifacts, not code. The `RSNA_IAD` class logic is refactored into `services/inference/domain/` — testable without GPU.

---

## 12. Evolution Plan

### Phase 1 — MVP (Months 1–3): Single-Tenant Clinical Pilot

**Architecture**: Simplified sync flow for a single hospital partner
```
Upload (FastAPI) → Queue (Redis/RQ) → Inference (GPU worker) → DB → REST response
```
- Single GPU server (A10G or RTX 4090)
- PostgreSQL (single node, daily backups)
- Basic auth (API key)
- MinIO on-prem for DICOM storage
- Manual deployment (Docker Compose)
- `core.py::RSNA_IAD` used as-is, wrapped in a FastAPI endpoint

**Milestones**: Predict on 100 real series, collect radiologist feedback, measure E2E latency.

### Phase 2 — Scale-Up (Months 4–9): Multi-Tenant SaaS

**Architecture transition to full event-driven design**:
- Introduce Kafka: decouple upload from inference
- Kubernetes: GPU autoscaling via KEDA
- Multi-tenant: org-scoped JWT, row-level security in PostgreSQL
- Audit logging (HIPAA): PostgreSQL audit table + S3 WORM
- HL7 FHIR R4 API adapter for EHR integration
- Model versioning: MLflow registry, model A/B testing via feature flags
- CDN caching of localizer images

**Milestones**: 10 hospital tenants onboarded, 1000 series/day, SLA 99.9%.

### Phase 3 — Scale to Millions (Year 2+): Global Medical AI Platform

**Architecture additions**:
- CockroachDB: multi-region active-active PostgreSQL
- Kafka MirrorMaker 2: cross-region event replication
- Ray on spot/preemptible GPUs (60% cost reduction)
- Federated learning module: hospitals contribute gradient updates, not raw PHI
- FHIR R4 native event streaming (replace REST polling)
- Model versioning: per-institution fine-tuning pipeline
- Global CDN: DICOM preview streaming (DICOMweb RS) via CloudFront+WADO

### Migration Strategy (MVP → Phase 2)

```
Week 1–2:  Deploy Kafka alongside existing Redis RQ. Mirror events.
Week 3:    Switch preprocessing to consume from Kafka (keep Redis fallback).
Week 4:    Switch inference to consume from Kafka. Retire Redis RQ.
Week 5:    Blue-green deploy K8s inference. Retire bare-metal GPU server.
Week 6:    Enable KEDA autoscaling. Load test.
```

No "big bang" migration. Each step is independently deployable and rollback-safe.

---

## 13. Trade-offs

### Why This Architecture Is Best

| Decision | Alternative Considered | Why This Wins |
|---|---|---|
| Kafka over RabbitMQ | RabbitMQ | Kafka's log-based storage enables event replay for retraining, audit, and debugging. DICOM uploads are bursty — Kafka's backpressure handling is superior. |
| Ray Serve over Triton | Nvidia Triton | Ray Serve natively handles heterogeneous Python models (YOLOv5 subprocess call, arbitrary SMP models). Triton requires ONNX/TensorRT export — risky for bleeding-edge timm models. |
| Hexagonal architecture over N-tier | Layered MVC | Domain logic is testable without GPU, Kafka, or database. ML engineers can iterate on `InferencePipeline` without spinning up K8s. |
| CockroachDB for Phase 3 | Citus/Partman on PG | CRDB provides transparent multi-region with serializable isolation. Citus sharding is operationally complex and requires manual shard management. |
| Async-first (Kafka) over sync gRPC | Synchronous chain | A GPU inference run can take 30–90s. Synchronous HTTP chains would time out, require complex retry logic, and waste client connections. Kafka makes backpressure natural. |
| mTLS + SPIFFE zero-trust | Network-level VPC isolation | VPC-level trust assumes everything inside the perimeter is safe. Zero-trust + SPIFFE provides cryptographic service identity — essential for HIPAA breach detection. |
| Crypto-shredding for PHI deletion | Record deletion | Disk blocks are not reliably zeroed. KMS key destruction makes DICOM cryptographically irrecoverable in < 1s without touching storage. |

### Known Trade-offs Accepted

1. **Operational complexity vs. MVP**: Kafka + K8s + mTLS is heavy for Phase 1. We mitigate this by using Docker Compose for MVP and designing services to be portable.
2. **Ray Serve cold start**: First request after pod scale-up takes 30–60s (model loading). Mitigated by minimum 2 warm replicas always running.
3. **Cost**: A10G GPU instances are expensive ($3–5/hr). Spot instances + request batching reduces this by 60%.
4. **CRDB licensing**: CockroachDB at scale requires a commercial license. PostgreSQL + read replicas + Citus is a fallback accepted trade-off.

---

## Recommended Architecture Summary

> **Build a GPU-inference, healthcare-grade, event-driven microservices platform on Kubernetes.**

**Core decisions, non-negotiable:**
1. **Kafka** as the spine — decouples every service, enables audit replay, survives any single point of failure
2. **Ray Serve** for GPU model serving — handles 6 heterogeneous PyTorch models with batching and auto-scaling
3. **PostgreSQL** (CRDB at scale) for all structured data — ACID for PHI, rich indexing, proven
4. **Hexagonal architecture** per service — domain logic isolated from infra, GPU-free unit testing
5. **mTLS + SPIFFE + KMS crypto-shredding** — the only acceptable HIPAA security posture
6. **KEDA on Kafka lag** for GPU autoscaling — scale exactly where the bottleneck is
7. **Blue-green + canary** deployments — zero-downtime model updates in a clinical environment

**Start simple** (Phase 1: Docker Compose + Redis RQ + FastAPI), but design the domain model and hexagonal boundaries from Day 1. Every layer is independently replaceable. The `RSNA_IAD.predict()` pipeline runs identically in a Jupyter notebook, a local FastAPI dev server, and a 20-node Ray Serve cluster.

**The diagnostic accuracy (0.89 AUC) is already world-class. The architecture's job is to deliver that accuracy reliably, securely, and at any scale — without the model ever becoming the bottleneck.**
