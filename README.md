# 🚗 AutoIQ — Autonomous Intelligence for Manufacturing

**Real-time failure prediction meets agent-driven root cause analysis**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![React](https://img.shields.io/badge/react-18+-61DAFB.svg)](https://reactjs.org/)

> **Built for**: EY-Techathon  
> **Category**: AI/ML + Industrial IoT

---

## 🎯 The Problem

Manufacturing systems fail. When they do, the cost is catastrophic:

- **$50K–$500K** per hour of downtime in automotive manufacturing
- **70%** of failures go undetected until it's too late
- Root cause analysis takes hours or days, delaying corrective action

Traditional monitoring systems are reactive. **AutoIQ is predictive.**

---

## 💡 What We Built

**AutoIQ** is an autonomous multi-agent system that predicts manufacturing equipment failures before they happen and automatically performs Root Cause Analysis (RCA) in real-time.

### Multi-Agent Architecture

```
┌─────────────────────────────────────────────────────┐
│                   Master Agent                      │
│           (Orchestration + Risk Fusion)             │
└──────────────┬──────────────────────┬───────────────┘
               │                      │
       ┌───────▼────────┐    ┌────────▼──────────┐
       │  ML Risk Agent │    │  Rule-Based Agent │
       │  (RandomForest)│    │  (Safety Limits)  │
       └───────┬────────┘    └────────┬──────────┘
               │                      │
               └──────────┬───────────┘
                          │
              ┌───────────▼────────────┐
              │    UEBA Audit Agent    │
              │  (Behavioral Logging)  │
              └───────────┬────────────┘
                          │
              ┌───────────▼────────────┐
              │  Manufacturing Agent   │
              │  (RCA + CAPA Logging)  │
              └────────────────────────┘
```

### How It Works

1. **Telemetry Ingestion** — Sensor data arrives (air temp, process temp, RPM, torque, tool wear, machine type)
2. **Dual-Track Risk Assessment**:
   - **ML Engine**: Calibrated Random Forest produces a failure probability
   - **Physics Engine**: Rule-based safety thresholds produce a rule risk score
   - **Risk Fusion**: Weighted ensemble — `Final Risk = 0.65 × ML + 0.35 × Rules`
3. **State Classification**: `NORMAL → DEGRADED → CRITICAL → EMERGENCY`
4. **Autonomous RCA**: When state hits EMERGENCY, the Manufacturing Agent auto-logs root cause to CSV
5. **UEBA Tracking**: Every agent action is time-stamped and audited for compliance and debugging
6. **Gemini-Powered RCA** *(Frontend)*: On EMERGENCY states, Gemini API generates structured RCA + CAPA JSON

---

## 🔥 Key Features

### Hybrid Intelligence
- **Machine Learning**: Calibrated Random Forest (200 trees, `max_depth=10`, `class_weight={0:1, 1:6}`) trained on the AI4I manufacturing dataset
- **Rule-Based Logic**: Hard safety limits on process temp, RPM, torque, and tool wear for interpretability
- **Smart Fusion**: 65% ML + 35% rules — accuracy and explainability in one score

### Real Model Performance (Actual Evaluation Output)

```
=== AutoIQ Model Evaluation ===

Confusion Matrix:
[[1916   16]
 [  18   50]]

Classification Report:
              precision    recall  f1-score   support
           0       0.99      0.99      0.99      1932
           1       0.76      0.74      0.75        68
    accuracy                           0.98      2000
   macro avg       0.87      0.86      0.87      2000
weighted avg       0.98      0.98      0.98      2000

ROC-AUC Score: 0.9732
Decision threshold used: 0.30
```

> Threshold is set to **0.30** (not 0.70) to maximise recall on rare failure events — catching more true positives at the cost of a small precision trade-off.

### Autonomous Root Cause Analysis
- RCA triggered automatically on EMERGENCY state
- Root cause (`High thermal stress + excessive tool wear`) logged with timestamp to `logs/manufacturing_feedback.csv`
- Frontend calls Gemini API for a structured `rootCause / remedy / capa` JSON response

### UEBA Audit Trail
- Every agent action logged to `logs/ueba_log.csv` with `timestamp | action | state`
- Full audit trail readable via `/api/ueba` REST endpoint
- Designed for compliance with ISO 9001, FDA 21 CFR Part 11

### Safe Mode Failover
- If the ML model or scaler is missing/fails, Master Agent catches the exception
- System returns `state: "SAFE MODE"` and continues without crashing
- Zero downtime during model updates or file errors

### Real-Time Dashboard (4 Views)
| View | Description |
|------|-------------|
| **Control Center** | Telemetry input sliders + risk score + agent log console |
| **Vehicle State** | System health visualization |
| **RCA / CAPA** | Gemini-generated root cause and corrective actions |
| **Agent Audit** | UEBA compliance log feed |

---

## 🛠️ Technical Architecture

### Backend (Python + Flask)

```
backend/
├── agents/
│   ├── master_agent.py          # Orchestration, state classification, safe mode
│   ├── ueba_agent.py            # CSV-based audit logging
│   └── manufacturing_agent.py   # RCA CSV logging
├── models/
│   ├── failure_model.pkl        # Calibrated RandomForest (serialised)
│   └── scaler.pkl               # StandardScaler (serialised)
├── evaluation/
│   └── evaluate_model.py        # Standalone evaluation script
├── tests/
│   └── test_sanity.py           # Sanity test runner
├── predict.py                   # Risk computation engine (ML + rules + fusion)
├── train_model.py               # Full training pipeline
└── app.py                       # Flask REST API (3 endpoints)
```

**REST API Endpoints:**

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/analyze` | Run full agent pipeline on telemetry input |
| `GET` | `/api/ueba` | Retrieve UEBA audit log entries |
| `GET` | `/api/rca` | Retrieve most recent RCA log entry |

**Dependencies:** `flask`, `flask-cors`, `pandas`, `numpy`, `scikit-learn`, `joblib`

### Frontend (React + TypeScript)

```
frontend/
├── App.tsx                      # Main dashboard + 4-view routing
├── types.ts                     # TypeScript interfaces & enums
├── constants.tsx                # System limits, risk weights, risk band thresholds
├── services/
│   └── geminiService.ts         # Gemini API structured RCA generation
├── index.tsx                    # Entry point
├── index.html
└── vite.config.ts               # Vite + proxy to Flask :5000
```

**Key Tech:** React 18 · TypeScript · Tailwind CSS · Vite · `@google/genai`

Vite proxies `/api/*` requests to `http://127.0.0.1:5000`, so no CORS issues in dev.

### ML Pipeline

**Dataset**: AI4I Manufacturing Dataset (10,000 samples)  
**Features**: Air temperature [K], Process temperature [K], Rotational speed [rpm], Torque [Nm], Tool wear [min], Machine Type (L/M/H encoded as 0/1/2)  
**Target**: `Machine failure` (binary)

**Training Pipeline:**
1. Drop `UDI` and `Product ID` columns
2. Encode `Type` → `{L:0, M:1, H:2}`
3. `StandardScaler` normalisation
4. 80/20 stratified train-test split (random_state=42)
5. `RandomForestClassifier(n_estimators=200, max_depth=10, class_weight={0:1, 1:6})`
6. `CalibratedClassifierCV(method="sigmoid", cv=5)` — Platt scaling for true probabilities
7. Custom decision threshold: **0.30** (optimised for rare-failure recall)

**Rule-Based Safety Limits:**

| Parameter | Threshold |
|-----------|-----------|
| Process Temperature | > 310 K |
| Rotational Speed | > 1700 rpm |
| Torque | > 55 Nm |
| Tool Wear | > 200 min |

Each violation adds 25 points to the rule risk score (max 100).

**State Thresholds:**

| State | Final Risk Score |
|-------|-----------------|
| NORMAL | < 50 |
| DEGRADED | ≥ 50 |
| CRITICAL | ≥ 75 |
| EMERGENCY | ≥ 90 |

---

## 🚀 Getting Started

### Prerequisites

```
python 3.9+
node 18+
npm or yarn
AI4I_dataset.csv (place in backend/)
```

### Installation

**Backend:**
```bash
cd backend
pip install -r requirements.txt
python train_model.py        # Trains model, saves to models/
python app.py                # Flask server on :5000
```

**Frontend:**
```bash
cd frontend
npm install
npm run dev                  # Vite dev server on :5173
```

> The Vite dev server proxies `/api` requests to Flask at `http://127.0.0.1:5000` — no extra config needed.

### Quick Test

1. Open `http://localhost:5173`
2. Input telemetry values (or use the defaults)
3. Click **Run Analysis**
4. Watch the agents fire in real-time in the activity log

**Emergency Trigger Example:**
```
Air Temp:       305 K
Process Temp:   320 K
RPM:            1800
Torque:         60 Nm
Tool Wear:      220 min
```
→ Rule risk: **100** (all 4 limits breached)  
→ ML risk: high  
→ Final risk: **~95%** → State: **EMERGENCY** → RCA auto-logged

---

## 📊 Agent Workflow

```
User inputs telemetry
       ↓
Master Agent receives data
       ↓
  ┌────┴────┐
  │         │
ML Agent  Rule Agent
  │         │
  └────┬────┘
       ↓
Risk Fusion (0.65 × ML + 0.35 × Rules)
       ↓
State Classification (NORMAL / DEGRADED / CRITICAL / EMERGENCY)
       ↓
UEBA Agent logs action to ueba_log.csv
       ↓
IF state == EMERGENCY:
    Manufacturing Agent logs RCA to manufacturing_feedback.csv
    Frontend calls Gemini API → structured rootCause + remedy + capa
       ↓
API returns JSON to frontend dashboard
```

---

## 🧠 Technical Decisions

**Why threshold = 0.30 instead of 0.5?**  
Machine failures are rare (class imbalance). A lower threshold catches more true positives at the cost of some false positives — the right trade-off when a missed failure costs $500K/hour.

**Why `class_weight={0:1, 1:6}`?**  
Directly penalises the model 6× more for missing a failure than for a false alarm, reinforcing the recall-first strategy alongside the low threshold.

**Why Calibrated Probabilities?**  
Raw Random Forest probabilities are overconfident and poorly calibrated. Platt scaling (`sigmoid` method, 5-fold CV) converts them into reliable probability estimates — critical for meaningful risk thresholds.

**Why 65% ML / 35% Rules?**  
ML handles complex non-linear patterns; rules enforce hard physical safety limits that must always be respected. The split gives ML primacy while ensuring physics can't be overridden.

**Why Flask over FastAPI?**  
Simplicity and speed for the hackathon timeline. The 3-endpoint API needs no async complexity, and Flask-CORS is trivial to configure.

**Why CSV logging for UEBA/RCA?**  
Zero-dependency persistence — no database setup required. Logs are human-readable, inspectable, and compliant audit artifacts. Easy to swap for a real DB in production.

---

## 🔮 Future Roadmap

- **Predictive Maintenance Scheduling** — Auto-schedule tool replacements before threshold breach
- **Multi-Plant Federation** — Centralised monitoring dashboard across factory sites
- **Anomaly Detection** — Unsupervised learning for novel, never-before-seen failure modes
- **Digital Twin Integration** — Sync with Siemens/PTC virtual factory models
- **MQTT / OPC-UA** — Native industrial IoT protocol support
- **Mobile Push Alerts** — EMERGENCY state notifications on mobile
- **Database Backend** — Replace CSV logs with PostgreSQL/TimescaleDB for scale

---

## 📁 Project Structure

```
AutoIQ/
├── backend/
│   ├── agents/
│   ├── evaluation/
│   ├── models/
│   ├── tests/
│   ├── app.py
│   ├── predict.py
│   ├── train_model.py
│   └── requirements.txt
├── frontend/
│   ├── services/
│   ├── App.tsx
│   ├── types.ts
│   ├── constants.tsx
│   └── vite.config.ts
├── .gitignore
└── README.md
```

---

## 📄 License

MIT License — Use this to save factories from downtime!

---

## 👥 Team

**Paridhi Sharma** — Full-stack engineering · ML training & pipeline · Frontend + UX

---

*Built with 🔥 for EY-Techathon*
