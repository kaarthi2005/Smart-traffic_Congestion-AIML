
<p align="center">
  <h1 align="center">🚦 SMART TRAFFIC AI</h1>
  <h3 align="center">Software Test Report</h3>
  <p align="center">Traffic Prediction & Emergency Route Optimization System</p>
</p>

---

<table>
  <tr><td><b>Document Title</b></td><td>Smart Traffic AI — Software Test Report</td></tr>
  <tr><td><b>Project Name</b></td><td>Smart Traffic Prediction & Emergency Route Optimization</td></tr>
  <tr><td><b>Version</b></td><td>1.0.0</td></tr>
  <tr><td><b>Date</b></td><td>September 06, 2026</td></tr>
  <tr><td><b>Prepared By</b></td><td>Kaarthi</td></tr>
  <tr><td><b>Platform</b></td><td>Windows 10/11 (AMD64)</td></tr>
  <tr><td><b>Overall Status</b></td><td>✅ <b>ALL TESTS PASSED</b></td></tr>
</table>

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Test Environment](#2-test-environment)
3. [Test Summary](#3-test-summary)
4. [Test Case Results](#4-test-case-results)
   - 4.1 [Environment & Setup](#41-tc-01--environment--setup)
   - 4.2 [Data Integrity Validation](#42-tc-02--data-integrity-validation)
   - 4.3 [SUMO Network Validation](#43-tc-03--sumo-network-validation)
   - 4.4 [LSTM Traffic Prediction](#44-tc-04--lstm-traffic-prediction)
   - 4.5 [Emergency Route Optimization](#45-tc-05--emergency-route-optimization)
   - 4.6 [Dashboard UI & Rendering](#46-tc-06--dashboard-ui--rendering)
   - 4.7 [Module Integration](#47-tc-07--module-integration)
5. [Data Analysis](#5-data-analysis)
6. [Performance Metrics](#6-performance-metrics)
7. [Warnings & Observations](#7-warnings--observations)
8. [Files Under Test](#8-files-under-test)
9. [Conclusion & Recommendation](#9-conclusion--recommendation)

---

## 1. Introduction

### 1.1 Purpose

This document presents the results of functional testing performed on the **Smart Traffic AI** system. The system combines SUMO microscopic traffic simulation, LSTM deep learning prediction, and AI-powered emergency route optimization to deliver real-time traffic insights through an interactive Streamlit dashboard.

### 1.2 Scope

Testing covers the following functional areas:

- **Data Layer** — CSV data integrity, schema validation, and null checks
- **Simulation Layer** — SUMO network XML parsing and topology verification
- **Prediction Layer** — LSTM model loading, scaler loading, and prediction accuracy
- **Optimization Layer** — Route cost calculation, path finding, and route comparison
- **Presentation Layer** — Streamlit dashboard rendering, interactive elements, and visualizations
- **Integration Layer** — Cross-module imports and data flow

### 1.3 Technologies Under Test

| Technology | Role |
|---|---|
| Python | Core programming language |
| Streamlit | Interactive web dashboard |
| TensorFlow / Keras | LSTM deep learning model |
| NetworkX | Graph-based route optimization |
| SUMO | Microscopic traffic simulation |
| Pandas | Data processing and analysis |
| Matplotlib | Data visualization |
| Scikit-learn | Feature scaling (StandardScaler) |

---

## 2. Test Environment

### 2.1 System Configuration

| Component | Specification |
|---|---|
| Operating System | Windows 10/11 (64-bit) |
| Architecture | AMD64 |
| Python Runtime | 3.13.14 (MSC v.1944 64 bit) |
| Virtual Environment | `venv/` (project-local) |

### 2.2 Dependency Versions

| Package | Version | Status |
|---|---|---|
| Streamlit | 1.63.0 | ✅ Installed |
| TensorFlow | 2.21.0 | ✅ Installed |
| Pandas | 3.0.5 | ✅ Installed |
| NumPy | 2.5.2 | ✅ Installed |
| Scikit-learn | 1.9.0 | ✅ Installed |
| NetworkX | 3.6.1 | ✅ Installed |
| Matplotlib | 3.11.1 | ✅ Installed |
| Joblib | 1.6.0 | ✅ Installed |

---

## 3. Test Summary

### 3.1 Results Overview

| Category | Test Cases | ✅ Passed | ❌ Failed | ⚠️ Warnings |
|---|---|---|---|---|
| TC-01: Environment & Setup | 3 | 3 | 0 | 0 |
| TC-02: Data Integrity | 6 | 6 | 0 | 0 |
| TC-03: SUMO Network | 3 | 3 | 0 | 0 |
| TC-04: LSTM Prediction | 4 | 4 | 0 | 0 |
| TC-05: Route Optimization | 4 | 4 | 0 | 0 |
| TC-06: Dashboard UI | 3 | 3 | 0 | 3 |
| TC-07: Module Integration | 2 | 2 | 0 | 0 |
| **TOTAL** | **25** | **25** | **0** | **3** |

### 3.2 Pass Rate

```
████████████████████████████████████████  100%  (25/25 PASSED)
```

---

## 4. Test Case Results

### 4.1 TC-01 — Environment & Setup

| Test ID | Description | Expected Result | Actual Result | Status |
|---|---|---|---|---|
| TC-01.1 | Virtual environment exists | `venv/` directory with Python binaries | `venv/Scripts/`, `venv/Lib/`, `venv/Include/` found | ✅ PASS |
| TC-01.2 | All dependencies installed | All 9 packages importable | All packages import successfully without errors | ✅ PASS |
| TC-01.3 | Streamlit server starts | Server binds to port 8501 | Server running at `http://localhost:8501` | ✅ PASS |

---

### 4.2 TC-02 — Data Integrity Validation

| Test ID | Description | Expected Result | Actual Result | Status |
|---|---|---|---|---|
| TC-02.1 | `traffic.csv` loads | Valid CSV, no parse errors | Loaded: 15,686 rows × 6 columns | ✅ PASS |
| TC-02.2 | `traffic.csv` schema | Columns: `time, vehicle_id, speed, edge, x, y` | Schema matches exactly | ✅ PASS |
| TC-02.3 | `traffic.csv` null values | Zero nulls | **0 null values** found | ✅ PASS |
| TC-02.4 | `sumo_traffic.csv` loads | Valid CSV | Loaded: 2,181 rows × 4 columns | ✅ PASS |
| TC-02.5 | `sumo_traffic_features.csv` schema | Columns: `time, average_speed, vehicle_count, congestion` | Schema matches exactly | ✅ PASS |
| TC-02.6 | `sumo_traffic_features.csv` nulls | Zero nulls | **0 null values** found | ✅ PASS |

**Evidence — Data file statistics:**

```
traffic.csv
├── Rows:            15,686
├── Columns:         6 (time, vehicle_id, speed, edge, x, y)
├── Null values:     0
├── Unique vehicles: 150
└── Unique edges:    103

sumo_traffic.csv
├── Rows:            2,181
├── Columns:         4 (time, vehicle_id, road, speed)
├── Null values:     0
├── Unique vehicles: 50
└── Unique roads:    74

sumo_traffic_features.csv
├── Rows:            100
├── Columns:         4 (time, average_speed, vehicle_count, congestion)
├── Null values:     0
└── Congestion:      81 MEDIUM, 19 LOW
```

---

### 4.3 TC-03 — SUMO Network Validation

| Test ID | Description | Expected Result | Actual Result | Status |
|---|---|---|---|---|
| TC-03.1 | `network.net.xml` parses | Valid XML without errors | Parsed successfully via ElementTree | ✅ PASS |
| TC-03.2 | Network topology correct | 3×3 grid: 9 nodes, 24 bidirectional edges | **9 nodes, 24 edges** confirmed | ✅ PASS |
| TC-03.3 | Node/Edge IDs valid | Nodes A0–C2, edges with proper naming convention | All IDs match expected pattern | ✅ PASS |

**Evidence — Network topology:**

```
Nodes (9):
  A0    A1    A2
  B0    B1    B2
  C0    C1    C2

Grid Layout (3×3 bidirectional):

  A0 ←——→ A1 ←——→ A2
  ↕        ↕        ↕
  B0 ←——→ B1 ←——→ B2
  ↕        ↕        ↕
  C0 ←——→ C1 ←——→ C2

Edges (24):
  A0A1, A0B0, A1A0, A1A2, A1B1, A2A1, A2B2,
  B0A0, B0B1, B0C0, B1A1, B1B0, B1B2, B1C1,
  B2A2, B2B1, B2C2, C0B0, C0C1, C1B1, C1C0,
  C1C2, C2B2, C2C1
```

---

### 4.4 TC-04 — LSTM Traffic Prediction

| Test ID | Description | Expected Result | Actual Result | Status |
|---|---|---|---|---|
| TC-04.1 | Model file loads | `traffic_lstm_sumo.keras` loads via TensorFlow | ✅ Loaded (403 KB) | ✅ PASS |
| TC-04.2 | Scaler file loads | `scaler_sumo.pkl` loads via Joblib | ✅ Loaded (743 bytes) | ✅ PASS |
| TC-04.3 | End-to-end prediction | Returns dict with `current_speed`, `predicted_speed`, `congestion` | ✅ Valid dict returned | ✅ PASS |
| TC-04.4 | Output value ranges | Speed > 0, congestion ∈ {LOW, MEDIUM, HIGH} | ✅ All within range | ✅ PASS |

**Evidence — Prediction output:**

```
==============================
 LSTM TRAFFIC PREDICTION
==============================
 Current average speed:    9.42
 Predicted average speed:  8.57
 Predicted congestion:     MEDIUM
 Speed trend:              ▼ Declining (-0.85)
==============================
```

**Prediction Pipeline:**

```
sumo_traffic_features.csv
    → Load last 10 time steps
    → Scale with StandardScaler (scaler_sumo.pkl)
    → Reshape to (1, 10, 2) tensor
    → Feed to LSTM model (traffic_lstm_sumo.keras)
    → Inverse transform prediction
    → Classify congestion level
    → Return {current_speed, predicted_speed, congestion}
```

---

### 4.5 TC-05 — Emergency Route Optimization

| Test ID | Description | Expected Result | Actual Result | Status |
|---|---|---|---|---|
| TC-05.1 | Full pipeline executes | No exceptions, returns result dict | ✅ Complete execution | ✅ PASS |
| TC-05.2 | Optimal route found | Valid shortest path A0 → C2 | **A0 → A1 → A2 → B2 → C2** | ✅ PASS |
| TC-05.3 | Route comparison | All simple paths enumerated and ranked | **12 routes** ranked by cost | ✅ PASS |
| TC-05.4 | Cost calculation | Total cost = Σ(edge costs) | **161.69** (sum verified) | ✅ PASS |

**Evidence — Optimal route details:**

```
========================================
     AI OPTIMAL EMERGENCY ROUTE
========================================
 A0 → A1 → A2 → B2 → C2

 Total Cost: 161.69
 Segments:   4
 Source:     A0
 Dest:      C2
========================================
```

**Evidence — Per-edge breakdown:**

| Segment | Road | Speed (Current) | Speed (Predicted) | Vehicles | Penalty | Cost |
|---|---|---|---|---|---|---|
| A0 → A1 | A0A1 | 9.50 | 9.26 | 3 | 2.61 | 53.33 |
| A1 → A2 | A1A2 | 9.46 | 9.33 | 1 | 2.60 | 52.88 |
| A2 → B2 | A2B2 | 11.20 | 9.87 | 2 | 2.10 | 40.44 |
| B2 → C2 | B2C2 | 13.89 | 12.60 | 0 | 1.00 | 15.04 |

**Evidence — All 12 route alternatives:**

| Rank | Route | Cost | Δ vs Best |
|---|---|---|---|
| 1 ⭐ | A0 → A1 → A2 → B2 → C2 | 161.69 | — |
| 2 | A0 → A1 → B1 → C1 → C2 | 173.88 | +7.5% |
| 3 | A0 → A1 → B1 → B2 → C2 | 174.86 | +8.1% |
| 4 | A0 → B0 → B1 → C1 → C2 | 253.81 | +57.0% |
| 5 | A0 → B0 → B1 → B2 → C2 | 254.80 | +57.6% |
| 6 | A0 → B0 → C0 → C1 → C2 | 293.85 | +81.7% |
| 7 | A0 → B0 → B1 → A1 → A2 → B2 → C2 | 308.00 | +90.5% |
| 8 | A0 → A1 → A2 → B2 → B1 → C1 → C2 | 324.71 | +100.8% |
| 9 | A0 → A1 → B1 → B0 → C0 → C1 → C2 | 337.94 | +109.0% |
| 10 | A0 → B0 → C0 → C1 → B1 → B2 → C2 | 397.17 | +145.6% |
| 11 | A0 → B0 → C0 → C1 → B1 → A1 → A2 → B2 → C2 | 450.38 | +178.5% |
| 12 | A0 → A1 → A2 → B2 → B1 → B0 → C0 → C1 → C2 | 488.77 | +202.3% |

**Cost Savings:** Best route saves **327.08 cost units (66.9%)** vs worst route.

---

### 4.6 TC-06 — Dashboard UI & Rendering

| Test ID | Description | Expected Result | Actual Result | Status |
|---|---|---|---|---|
| TC-06.1 | Server responds to HTTP | HTTP 200 at localhost:8501 | ✅ Page loads with full content | ✅ PASS |
| TC-06.2 | All 12 sections render | No Python tracebacks in logs | ✅ All sections render | ✅ PASS |
| TC-06.3 | Interactive elements work | Buttons trigger actions | ✅ Both buttons functional | ✅ PASS |

**Evidence — All 12 dashboard sections verified:**

| # | Section | Component | Data Source | Verified |
|---|---|---|---|---|
| 1 | Page Title & Description | `st.title`, `st.markdown` | Static text | ✅ |
| 2 | Top Metrics Bar | 4× `st.metric` | `traffic.csv` | ✅ |
| 3 | Traffic Speed Trend | `st.pyplot` (line chart) | `traffic.csv → speed` | ✅ |
| 4 | AI Traffic Prediction | `st.button` → subprocess | `predict_traffic.py` | ✅ |
| 5 | Road-Level Analysis | `st.dataframe` | `traffic.csv` grouped by edge | ✅ |
| 6 | Congestion Distribution | `st.pyplot` (bar chart) | Speed → classification | ✅ |
| 7 | Traffic Level Summary | 4× `st.metric` | Speed range counts | ✅ |
| 8 | Overall Traffic Status | `st.info` | Average speed threshold | ✅ |
| 9 | Emergency Route Optimization | `st.button` → `optimize_route()` | `smart_route.py` | ✅ |
| 10 | Route Comparison | `st.dataframe` + `st.pyplot` | All simple paths | ✅ |
| 11 | Network Map | `st.pyplot` (scatter + lines) | `network.net.xml` geometry | ✅ |
| 12 | Footer | `st.markdown` | Static text | ✅ |

**Expected metric values on page load:**

| Metric | Expected Value |
|---|---|
| Average Speed | 7.19 |
| Vehicles | 150 |
| Traffic Records | 15,686 |
| Network Roads | 24 |
| Overall Status | 🟠 HIGH |

---

### 4.7 TC-07 — Module Integration

| Test ID | Description | Expected Result | Actual Result | Status |
|---|---|---|---|---|
| TC-07.1 | Dashboard → Optimization import | `from optimization.smart_route import optimize_route` succeeds | ✅ Import successful | ✅ PASS |
| TC-07.2 | Optimization → Prediction import | `from prediction.predict_traffic import predict_traffic` succeeds | ✅ Import successful | ✅ PASS |

**Evidence — Full integration dependency graph:**

```
┌─────────────────────────────────────────────────────────┐
│                    dashboard.py                          │
│                  (Streamlit App)                          │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌──────────────────┐    ┌────────────────────────────┐ │
│  │  data/            │    │  simulation/               │ │
│  │  traffic.csv      │───→│  network.net.xml           │ │
│  │  (15,686 rows)    │    │  (9 nodes, 24 edges)       │ │
│  └──────────────────┘    └────────────────────────────┘ │
│           │                            │                 │
│           ▼                            ▼                 │
│  ┌─────────────────────────────────────────────────┐    │
│  │        optimization/smart_route.py               │    │
│  │            optimize_route()                       │    │
│  ├─────────────────────────────────────────────────┤    │
│  │  • Loads SUMO network                            │    │
│  │  • Reads sumo_traffic.csv                        │    │
│  │  • Applies traffic conditions                    │    │
│  │  • Calls LSTM prediction                         │    │
│  │  • Calculates route costs                        │    │
│  │  • Finds optimal path (Dijkstra)                 │    │
│  │  • Ranks all alternative routes                  │    │
│  └──────────────────┬──────────────────────────────┘    │
│                      │                                   │
│                      ▼                                   │
│  ┌─────────────────────────────────────────────────┐    │
│  │        prediction/predict_traffic.py              │    │
│  │            predict_traffic()                       │    │
│  ├─────────────────────────────────────────────────┤    │
│  │  • Reads sumo_traffic_features.csv               │    │
│  │  • Loads scaler_sumo.pkl                         │    │
│  │  • Loads traffic_lstm_sumo.keras                  │    │
│  │  • Predicts next speed value                     │    │
│  │  • Classifies congestion level                   │    │
│  └─────────────────────────────────────────────────┘    │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

---

## 5. Data Analysis

### 5.1 Traffic Speed Distribution

```
Speed Range      Count     Bar
─────────────────────────────────────────────────
SEVERE  (< 5)    6,304     ████████████████████░░░░░░░░░░░  40.2%
MEDIUM  (10–20)  7,145     ██████████████████████░░░░░░░░░  45.5%
HIGH    (5–10)   2,237     ███████░░░░░░░░░░░░░░░░░░░░░░░  14.3%
LOW     (≥ 20)       0     ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   0.0%
```

### 5.2 Top Congested Roads

| Rank | Road | Vehicle Count |
|---|---|---|
| 1 | B1B2 | Highest traffic |
| 2 | B1A1 | High traffic |
| 3 | B0B1 | High traffic |

### 5.3 LSTM Feature Distribution

| Feature | Mean | Description |
|---|---|---|
| Average Speed | 10.24 | Aggregated per time step |
| Vehicle Count | 20.58 | Unique vehicles per step |
| Congestion | 81% MEDIUM, 19% LOW | Classification labels |

---

## 6. Performance Metrics

### 6.1 Route Optimization Performance

| Metric | Value |
|---|---|
| Network size | 9 nodes, 24 edges |
| Routes evaluated | 12 |
| Optimal route cost | 161.69 |
| Worst route cost | 488.77 |
| Cost reduction | 66.9% |
| Route segments (optimal) | 4 |

### 6.2 LSTM Prediction Performance

| Metric | Value |
|---|---|
| Input sequence length | 10 time steps |
| Feature dimensions | 2 (speed, vehicle count) |
| Model file size | 403 KB |
| Current speed | 9.42 |
| Predicted speed | 8.57 |
| Speed change | -0.85 (9.0% decline) |
| Prediction | Congestion increasing |

---

## 7. Warnings & Observations

| ID | Type | Description | Severity | Recommended Action |
|---|---|---|---|---|
| W-01 | ⚠️ Deprecation | `use_container_width=True` is deprecated in Streamlit ≥1.63. Appears 3 times in `dashboard.py`. | **Low** | Replace with `width='stretch'` |
| W-02 | ℹ️ Informational | TensorFlow GPU not available on native Windows. CPU inference is used. | **None** | Expected on Windows without WSL2 |
| W-03 | ℹ️ Informational | TensorFlow oneDNN custom operations enabled. May produce minor floating-point variance. | **None** | No action needed |

> **Note:** All warnings are non-blocking and do not affect system functionality.

---

## 8. Files Under Test

| # | File Path | Description | Size | Lines | Status |
|---|---|---|---|---|---|
| 1 | `dashboard.py` | Main Streamlit dashboard | 42,322 B | 2,102 | ✅ Verified |
| 2 | `optimization/smart_route.py` | AI route optimization engine | 13,128 B | 543 | ✅ Verified |
| 3 | `prediction/predict_traffic.py` | LSTM prediction module | 1,826 B | 87 | ✅ Verified |
| 4 | `prediction/traffic_lstm_sumo.keras` | Trained LSTM model | 403,405 B | — | ✅ Verified |
| 5 | `prediction/scaler_sumo.pkl` | Feature scaler (StandardScaler) | 743 B | — | ✅ Verified |
| 6 | `simulation/network.net.xml` | SUMO road network definition | 47,787 B | — | ✅ Verified |
| 7 | `data/traffic.csv` | Main traffic dataset | 868,463 B | 15,687 | ✅ Verified |
| 8 | `data/sumo_traffic.csv` | SUMO simulation traffic data | 37,388 B | 2,182 | ✅ Verified |
| 9 | `data/sumo_traffic_features.csv` | Engineered features for LSTM | 1,905 B | 101 | ✅ Verified |
| 10 | `dashboard/emergency_map.py` | Emergency route map page | 1,806 B | — | ✅ Verified |
| 11 | `evaluation/evaluate_routes.py` | Route evaluation harness | 6,617 B | — | ✅ Verified |

---

## 9. Conclusion & Recommendation

### 9.1 Test Results

<table>
  <tr>
    <td><b>Total Test Cases</b></td>
    <td>25</td>
  </tr>
  <tr>
    <td><b>Passed</b></td>
    <td>25 (100%)</td>
  </tr>
  <tr>
    <td><b>Failed</b></td>
    <td>0 (0%)</td>
  </tr>
  <tr>
    <td><b>Warnings</b></td>
    <td>3 (non-blocking)</td>
  </tr>
  <tr>
    <td><b>Blockers</b></td>
    <td>0</td>
  </tr>
</table>

### 9.2 Key Findings

1. **All data files** are valid with zero null values and correct schemas.
2. **SUMO network** correctly represents a 3×3 bidirectional grid (9 nodes, 24 edges).
3. **LSTM model** produces meaningful predictions — forecasts speed decline from 9.42 → 8.57 with MEDIUM congestion.
4. **Route optimizer** correctly evaluates all 12 possible paths and selects the lowest-cost route, saving 66.9% over the worst alternative.
5. **Dashboard** renders all 12 sections without errors and both interactive buttons function correctly.
6. **Module integration** is clean with no circular dependencies or import failures.

### 9.3 Recommendation

> ✅ **APPROVED FOR USE**
>
> The Smart Traffic AI system has passed all 25 functional test cases with a 100% pass rate. All modules — data processing, LSTM prediction, route optimization, and dashboard visualization — are fully operational. The 3 identified warnings are informational and do not impact system functionality.

---

<p align="center">
  <i>End of Test Report</i><br>
  <i>Smart Traffic AI — Version 1.0.0</i><br>
  <i>September 06, 2026</i>
</p>
