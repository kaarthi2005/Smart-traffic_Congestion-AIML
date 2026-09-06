# Smart Traffic AI — Traffic Prediction & Emergency Route Optimization

## Project status

The project combines **SUMO + TraCI + Python + NetworkX + LSTM + Streamlit** to predict traffic congestion and continuously optimize an emergency vehicle route.

### Completed pipeline
1. SUMO microscopic traffic simulation
2. Traffic data extraction
3. Traffic feature generation
4. LSTM traffic prediction
5. AI traffic-aware route optimization
6. Streamlit dashboard
7. Dynamic real-time rerouting
8. Real emergency vehicle scenario
9. Static vs traffic-aware route comparison and explanation
10. Emergency route visualization
11. Evaluation harness for travel time/speed/reroutes
12. Reproducible demo documentation

## Repository structure

- `simulation/` — SUMO network, routes, configs, and runners
- `data/` — traffic observations and engineered features
- `prediction/` — LSTM training, prediction, model, and scaler
- `optimization/` — static, traffic-aware, emergency, dynamic, and comparison logic
- `evaluation/` — baseline vs dynamic emergency evaluation
- `dashboard/` — Streamlit UI and route visualization

## Quick start (Windows)

Install SUMO and make sure `sumo` / `sumo-gui` are available on PATH. Then install Python dependencies used by the project.

### Normal simulation

```bash
python simulation\run_simulation.py
```

### Dynamic routing

```bash
python optimization\dynamic_route.py --gui --interval 5 --steps 300
```

### Emergency vehicle demo

```bash
python simulation\run_emergency_simulation.py --gui --interval 5 --steps 300
```

The emergency vehicle is `emergency_0` and its destination is node `C2`.

### Route comparison

```bash
python optimization\route_comparison.py --source A0 --destination C2
```

### Evaluation

```bash
python evaluation\evaluate_routes.py --interval 5
```

This creates `data/emergency_evaluation.csv` when SUMO/TraCI is available. The evaluation compares a static emergency route with a live traffic-rerouted emergency route using travel time, average speed, distance, and reroute count.

### Route map

```bash
streamlit run dashboard\emergency_map.py
```

## Demo story

**Normal traffic → congestion develops → emergency vehicle starts → live road speeds/vehicle counts are read through TraCI → dynamic route cost changes → emergency vehicle is rerouted → travel time is measured against the static baseline.**

## AI explanation

The route comparison module explains a decision using live/historical road speed and vehicle count. The dynamic engine uses travel time plus a density penalty, while the LSTM stage supplies predicted traffic information for the broader AI route optimizer.

## Important limitation

The repository's SUMO network is a synthetic 3×3 grid generated with `netgenerate`; it is not yet a georeferenced real-world city map. The visualization therefore shows the actual SUMO network geometry rather than latitude/longitude street tiles.
