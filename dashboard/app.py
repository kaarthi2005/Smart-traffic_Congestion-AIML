import os
import sys
import subprocess
from datetime import datetime

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PROJECT PATH
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)


# ============================================================
# PROJECT IMPORTS
# ============================================================

from optimization.smart_route import optimize_route
from prediction.predict_traffic import predict_traffic


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Smart Traffic AI",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* Main page */
    .main {
        padding-top: 1rem;
    }

    /* Headers */
    h1 {
        font-size: 2.6rem !important;
        font-weight: 800 !important;
    }

    h2 {
        font-weight: 750 !important;
    }

    h3 {
        font-weight: 700 !important;
    }

    /* Metric cards */
    [data-testid="stMetric"] {
        background: rgba(255,255,255,0.035);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 16px;
    }

    /* Buttons */
    .stButton > button {
        border-radius: 10px;
        font-weight: 650;
        min-height: 45px;
    }

    /* Success route box */
    .route-box {
        padding: 18px 22px;
        border-radius: 12px;
        background: rgba(0, 180, 100, 0.12);
        border: 1px solid rgba(0, 220, 120, 0.30);
        margin: 12px 0;
    }

    .route-text {
        font-size: 1.35rem;
        font-weight: 750;
        letter-spacing: 0.3px;
    }

    /* Info boxes */
    .info-box {
        padding: 15px 18px;
        border-radius: 10px;
        background: rgba(30, 120, 220, 0.10);
        border: 1px solid rgba(80, 150, 240, 0.25);
    }

    /* Pipeline cards */
    .pipeline-card {
        padding: 18px;
        border-radius: 12px;
        background: rgba(255,255,255,0.035);
        border: 1px solid rgba(255,255,255,0.08);
        min-height: 150px;
    }

    .pipeline-number {
        font-size: 1.6rem;
        font-weight: 800;
    }

    .pipeline-title {
        font-weight: 700;
        font-size: 1.05rem;
        margin-top: 5px;
    }

    .pipeline-description {
        color: #aaa;
        font-size: 0.88rem;
        margin-top: 5px;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #777;
        padding: 20px 0;
        font-size: 0.85rem;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# FILE PATHS
# ============================================================

SUMO_TRAFFIC_FILE = os.path.join(
    BASE_DIR,
    "data",
    "sumo_traffic.csv"
)

TRAFFIC_FEATURES_FILE = os.path.join(
    BASE_DIR,
    "data",
    "sumo_traffic_features.csv"
)

LSTM_MODEL_FILE = os.path.join(
    BASE_DIR,
    "prediction",
    "traffic_lstm_sumo.keras"
)

LSTM_SCALER_FILE = os.path.join(
    BASE_DIR,
    "prediction",
    "scaler_sumo.pkl"
)


# ============================================================
# NETWORK VISUALIZATION
# ============================================================

def draw_emergency_route(route):

    import networkx as nx

    G = nx.DiGraph()

    # Network connections
    edges = [
        ("A0", "A1"),
        ("A1", "A0"),

        ("A1", "A2"),
        ("A2", "A1"),

        ("A0", "B0"),
        ("B0", "A0"),

        ("B0", "B1"),
        ("B1", "B0"),

        ("B1", "B2"),
        ("B2", "B1"),

        ("A2", "B2"),
        ("B2", "A2"),

        ("B0", "C0"),
        ("C0", "B0"),

        ("C0", "C1"),
        ("C1", "C0"),

        ("C1", "C2"),
        ("C2", "C1"),

        ("B1", "C1"),
        ("C1", "B1"),

        ("B2", "C2"),
        ("C2", "B2"),

        ("A1", "B1"),
        ("B1", "A1"),
    ]

    G.add_edges_from(edges)

    pos = {
        "A0": (0, 2),
        "A1": (1, 2),
        "A2": (2, 2),

        "B0": (0, 1),
        "B1": (1, 1),
        "B2": (2, 1),

        "C0": (0, 0),
        "C1": (1, 0),
        "C2": (2, 0),
    }

    fig, ax = plt.subplots(
        figsize=(12, 7)
    )

    # --------------------------------------------------------
    # Complete network
    # --------------------------------------------------------

    nx.draw_networkx_edges(
        G,
        pos,
        ax=ax,
        arrows=True,
        arrowsize=15,
        alpha=0.25,
        width=1.5,
        connectionstyle="arc3,rad=0.02"
    )

    nx.draw_networkx_nodes(
        G,
        pos,
        ax=ax,
        node_size=1200
    )

    nx.draw_networkx_labels(
        G,
        pos,
        ax=ax,
        font_weight="bold",
        font_size=12
    )

    # --------------------------------------------------------
    # Emergency route
    # --------------------------------------------------------

    if route and len(route) > 1:

        route_edges = [
            (route[i], route[i + 1])
            for i in range(len(route) - 1)
        ]

        nx.draw_networkx_edges(
            G,
            pos,
            ax=ax,
            edgelist=route_edges,
            width=5,
            arrows=True,
            arrowsize=22,
            edge_color="red",
            connectionstyle="arc3,rad=0.02"
        )

        nx.draw_networkx_nodes(
            G,
            pos,
            ax=ax,
            nodelist=route,
            node_size=1350,
            node_color="orange"
        )

        # Starting point
        nx.draw_networkx_nodes(
            G,
            pos,
            ax=ax,
            nodelist=[route[0]],
            node_size=1500,
            node_color="green"
        )

        # Destination
        nx.draw_networkx_nodes(
            G,
            pos,
            ax=ax,
            nodelist=[route[-1]],
            node_size=1500,
            node_color="red"
        )

        nx.draw_networkx_labels(
            G,
            pos,
            ax=ax,
            font_weight="bold",
            font_size=12
        )

    ax.set_title(
        "AI Emergency Route on Traffic Network",
        fontsize=18,
        fontweight="bold"
    )

    ax.axis("off")

    plt.tight_layout()

    return fig


# ============================================================
# RUN SUMO
# ============================================================

def run_sumo_simulation():

    result = subprocess.run(
        [
            sys.executable,
            os.path.join(
                BASE_DIR,
                "simulation",
                "run_simulation.py"
            )
        ],
        capture_output=True,
        text=True,
        cwd=BASE_DIR
    )

    if result.returncode == 0:
        return True, result.stdout

    error = result.stderr

    if result.stdout:
        error += "\n\n" + result.stdout

    return False, error


# ============================================================
# PROCESS SUMO DATA
# ============================================================

def process_sumo_data():

    result = subprocess.run(
        [
            sys.executable,
            os.path.join(
                BASE_DIR,
                "simulation",
                "process_sumo_data.py"
            )
        ],
        capture_output=True,
        text=True,
        cwd=BASE_DIR
    )

    if result.returncode == 0:
        return True, result.stdout

    error = result.stderr

    if result.stdout:
        error += "\n\n" + result.stdout

    return False, error


# ============================================================
# TRAIN LSTM MODEL
# ============================================================

def train_lstm_model():

    result = subprocess.run(
        [
            sys.executable,
            os.path.join(
                BASE_DIR,
                "prediction",
                "train_lstm.py"
            )
        ],
        capture_output=True,
        text=True,
        cwd=BASE_DIR
    )

    if result.returncode == 0:

        return True, result.stdout

    error = result.stderr

    if result.stdout:

        error += "\n\n" + result.stdout

    return False, error


# ============================================================
# COMPLETE AI PIPELINE
# ============================================================

def run_complete_pipeline():

    # ========================================================
    # STEP 1 - SUMO
    # ========================================================

    st.write(
        "### Step 1/4 — Running SUMO Simulation"
    )

    success, output = run_sumo_simulation()

    if not success:

        st.error(
            "SUMO simulation failed."
        )

        st.code(output)

        return False

    st.success(
        "SUMO simulation completed successfully."
    )

    with st.expander(
        "📄 View SUMO Output"
    ):

        st.code(output)


    # ========================================================
    # STEP 2 - PROCESS DATA
    # ========================================================

    st.write(
        "### Step 2/4 — Processing Traffic Data"
    )

    processed, process_output = process_sumo_data()

    if not processed:

        st.error(
            "SUMO traffic data processing failed."
        )

        st.code(process_output)

        return False

    st.success(
        "Traffic data processed successfully."
    )

    with st.expander(
        "📊 View Traffic Processing Output"
    ):

        st.code(process_output)


    # ========================================================
    # STEP 3 - TRAIN LSTM
    # ========================================================

    st.write(
        "### Step 3/4 — Training LSTM Prediction Model"
    )

    trained, train_output = train_lstm_model()

    if not trained:

        st.error(
            "LSTM model training failed."
        )

        st.code(train_output)

        return False

    st.success(
        "LSTM model trained successfully."
    )

    with st.expander(
        "🧠 View LSTM Training Output"
    ):

        st.code(train_output)


    # ========================================================
    # STEP 4 - EMERGENCY ROUTE
    # ========================================================

    st.write(
        "### Step 4/4 — Optimizing Emergency Route"
    )

    try:

        route_result = optimize_route()

        if not route_result:

            st.error(
                "Emergency route optimization returned no result."
            )

            return False

        st.session_state[
            "route_result"
        ] = route_result

        st.success(
            "Emergency route optimized successfully."
        )

    except Exception as e:

        st.error(
            "Emergency route optimization failed."
        )

        st.exception(e)

        return False

    return True


# ============================================================
# TITLE
# ============================================================

st.title(
    "🚦 Smart Traffic AI"
)

st.markdown(
    "**AI-Based Traffic Prediction & Emergency Route Optimization**"
)

st.caption(
    "SUMO Simulation  •  LSTM Prediction  •  "
    "Traffic Analysis  •  AI Emergency Routing"
)

st.divider()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header(
        "⚙️ System Control"
    )

    st.markdown(
        "### Emergency Scenario"
    )

    source = "A0"
    destination = "C2"

    st.info(
        f"🚑 Emergency vehicle\n\n"
        f"**{source} → {destination}**"
    )

    st.divider()

    st.markdown(
        "### System Files"
    )

    sumo_status = os.path.exists(
        SUMO_TRAFFIC_FILE
    )

    features_status = os.path.exists(
        TRAFFIC_FEATURES_FILE
    )

    model_status = os.path.exists(
        LSTM_MODEL_FILE
    )

    scaler_status = os.path.exists(
        LSTM_SCALER_FILE
    )

    st.write(
        f"{'🟢' if sumo_status else '🔴'} Raw SUMO data"
    )

    st.write(
        f"{'🟢' if features_status else '🔴'} Traffic features"
    )

    st.write(
        f"{'🟢' if model_status else '🔴'} LSTM model"
    )

    st.write(
        f"{'🟢' if scaler_status else '🔴'} LSTM scaler"
    )

    st.divider()

    st.caption(
        "Smart Traffic AI"
    )

    st.caption(
        "SUMO + TraCI + LSTM + NetworkX"
    )


# ============================================================
# SUMO SIMULATION SECTION
# ============================================================

st.header(
    "🚦 SUMO Traffic Simulation"
)

st.write(
    "Generate fresh traffic data from the SUMO microscopic "
    "traffic simulation."
)

col1, col2 = st.columns(
    [3, 1]
)

with col1:

    if st.button(
        "▶ Run SUMO Simulation",
        use_container_width=True
    ):

        with st.spinner(
            "Running SUMO simulation..."
        ):

            success, output = run_sumo_simulation()

        if success:

            st.success(
                "SUMO simulation completed successfully."
            )

            with st.expander(
                "📄 View SUMO Output"
            ):

                st.code(output)

            # Process automatically
            with st.spinner(
                "Processing SUMO traffic data..."
            ):

                processed, process_output = (
                    process_sumo_data()
                )

            if processed:

                st.success(
                    "SUMO traffic data processed successfully."
                )

                with st.expander(
                    "📊 View Processing Output"
                ):

                    st.code(process_output)

                st.rerun()

            else:

                st.error(
                    "SUMO traffic data processing failed."
                )

                st.code(process_output)

        else:

            st.error(
                "SUMO simulation failed."
            )

            st.code(output)


with col2:

    if os.path.exists(
        SUMO_TRAFFIC_FILE
    ):

        modified_time = os.path.getmtime(
            SUMO_TRAFFIC_FILE
        )

        readable_time = datetime.fromtimestamp(
            modified_time
        ).strftime(
            "%H:%M:%S"
        )

        st.metric(
            "Last Data Update",
            readable_time
        )


# ============================================================
# CHECK DATA
# ============================================================

if not os.path.exists(
    TRAFFIC_FEATURES_FILE
):

    st.warning(
        "Traffic data is not available yet."
    )

    st.info(
        "Run the SUMO simulation or use the AI pipeline below."
    )


# ============================================================
# LOAD DATA
# ============================================================

if not os.path.exists(
    TRAFFIC_FEATURES_FILE
):

    df = pd.DataFrame(
        columns=[
            "time",
            "average_speed",
            "vehicle_count",
            "congestion"
        ]
    )

else:

    try:

        df = pd.read_csv(
            TRAFFIC_FEATURES_FILE
        )

    except Exception as e:

        st.error(
            f"Error loading traffic data: {e}"
        )

        st.stop()


# ============================================================
# VALIDATE DATA
# ============================================================

required_columns = [
    "time",
    "average_speed",
    "vehicle_count",
    "congestion"
]

if not df.empty:

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:

        st.error(
            "Missing columns: "
            + ", ".join(missing_columns)
        )

        st.stop()


# ============================================================
# CURRENT TRAFFIC
# ============================================================

if not df.empty:

    latest = df.iloc[-1]

    current_speed = float(
        latest["average_speed"]
    )

    vehicle_count = int(
        latest["vehicle_count"]
    )

    current_congestion = str(
        latest["congestion"]
    )

    simulation_time = float(
        latest["time"]
    )

else:

    current_speed = 0
    vehicle_count = 0
    current_congestion = "N/A"
    simulation_time = 0


# ============================================================
# CURRENT TRAFFIC
# ============================================================

st.divider()

st.header(
    "📊 Current Traffic"
)

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "🚗 Vehicles",
        vehicle_count
    )

with col2:

    st.metric(
        "⚡ Average Speed",
        f"{current_speed:.2f} m/s"
    )

with col3:

    st.metric(
        "🚦 Congestion",
        current_congestion
    )

with col4:

    st.metric(
        "⏱ Simulation Time",
        f"{simulation_time:.0f} s"
    )


# ============================================================
# TRAFFIC OVERVIEW
# ============================================================

if not df.empty:

    st.divider()

    st.header(
        "📈 Traffic Overview"
    )

    tab1, tab2 = st.tabs(
        [
            "🚗 Speed",
            "🚙 Vehicle Count"
        ]
    )

    with tab1:

        fig, ax = plt.subplots(
            figsize=(12, 4)
        )

        ax.plot(
            df["time"],
            df["average_speed"],
            linewidth=2
        )

        ax.set_xlabel(
            "Time (seconds)"
        )

        ax.set_ylabel(
            "Average Speed (m/s)"
        )

        ax.set_title(
            "SUMO Average Traffic Speed"
        )

        ax.grid(
            True,
            alpha=0.25
        )

        st.pyplot(
            fig,
            use_container_width=True
        )

        plt.close(fig)

    with tab2:

        fig, ax = plt.subplots(
            figsize=(12, 4)
        )

        ax.plot(
            df["time"],
            df["vehicle_count"],
            linewidth=2
        )

        ax.set_xlabel(
            "Time (seconds)"
        )

        ax.set_ylabel(
            "Vehicles"
        )

        ax.set_title(
            "SUMO Vehicle Count"
        )

        ax.grid(
            True,
            alpha=0.25
        )

        st.pyplot(
            fig,
            use_container_width=True
        )

        plt.close(fig)


# ============================================================
# AI TRAFFIC PREDICTION
# ============================================================

st.divider()

st.header(
    "🧠 AI Traffic Prediction"
)

if not model_status:

    st.warning(
        "LSTM model not found."
    )

    st.info(
        "Run the AI Traffic Pipeline to train the model."
    )

    prediction_available = False

else:

    try:

        with st.spinner(
            "Running LSTM prediction..."
        ):

            prediction = predict_traffic()

        predicted_speed = float(
            prediction["predicted_speed"]
        )

        predicted_congestion = str(
            prediction["congestion"]
        )

        prediction_current_speed = float(
            prediction["current_speed"]
        )

        prediction_available = True

    except Exception as e:

        st.error(
            f"LSTM prediction failed: {e}"
        )

        prediction_available = False


if prediction_available:

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Current Speed",
            f"{prediction_current_speed:.2f} m/s"
        )

    with col2:

        speed_difference = (
            predicted_speed
            - prediction_current_speed
        )

        st.metric(
            "Predicted Speed",
            f"{predicted_speed:.2f} m/s",
            f"{speed_difference:+.2f} m/s"
        )

    with col3:

        st.metric(
            "Predicted Congestion",
            predicted_congestion
        )

    if predicted_speed < prediction_current_speed:

        st.warning(
            "⚠️ LSTM predicts decreasing traffic speed."
        )

    elif predicted_speed > prediction_current_speed:

        st.success(
            "🟢 LSTM predicts improving traffic speed."
        )

    else:

        st.info(
            "Traffic speed is predicted to remain stable."
        )


# ============================================================
# AI PIPELINE
# ============================================================

st.divider()

st.header(
    "⚙️ Complete AI Traffic & Emergency Pipeline"
)

st.write(
    "Run the complete AI workflow automatically: "
    "**SUMO → Traffic Processing → LSTM Training → "
    "Emergency Route Optimization**."
)


# ============================================================
# PIPELINE VISUALIZATION
# ============================================================

p1, p2, p3, p4 = st.columns(4)


with p1:

    st.markdown(
        """
        <div class="pipeline-card">

        <div class="pipeline-number">01</div>

        <div class="pipeline-title">
        🚦 SUMO Simulation
        </div>

        <div class="pipeline-description">
        Generate fresh traffic records
        using microscopic traffic simulation.
        </div>

        </div>
        """,
        unsafe_allow_html=True
    )


with p2:

    st.markdown(
        """
        <div class="pipeline-card">

        <div class="pipeline-number">02</div>

        <div class="pipeline-title">
        📊 Traffic Processing
        </div>

        <div class="pipeline-description">
        Convert vehicle-level data into
        speed, vehicle count and congestion.
        </div>

        </div>
        """,
        unsafe_allow_html=True
    )


with p3:

    st.markdown(
        """
        <div class="pipeline-card">

        <div class="pipeline-number">03</div>

        <div class="pipeline-title">
        🧠 LSTM Prediction
        </div>

        <div class="pipeline-description">
        Retrain the AI model using the
        latest SUMO traffic data.
        </div>

        </div>
        """,
        unsafe_allow_html=True
    )


with p4:

    st.markdown(
        """
        <div class="pipeline-card">

        <div class="pipeline-number">04</div>

        <div class="pipeline-title">
        🚑 Emergency Routing
        </div>

        <div class="pipeline-description">
        Select the safest and fastest
        emergency route using AI.
        </div>

        </div>
        """,
        unsafe_allow_html=True
    )


st.write("")


# ============================================================
# ONE-CLICK FULL AI SYSTEM
# ============================================================

if st.button(
    "🚨 RUN COMPLETE AI SYSTEM",
    use_container_width=True,
    type="primary"
):

    with st.spinner(
        "AI system is running: SUMO → Processing → LSTM → Routing..."
    ):

        success = run_complete_pipeline()

    if success:

        st.success(
            "🎉 Complete AI system finished successfully!"
        )

        st.rerun()


# ============================================================
# EMERGENCY ROUTE OPTIMIZATION
# ============================================================

st.divider()

st.header(
    "🚑 Emergency Route Optimization"
)

st.write(
    f"Emergency vehicle route: "
    f"**{source} → {destination}**"
)

if st.button(
    "🚑 Optimize Emergency Route",
    use_container_width=True,
    type="primary"
):

    with st.spinner(
        "AI is calculating the safest and fastest route..."
    ):

        try:

            result = optimize_route()

            st.session_state[
                "route_result"
            ] = result

            st.success(
                "Emergency route optimized successfully."
            )

        except Exception as e:

            st.error(
                f"Route optimization failed: {e}"
            )

            st.session_state.pop(
                "route_result",
                None
            )


# ============================================================
# DISPLAY ROUTE RESULT
# ============================================================

if "route_result" in st.session_state:

    result = st.session_state[
        "route_result"
    ]

    route = result.get(
        "route",
        []
    )

    total_cost = float(
        result.get(
            "total_cost",
            0
        )
    )

    route_predicted_speed = float(
        result.get(
            "predicted_speed",
            predicted_speed if prediction_available else 0
        )
    )

    route_predicted_congestion = str(
        result.get(
            "predicted_congestion",
            predicted_congestion if prediction_available else "N/A"
        )
    )

    road_conditions = result.get(
        "road_conditions",
        []
    )

    route_comparison = result.get(
        "route_comparison",
        []
    )


    # ========================================================
    # ROUTE SUMMARY
    # ========================================================

    st.divider()

    st.header(
        "🏆 Optimized Route Result"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Starting Point",
            source
        )

    with col2:

        st.metric(
            "Destination",
            destination
        )

    with col3:

        st.metric(
            "Total Route Cost",
            f"{total_cost:.2f}"
        )


    if route:

        route_string = " → ".join(
            route
        )

        st.markdown(
            f"""
            <div class="route-box">

            <div>
            🚑 AI OPTIMAL EMERGENCY ROUTE
            </div>

            <div class="route-text">
            {route_string}
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        st.warning(
            "No emergency route was found."
        )


    # ========================================================
    # AI PREDICTION USED
    # ========================================================

    st.subheader(
        "🧠 AI Prediction Used for Routing"
    )

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "LSTM Predicted Speed",
            f"{route_predicted_speed:.2f} m/s"
        )

    with col2:

        st.metric(
            "Predicted Congestion",
            route_predicted_congestion
        )

    st.markdown(
        """
        <div class="info-box">

        The route optimizer combines:

        **Current traffic + LSTM prediction + vehicle density
        + traffic penalties + road cost**

        to select the emergency route.

        </div>
        """,
        unsafe_allow_html=True
    )


    # ========================================================
    # ROAD CONDITIONS
    # ========================================================

    st.subheader(
        "🛣️ Road Conditions Along Optimal Route"
    )

    if road_conditions:

        for condition in road_conditions:

            road = condition.get(
                "road",
                "Unknown"
            )

            current_road_speed = float(
                condition.get(
                    "current_speed",
                    0
                )
            )

            predicted_road_speed = float(
                condition.get(
                    "predicted_speed",
                    0
                )
            )

            vehicles = int(
                condition.get(
                    "vehicles",
                    0
                )
            )

            final_penalty = float(
                condition.get(
                    "final_penalty",
                    0
                )
            )

            cost = float(
                condition.get(
                    "cost",
                    0
                )
            )

            with st.expander(
                f"🛣️ {road}"
            ):

                col1, col2, col3, col4 = st.columns(4)

                with col1:

                    st.metric(
                        "Current Speed",
                        f"{current_road_speed:.2f} m/s"
                    )

                with col2:

                    st.metric(
                        "Predicted Speed",
                        f"{predicted_road_speed:.2f} m/s"
                    )

                with col3:

                    st.metric(
                        "Vehicles",
                        vehicles
                    )

                with col4:

                    st.metric(
                        "Route Cost",
                        f"{cost:.2f}"
                    )

                st.progress(
                    min(
                        max(
                            final_penalty / 3,
                            0
                        ),
                        1
                    )
                )

                st.caption(
                    f"Traffic penalty: {final_penalty:.2f}"
                )

    else:

        st.info(
            "No road-condition data available."
        )


    # ========================================================
    # ROUTE COMPARISON
    # ========================================================

    st.subheader(
        "📊 Route Comparison"
    )

    if route_comparison:

        comparison_data = []

        for index, item in enumerate(
            route_comparison,
            start=1
        ):

            if isinstance(
                item,
                dict
            ):

                route_value = item.get(
                    "route",
                    []
                )

                if isinstance(
                    route_value,
                    list
                ):

                    route_value = " → ".join(
                        route_value
                    )

                comparison_data.append(
                    {
                        "Rank": index,
                        "Route": route_value,
                        "Cost": float(
                            item.get(
                                "cost",
                                0
                            )
                        )
                    }
                )

            else:

                try:

                    comparison_data.append(
                        {
                            "Rank": index,
                            "Route": " → ".join(
                                item[0]
                            ),
                            "Cost": float(
                                item[1]
                            )
                        }
                    )

                except Exception:

                    continue


        if comparison_data:

            df_comparison = pd.DataFrame(
                comparison_data
            )

            st.dataframe(
                df_comparison,
                use_container_width=True,
                hide_index=True
            )


            # ------------------------------------------------
            # COST ADVANTAGE
            # ------------------------------------------------

            if len(
                df_comparison
            ) >= 2:

                best_cost = float(
                    df_comparison.iloc[0]["Cost"]
                )

                second_cost = float(
                    df_comparison.iloc[1]["Cost"]
                )

                if second_cost > 0:

                    savings = (
                        (
                            second_cost
                            - best_cost
                        )
                        / second_cost
                    ) * 100

                    st.metric(
                        "🏆 AI Cost Advantage vs 2nd Best",
                        f"{savings:.2f}%"
                    )


            # ------------------------------------------------
            # ROUTE COST CHART
            # ------------------------------------------------

            if len(
                df_comparison
            ) >= 2:

                chart_df = (
                    df_comparison
                    .head(10)
                    .copy()
                )

                chart_df["Route Label"] = (
                    "Route "
                    + chart_df["Rank"].astype(str)
                )

                st.bar_chart(
                    chart_df.set_index(
                        "Route Label"
                    )["Cost"]
                )

        else:

            st.info(
                "No route comparison data available."
            )

    else:

        st.info(
            "No route comparison data available."
        )


    # ========================================================
    # EMERGENCY ROUTE MAP
    # ========================================================

    st.divider()

    st.header(
        "🗺️ Emergency Route Map"
    )

    if route:

        fig = draw_emergency_route(
            route
        )

        st.pyplot(
            fig,
            use_container_width=True
        )

        plt.close(fig)

        st.caption(
            "Green = starting point  •  "
            "Red = destination  •  "
            "Orange = emergency route"
        )

    else:

        st.info(
            "Optimize an emergency route to display the map."
        )


# ============================================================
# TRAFFIC VISUALIZATION
# ============================================================

if not df.empty:

    st.divider()

    st.header(
        "📊 Detailed Traffic Visualization"
    )

    traffic_df = df.copy()

    tab1, tab2, tab3 = st.tabs(
        [
            "🚗 Average Speed",
            "🚙 Vehicle Count",
            "🚦 Congestion"
        ]
    )

    with tab1:

        st.line_chart(
            traffic_df.set_index(
                "time"
            )[
                "average_speed"
            ]
        )

    with tab2:

        st.line_chart(
            traffic_df.set_index(
                "time"
            )[
                "vehicle_count"
            ]
        )

    with tab3:

        congestion_map = {
            "LOW": 1,
            "MEDIUM": 2,
            "HIGH": 3
        }

        congestion_df = traffic_df.copy()

        congestion_df[
            "congestion_level"
        ] = congestion_df[
            "congestion"
        ].map(
            congestion_map
        )

        st.line_chart(
            congestion_df.set_index(
                "time"
            )[
                "congestion_level"
            ]
        )

        st.caption(
            "Congestion scale: 1 = LOW, 2 = MEDIUM, 3 = HIGH"
        )


# ============================================================
# SIMULATION INFORMATION
# ============================================================

st.divider()

st.header(
    "📁 Simulation Information"
)

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "Traffic Time Steps",
        len(df)
    )

with col2:

    if os.path.exists(
        SUMO_TRAFFIC_FILE
    ):

        try:

            sumo_df = pd.read_csv(
                SUMO_TRAFFIC_FILE
            )

            st.metric(
                "Raw Vehicle Records",
                len(sumo_df)
            )

        except Exception:

            st.metric(
                "Raw Vehicle Records",
                0
            )

    else:

        st.metric(
            "Raw Vehicle Records",
            0
        )


with col3:

    st.metric(
        "SUMO Network",
        "9 Nodes / 24 Roads"
    )


with col4:

    st.metric(
        "LSTM Model",
        "Ready" if model_status else "Not Ready"
    )


# ============================================================
# FINAL STATUS
# ============================================================

st.divider()

if (
    sumo_status
    and features_status
    and model_status
    and scaler_status
):

    st.success(
        "🟢 AI Traffic System Ready"
    )

else:

    st.warning(
        "🟡 AI Traffic System requires initialization."
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">

    Smart Traffic AI<br>

    <b>SUMO + TraCI + LSTM + NetworkX</b><br>

    Traffic Prediction & Emergency Route Optimization

    </div>
    """,
    unsafe_allow_html=True
)