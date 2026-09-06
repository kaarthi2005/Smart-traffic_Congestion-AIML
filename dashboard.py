import streamlit as st
import pandas as pd
from optimization.smart_route import optimize_route

import sys
import subprocess
import networkx as nx
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import re
import os


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Smart Traffic AI",
    page_icon="🚦",
    layout="wide"
)


# ============================================================
# PROJECT PATH
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)


# ============================================================
# TITLE
# ============================================================

st.title(
    "🚦 Smart Traffic Prediction & Emergency Route Optimization"
)

st.markdown(
    """
    **AI-powered traffic monitoring, congestion prediction,
    and emergency route optimization**
    """
)


# ============================================================
# LOAD TRAFFIC DATA
# ============================================================

traffic_path = os.path.join(
    BASE_DIR,
    "data",
    "traffic.csv"
)

try:

    traffic = pd.read_csv(
        traffic_path
    )

except Exception as e:

    st.error(
        f"❌ Could not load traffic data: {e}"
    )

    st.stop()


# ============================================================
# BASIC TRAFFIC INFORMATION
# ============================================================

if "speed" in traffic.columns:

    current_average_speed = (
        traffic["speed"]
        .mean()
    )

else:

    current_average_speed = 0


if "vehicle_id" in traffic.columns:

    vehicle_count = (
        traffic["vehicle_id"]
        .nunique()
    )

else:

    vehicle_count = len(
        traffic
    )


traffic_records = len(
    traffic
)


# ============================================================
# LOAD SUMO NETWORK
# ============================================================

def load_network():

    network_path = os.path.join(
        BASE_DIR,
        "simulation",
        "network.net.xml"
    )

    graph = nx.DiGraph()

    try:

        tree = ET.parse(
            network_path
        )

        root = tree.getroot()

        for edge in root.findall(
            "edge"
        ):

            edge_id = edge.get(
                "id"
            )

            from_node = edge.get(
                "from"
            )

            to_node = edge.get(
                "to"
            )

            # ------------------------------------------------
            # Ignore SUMO internal edges
            # ------------------------------------------------

            if (
                edge_id is None
                or edge_id.startswith(":")
                or from_node is None
                or to_node is None
            ):

                continue


            # ------------------------------------------------
            # Read road shape
            # ------------------------------------------------

            shape = None

            lane = edge.find(
                "lane"
            )

            if lane is not None:

                shape_string = lane.get(
                    "shape"
                )

                if shape_string:

                    shape = []

                    for point in (
                        shape_string.split()
                    ):

                        try:

                            x, y = point.split(
                                ","
                            )

                            shape.append(
                                (
                                    float(x),
                                    float(y)
                                )
                            )

                        except ValueError:

                            continue


            # ------------------------------------------------
            # Read actual SUMO road length
            # ------------------------------------------------

            road_length = 0.0

            if lane is not None:

                length_value = lane.get(
                    "length"
                )

                if length_value:

                    try:

                        road_length = float(
                            length_value
                        )

                    except ValueError:

                        road_length = 0.0


            # ------------------------------------------------
            # Add road to NetworkX
            # ------------------------------------------------

            graph.add_edge(
                from_node,
                to_node,
                edge_id=edge_id,
                shape=shape,
                length=road_length
            )


        return graph


    except Exception as e:

        st.error(
            f"❌ Could not load SUMO network: {e}"
        )

        return nx.DiGraph()


network = load_network()

network_roads = (
    network.number_of_edges()
)


# ============================================================
# TOP METRICS
# ============================================================

st.divider()

col1, col2, col3, col4 = (
    st.columns(4)
)


with col1:

    st.metric(
        "Average Speed",
        f"{current_average_speed:.2f}"
    )


with col2:

    st.metric(
        "Vehicles",
        vehicle_count
    )


with col3:

    st.metric(
        "Traffic Records",
        traffic_records
    )


with col4:

    st.metric(
        "Network Roads",
        network_roads
    )


# ============================================================
# TRAFFIC OVERVIEW
# ============================================================

st.divider()

st.subheader(
    "📊 Traffic Overview"
)


if "speed" in traffic.columns:

    speed_data = (
        traffic["speed"]
        .dropna()
    )

    if len(speed_data) > 0:

        fig, ax = plt.subplots(
            figsize=(12, 5)
        )

        ax.plot(
            speed_data.reset_index(
                drop=True
            )
        )

        ax.set_title(
            "Traffic Speed Trend"
        )

        ax.set_xlabel(
            "Traffic Record"
        )

        ax.set_ylabel(
            "Speed"
        )

        ax.grid(
            True
        )

        st.pyplot(
            fig
        )

        plt.close(
            fig
        )


# ============================================================
# LSTM TRAFFIC PREDICTION
# ============================================================

st.divider()

st.subheader(
    "🤖 AI Traffic Prediction"
)

st.write(
    "The LSTM model predicts future traffic speed "
    "and congestion using historical traffic data."
)


if st.button(
    "🤖 Run AI Traffic Prediction"
):

    try:

        prediction_script = os.path.join(
            BASE_DIR,
            "prediction",
            "predict_traffic.py"
        )

        result = subprocess.run(
            [
                sys.executable,
                prediction_script
            ],
            capture_output=True,
            text=True
        )

        output = result.stdout


        if result.returncode != 0:

            st.error(
                "❌ Prediction script failed."
            )

            st.code(
                result.stderr
            )


        else:

            st.code(
                output
            )


            # ------------------------------------------------
            # Extract predicted speed
            # ------------------------------------------------

            speed_match = re.search(
                r"Predicted.*?speed.*?(\d+(?:\.\d+)?)",
                output,
                re.IGNORECASE
            )


            # ------------------------------------------------
            # Extract congestion
            # ------------------------------------------------

            congestion_match = re.search(
                r"congestion.*?(LOW|MEDIUM|HIGH|SEVERE)",
                output,
                re.IGNORECASE
            )


            col1, col2 = (
                st.columns(2)
            )


            with col1:

                if speed_match:

                    predicted_speed = float(
                        speed_match.group(1)
                    )

                    st.metric(
                        "Predicted Speed",
                        f"{predicted_speed:.2f}"
                    )

                else:

                    st.info(
                        "Predicted speed not detected."
                    )


            with col2:

                if congestion_match:

                    predicted_congestion = (
                        congestion_match
                        .group(1)
                        .upper()
                    )

                    st.metric(
                        "Predicted Congestion",
                        predicted_congestion
                    )

                else:

                    st.info(
                        "Predicted congestion not detected."
                    )


    except Exception as e:

        st.error(
            f"❌ Prediction failed: {e}"
        )


# ============================================================
# ROAD LEVEL TRAFFIC ANALYSIS
# ============================================================

st.divider()

st.subheader(
    "🛣️ Road-Level Traffic Analysis"
)


if "edge" in traffic.columns:

    if "vehicle_id" in traffic.columns:

        vehicles_aggregation = (
            "vehicle_id",
            "nunique"
        )

    else:

        vehicles_aggregation = (
            "edge",
            "count"
        )


    if "speed" in traffic.columns:

        speed_aggregation = (
            "speed",
            "mean"
        )

    else:

        speed_aggregation = (
            "edge",
            "count"
        )


    road_analysis = (
        traffic
        .groupby("edge")
        .agg(
            vehicles=vehicles_aggregation,
            average_speed=speed_aggregation
        )
        .reset_index()
    )


    road_analysis[
        "average_speed"
    ] = (
        road_analysis[
            "average_speed"
        ].round(2)
    )


    road_analysis = (
        road_analysis
        .sort_values(
            "vehicles",
            ascending=False
        )
    )


    st.dataframe(
        road_analysis,
        use_container_width=True
    )


else:

    st.info(
        "Edge information is not available."
    )


# ============================================================
# CONGESTION DISTRIBUTION
# ============================================================

st.divider()

st.subheader(
    "🚦 Congestion Distribution"
)


if "speed" in traffic.columns:

    def classify_congestion(
        speed
    ):

        if speed < 5:

            return "SEVERE"

        elif speed < 10:

            return "HIGH"

        elif speed < 20:

            return "MEDIUM"

        else:

            return "LOW"


    congestion_data = (
        traffic["speed"]
        .dropna()
        .apply(
            classify_congestion
        )
    )


    congestion_counts = (
        congestion_data
        .value_counts()
        .reindex(
            [
                "LOW",
                "MEDIUM",
                "HIGH",
                "SEVERE"
            ],
            fill_value=0
        )
    )


    fig, ax = plt.subplots(
        figsize=(10, 5)
    )


    congestion_counts.plot(
        kind="bar",
        ax=ax
    )


    ax.set_title(
        "Traffic Congestion Distribution"
    )

    ax.set_xlabel(
        "Congestion Level"
    )

    ax.set_ylabel(
        "Number of Records"
    )

    ax.grid(
        axis="y"
    )


    st.pyplot(
        fig
    )

    plt.close(
        fig
    )


# ============================================================
# TRAFFIC LEVEL SUMMARY
# ============================================================

st.divider()

st.subheader(
    "📋 Traffic Level Summary"
)


if "speed" in traffic.columns:

    low_count = (
        traffic["speed"] >= 20
    ).sum()


    medium_count = (
        (
            traffic["speed"] >= 10
        )
        &
        (
            traffic["speed"] < 20
        )
    ).sum()


    high_count = (
        (
            traffic["speed"] >= 5
        )
        &
        (
            traffic["speed"] < 10
        )
    ).sum()


    severe_count = (
        traffic["speed"] < 5
    ).sum()


    col1, col2, col3, col4 = (
        st.columns(4)
    )


    with col1:

        st.metric(
            "🟢 Low",
            int(low_count)
        )


    with col2:

        st.metric(
            "🟡 Medium",
            int(medium_count)
        )


    with col3:

        st.metric(
            "🟠 High",
            int(high_count)
        )


    with col4:

        st.metric(
            "🔴 Severe",
            int(severe_count)
        )


# ============================================================
# OVERALL TRAFFIC STATUS
# ============================================================

st.divider()

st.subheader(
    "🚦 Overall Traffic Status"
)


if current_average_speed >= 20:

    overall_status = "🟢 LOW"

elif current_average_speed >= 10:

    overall_status = "🟡 MEDIUM"

elif current_average_speed >= 5:

    overall_status = "🟠 HIGH"

else:

    overall_status = "🔴 SEVERE"


st.info(
    f"Current overall traffic condition: "
    f"**{overall_status}**"
)


# ============================================================
# EMERGENCY ROUTE OPTIMIZATION
# ============================================================

st.divider()

st.subheader(
    "🚑 Emergency Route Optimization"
)


st.write(
    """
    The AI optimizer combines predicted traffic conditions,
    vehicle density, road conditions, and network distance
    to find the best emergency route.
    """
)


if st.button(
    "🚑 Optimize Emergency Route"
):

    st.write(
        "Running AI route optimization..."
    )


    try:

        # ====================================================
        # RUN OPTIMIZATION
        # ====================================================

        result = optimize_route()


        # ====================================================
        # GET RESULTS
        # ====================================================

        route_nodes = result[
            "route"
        ]

        total_cost = float(
            result["total_cost"]
        )

        predicted_speed = float(
            result["predicted_speed"]
        )

        predicted_congestion = result[
            "predicted_congestion"
        ]

        road_conditions = result[
            "road_conditions"
        ]

        route_comparison = result[
            "route_comparison"
        ]


        # ====================================================
        # ROUTE STRING
        # ====================================================

        route = " → ".join(
            route_nodes
        )


        # ====================================================
        # ROUTE COMPARISON DATAFRAME
        # ====================================================

        comparison_df = pd.DataFrame(
            route_comparison
        )


        if comparison_df.empty:

            st.error(
                "❌ No route comparison data was returned."
            )

            st.stop()


        # ====================================================
        # NORMALIZE COLUMN NAMES
        # ====================================================

        comparison_df.columns = [
            str(column)
            .strip()
            .lower()
            for column
            in comparison_df.columns
        ]


        # ====================================================
        # NORMALIZE ROUTE COLUMN
        # ====================================================

        if "route" not in comparison_df.columns:

            if "path" in comparison_df.columns:

                comparison_df[
                    "route"
                ] = comparison_df[
                    "path"
                ]

            elif "nodes" in comparison_df.columns:

                comparison_df[
                    "route"
                ] = comparison_df[
                    "nodes"
                ]


        # ====================================================
        # NORMALIZE COST COLUMN
        # ====================================================

        if "cost" not in comparison_df.columns:

            if "total_cost" in comparison_df.columns:

                comparison_df[
                    "cost"
                ] = comparison_df[
                    "total_cost"
                ]

            elif "optimization_cost" in comparison_df.columns:

                comparison_df[
                    "cost"
                ] = comparison_df[
                    "optimization_cost"
                ]


        # ====================================================
        # CHECK COST
        # ====================================================

        if "cost" not in comparison_df.columns:

            st.error(
                "❌ Route comparison does not contain "
                "a cost column."
            )

            st.write(
                "Available columns:"
            )

            st.write(
                comparison_df.columns.tolist()
            )

            st.dataframe(
                comparison_df,
                use_container_width=True
            )

            st.stop()


        # ====================================================
        # CHECK ROUTE
        # ====================================================

        if "route" not in comparison_df.columns:

            st.error(
                "❌ Route comparison does not contain "
                "a route column."
            )

            st.write(
                "Available columns:"
            )

            st.write(
                comparison_df.columns.tolist()
            )

            st.dataframe(
                comparison_df,
                use_container_width=True
            )

            st.stop()


        # ====================================================
        # RENAME TO DISPLAY NAMES
        # ====================================================

        comparison_df = comparison_df.rename(
            columns={
                "rank": "Rank",
                "route": "Route",
                "cost": "Cost"
            }
        )


        # ====================================================
        # CONVERT COST
        # ====================================================

        comparison_df[
            "Cost"
        ] = pd.to_numeric(
            comparison_df[
                "Cost"
            ],
            errors="coerce"
        )


        comparison_df = (
            comparison_df
            .dropna(
                subset=["Cost"]
            )
        )


        # ====================================================
        # RANK
        # ====================================================

        if "Rank" in comparison_df.columns:

            comparison_df[
                "Rank"
            ] = pd.to_numeric(
                comparison_df[
                    "Rank"
                ],
                errors="coerce"
            )


            comparison_df = (
                comparison_df
                .sort_values(
                    "Rank"
                )
                .reset_index(
                    drop=True
                )
            )


        else:

            comparison_df.insert(
                0,
                "Rank",
                range(
                    1,
                    len(comparison_df) + 1
                )
            )


        # ====================================================
        # SUCCESS MESSAGE
        # ====================================================

        st.success(
            "✅ Emergency route optimized successfully!"
        )


        # ====================================================
        # RECOMMENDED ROUTE
        # ====================================================

        st.markdown(
            "### 🚑 Recommended Emergency Route"
        )


        st.success(
            route
        )


        # ====================================================
        # MAIN ROUTE METRICS
        # ====================================================

        col1, col2, col3, col4 = (
            st.columns(4)
        )


        with col1:

            st.metric(
                "Route Cost",
                f"{total_cost:.2f}"
            )


        with col2:

            st.metric(
                "Predicted Speed",
                f"{predicted_speed:.2f}"
            )


        with col3:

            st.metric(
                "Predicted Congestion",
                str(
                    predicted_congestion
                ).upper()
            )


        with col4:

            st.metric(
                "Roads",
                len(route_nodes) - 1
            )


        # ====================================================
        # ROUTE COMPARISON
        # ====================================================

        st.divider()

        st.subheader(
            "📊 Route Comparison"
        )


        display_columns = [
            column
            for column in [
                "Rank",
                "Route",
                "Cost"
            ]
            if column
            in comparison_df.columns
        ]


        st.dataframe(
            comparison_df[
                display_columns
            ],
            use_container_width=True
        )


        # ====================================================
        # COST SAVINGS
        # ====================================================

        if len(comparison_df) > 1:

            best_cost = float(
                comparison_df[
                    "Cost"
                ].min()
            )

            worst_cost = float(
                comparison_df[
                    "Cost"
                ].max()
            )


            savings = (
                worst_cost
                - best_cost
            )


            if worst_cost > 0:

                savings_percentage = (
                    savings
                    / worst_cost
                    * 100
                )

            else:

                savings_percentage = 0


            col1, col2 = (
                st.columns(2)
            )


            with col1:

                st.metric(
                    "Cost Saving",
                    f"{savings:.2f}"
                )


            with col2:

                st.metric(
                    "Improvement",
                    f"{savings_percentage:.2f}%"
                )


        # ====================================================
        # ROUTE COST CHART
        # ====================================================

        st.divider()

        st.subheader(
            "📈 Route Cost Comparison"
        )


        chart_df = (
            comparison_df
            .copy()
        )


        chart_df[
            "Route Label"
        ] = (
            chart_df[
                "Rank"
            ]
            .astype(str)
            .apply(
                lambda x:
                f"Route {x}"
            )
        )


        fig, ax = plt.subplots(
            figsize=(12, 5)
        )


        ax.bar(
            chart_df[
                "Route Label"
            ],
            chart_df[
                "Cost"
            ]
        )


        ax.set_title(
            "Emergency Route Cost Comparison"
        )

        ax.set_xlabel(
            "Route"
        )

        ax.set_ylabel(
            "Optimization Cost"
        )

        ax.grid(
            axis="y"
        )


        st.pyplot(
            fig
        )

        plt.close(
            fig
        )


        # ====================================================
        # ROUTE DISTANCE
        # ====================================================

        st.divider()

        st.subheader(
            "📏 Route Distance"
        )


        route_distance = 0.0


        for i in range(
            len(route_nodes) - 1
        ):

            source = route_nodes[i]

            destination = route_nodes[
                i + 1
            ]


            if network.has_edge(
                source,
                destination
            ):

                edge_data = network[
                    source
                ][
                    destination
                ]


                edge_length = edge_data.get(
                    "length",
                    0.0
                )


                try:

                    route_distance += float(
                        edge_length
                    )

                except (
                    TypeError,
                    ValueError
                ):

                    pass


        st.metric(
            "Total Route Distance",
            f"{route_distance:.2f} m"
        )


        st.metric(
            "Route Segments",
            len(route_nodes) - 1
        )


        # ====================================================
        # TRAFFIC BY ROUTE EDGE
        # ====================================================

        st.divider()

        st.subheader(
            "🛣️ Traffic Conditions on Selected Route"
        )


        route_edge_data = []


        for i in range(
            len(route_nodes) - 1
        ):

            source = route_nodes[i]

            destination = route_nodes[
                i + 1
            ]


            if not network.has_edge(
                source,
                destination
            ):

                continue


            edge_data = network[
                source
            ][
                destination
            ]


            edge_id = edge_data.get(
                "edge_id"
            )


            if edge_id is None:

                continue


            if "edge" not in traffic.columns:

                continue


            edge_rows = traffic[
                traffic["edge"]
                == edge_id
            ]


            if len(edge_rows) > 0:

                if "vehicle_id" in edge_rows.columns:

                    vehicles = (
                        edge_rows[
                            "vehicle_id"
                        ].nunique()
                    )

                else:

                    vehicles = len(
                        edge_rows
                    )


                if "speed" in edge_rows.columns:

                    average_speed = (
                        edge_rows[
                            "speed"
                        ].mean()
                    )

                else:

                    average_speed = 0


            else:

                vehicles = 0

                average_speed = 0


            route_edge_data.append(
                {
                    "From": source,
                    "To": destination,
                    "Edge": edge_id,
                    "Vehicles": vehicles,
                    "Average Speed": round(
                        average_speed,
                        2
                    )
                }
            )


        if route_edge_data:

            route_traffic_df = pd.DataFrame(
                route_edge_data
            )


            st.dataframe(
                route_traffic_df,
                use_container_width=True
            )


        else:

            st.info(
                "No traffic data found for "
                "the selected route."
            )


        # ====================================================
        # ROUTE PERFORMANCE
        # ====================================================

        st.divider()

        st.subheader(
            "⚡ Route Performance"
        )


        route_speed_values = [
            row["Average Speed"]
            for row in route_edge_data
            if row["Average Speed"] > 0
        ]


        if route_speed_values:

            average_route_speed = (
                sum(
                    route_speed_values
                )
                /
                len(
                    route_speed_values
                )
            )

        else:

            average_route_speed = (
                predicted_speed
            )


        # ----------------------------------------------------
        # Estimated travel time
        #
        # Speed is assumed to be compatible with the
        # distance units used by the SUMO network.
        # ----------------------------------------------------

        if average_route_speed > 0:

            estimated_time = (
                route_distance
                /
                average_route_speed
            )

        else:

            estimated_time = 0


        col1, col2 = (
            st.columns(2)
        )


        with col1:

            st.metric(
                "Average Route Speed",
                f"{average_route_speed:.2f}"
            )


        with col2:

            st.metric(
                "Estimated Travel Time",
                f"{estimated_time:.2f}"
            )


        # ====================================================
        # ROAD CONDITIONS
        # ====================================================

        st.divider()

        st.subheader(
            "🛣️ Road Conditions"
        )


        if road_conditions:

            road_condition_df = pd.DataFrame(
                road_conditions
            )


            road_condition_df.columns = [
                str(column).strip()
                for column
                in road_condition_df.columns
            ]


            st.dataframe(
                road_condition_df,
                use_container_width=True
            )


        else:

            st.info(
                "No detailed road condition data."
            )


        # ====================================================
        # OVERALL ROUTE CONGESTION
        # ====================================================

        st.divider()

        st.subheader(
            "🚦 Route Congestion"
        )


        congestion_text = str(
            predicted_congestion
        ).upper()


        if congestion_text == "LOW":

            st.success(
                "🟢 Low predicted congestion"
            )


        elif congestion_text == "MEDIUM":

            st.warning(
                "🟡 Medium predicted congestion"
            )


        elif congestion_text == "HIGH":

            st.warning(
                "🟠 High predicted congestion"
            )


        else:

            st.error(
                "🔴 Severe predicted congestion"
            )


        # ====================================================
        # ROUTE EFFICIENCY
        # ====================================================

        st.divider()

        st.subheader(
            "📊 Route Efficiency"
        )


        if len(comparison_df) > 1:

            best_cost = float(
                comparison_df[
                    "Cost"
                ].min()
            )


            selected_cost = float(
                total_cost
            )


            if selected_cost > 0:

                efficiency = (
                    best_cost
                    /
                    selected_cost
                    *
                    100
                )

            else:

                efficiency = 100


            efficiency = min(
                max(
                    efficiency,
                    0
                ),
                100
            )


            st.progress(
                efficiency / 100
            )


            st.write(
                f"Route efficiency: "
                f"**{efficiency:.2f}%**"
            )


        # ====================================================
        # KPI SUMMARY
        # ====================================================

        st.divider()

        st.subheader(
            "📌 Emergency Route KPI Summary"
        )


        kpi1, kpi2, kpi3, kpi4 = (
            st.columns(4)
        )


        with kpi1:

            st.metric(
                "Route Nodes",
                len(route_nodes)
            )


        with kpi2:

            st.metric(
                "Route Segments",
                len(route_nodes) - 1
            )


        with kpi3:

            st.metric(
                "Route Cost",
                f"{total_cost:.2f}"
            )


        with kpi4:

            st.metric(
                "Predicted Speed",
                f"{predicted_speed:.2f}"
            )


        # ====================================================
        # AI RECOMMENDATION
        # ====================================================

        st.divider()

        st.subheader(
            "🧠 AI Recommendation"
        )


        if congestion_text == "LOW":

            recommendation = (
                "The selected emergency route "
                "has relatively low predicted "
                "congestion and is suitable "
                "for emergency movement."
            )


        elif congestion_text == "MEDIUM":

            recommendation = (
                "The selected route has moderate "
                "traffic. The AI optimizer selected "
                "the route with the best combined "
                "traffic and routing cost."
            )


        else:

            recommendation = (
                "Traffic conditions are high. "
                "The AI optimizer has selected "
                "the lowest-cost available route "
                "based on predicted conditions."
            )


        st.info(
            recommendation
        )


        # ====================================================
        # WHY AI SELECTED THIS ROUTE
        # ====================================================

        st.subheader(
            "🔍 Why did the AI select this route?"
        )


        st.markdown(
            f"""
            The optimizer selected:

            **{route}**

            because the route has the lowest
            combined optimization cost among
            the evaluated alternatives.

            The optimization considers:

            - 🚦 Predicted traffic congestion
            - 🚗 Vehicle density
            - ⚡ Predicted traffic speed
            - 🛣️ Road conditions
            - 📏 Network routing cost
            - 🤖 LSTM traffic prediction
            """
        )


        # ====================================================
        # NETWORK MAP
        # ====================================================

        st.divider()

        st.subheader(
            "🗺️ Emergency Route Network Map"
        )


        fig, ax = plt.subplots(
            figsize=(12, 8)
        )


        # ----------------------------------------------------
        # Draw all roads
        # ----------------------------------------------------

        for source, destination, data in (
            network.edges(
                data=True
            )
        ):

            shape = data.get(
                "shape"
            )


            if shape:

                x_values = [
                    point[0]
                    for point
                    in shape
                ]

                y_values = [
                    point[1]
                    for point
                    in shape
                ]


                ax.plot(
                    x_values,
                    y_values,
                    linewidth=1,
                    alpha=0.5
                )


        # ----------------------------------------------------
        # Draw selected route
        # ----------------------------------------------------

        for i in range(
            len(route_nodes) - 1
        ):

            source = route_nodes[i]

            destination = route_nodes[
                i + 1
            ]


            if not network.has_edge(
                source,
                destination
            ):

                continue


            edge_data = network[
                source
            ][
                destination
            ]


            shape = edge_data.get(
                "shape"
            )


            if shape:

                x_values = [
                    point[0]
                    for point
                    in shape
                ]

                y_values = [
                    point[1]
                    for point
                    in shape
                ]


                ax.plot(
                    x_values,
                    y_values,
                    linewidth=4
                )


        # ----------------------------------------------------
        # Find route node positions
        # ----------------------------------------------------

        positions = {}


        for node in route_nodes:

            node_data = None


            for source, destination, data in (
                network.edges(
                    data=True
                )
            ):

                if source == node:

                    shape = data.get(
                        "shape"
                    )


                    if shape:

                        node_data = (
                            shape[0]
                        )

                        break


                if destination == node:

                    shape = data.get(
                        "shape"
                    )


                    if shape:

                        node_data = (
                            shape[-1]
                        )

                        break


            if node_data:

                positions[node] = (
                    node_data[0],
                    node_data[1]
                )


        # ----------------------------------------------------
        # Draw route nodes
        # ----------------------------------------------------

        for node, position in (
            positions.items()
        ):

            ax.scatter(
                position[0],
                position[1],
                s=100,
                zorder=5
            )


            ax.text(
                position[0],
                position[1],
                f" {node}",
                fontsize=10
            )


        ax.set_title(
            "AI Optimized Emergency Route"
        )

        ax.set_xlabel(
            "X"
        )

        ax.set_ylabel(
            "Y"
        )

        ax.grid(
            True
        )


        st.pyplot(
            fig
        )

        plt.close(
            fig
        )


        # ====================================================
        # MAP LEGEND
        # ====================================================

        st.caption(
            "Map shows the SUMO road network "
            "and the AI-selected emergency route."
        )


    except Exception as e:

        st.error(
            f"❌ Emergency route optimization failed: {e}"
        )

        st.exception(
            e
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown(
    """
    ### 🚦 Smart Traffic AI

    AI Traffic Prediction + Emergency Route Optimization

    **Technologies:**  
    Python • Streamlit • TensorFlow • LSTM • NetworkX • SUMO
    """
)

