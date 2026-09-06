import sys
import os
import pandas as pd
import networkx as nx
import xml.etree.ElementTree as ET

sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from prediction.predict_traffic import predict_traffic


def optimize_route():

    # ========================================
    # START
    # ========================================

    print("\n========================================")
    print(" SMART TRAFFIC EMERGENCY ROUTE SYSTEM")
    print("========================================")

    # ========================================
    # LOAD SUMO NETWORK
    # ========================================

    print("\n[1] Loading SUMO network...")

    net_file = os.path.join(
        BASE_DIR,
        "simulation",
        "network.net.xml"
    )

    tree = ET.parse(net_file)
    root = tree.getroot()

    G = nx.DiGraph()

    for edge in root.findall("edge"):

        edge_id = edge.get("id")

        # Ignore internal SUMO edges
        if edge_id.startswith(":"):
            continue

        from_node = edge.get("from")
        to_node = edge.get("to")

        lane = edge.find("lane")

        if lane is not None:
            length = float(lane.get("length"))
            speed = float(lane.get("speed"))
        else:
            length = 1
            speed = 1

        G.add_edge(
            from_node,
            to_node,
            edge_id=edge_id,
            length=length,
            speed=speed
        )

    print(
        f"Network: {G.number_of_nodes()} nodes, "
        f"{G.number_of_edges()} roads"
    )

    # ========================================
    # LOAD TRAFFIC DATA
    # ========================================

    print("\n[2] Loading traffic data...")

    traffic_file = os.path.join(BASE_DIR, "data", "sumo_traffic.csv")

    traffic = pd.read_csv(traffic_file)

    print(f"Traffic records: {len(traffic)}")

    # ========================================
    # CURRENT TRAFFIC CONDITIONS
    # ========================================

    print("\n[3] Applying traffic conditions...")

    # --------------------------------
    # RECENT TRAFFIC WINDOW
    # --------------------------------

    latest_time = traffic["time"].max()

    # Use latest 20% of simulation time
    time_window = latest_time * 0.20

    recent_traffic = traffic[
        traffic["time"] >= latest_time - time_window
    ]

    print(f"Latest traffic time: {latest_time:.2f}")
    print(
        f"Recent traffic window: "
        f"{time_window:.2f} seconds"
    )

    print(
        f"Traffic records in recent window: "
        f"{len(recent_traffic)}"
    )

    # --------------------------------
    # REAL-TIME TRAFFIC BY ROAD
    # --------------------------------

    traffic_by_edge = (
        recent_traffic
        .groupby("road")
        .agg(
            average_speed=("speed", "mean"),
            vehicle_count=("vehicle_id", "nunique")
        )
    )

    for u, v, data in G.edges(data=True):

        edge_id = data["edge_id"]

        if edge_id in traffic_by_edge.index:

            speed = traffic_by_edge.loc[
                edge_id,
                "average_speed"
            ]

            vehicles = traffic_by_edge.loc[
                edge_id,
                "vehicle_count"
            ]

            data["traffic_speed"] = speed
            data["vehicles"] = vehicles

            if speed < 10:
                penalty = 3

            elif speed < 12:
                penalty = 2

            else:
                penalty = 1

            data["penalty"] = penalty

        else:

            data["traffic_speed"] = data["speed"]
            data["vehicles"] = 0
            data["penalty"] = 1

    print("Traffic conditions applied.")

    # ========================================
    # LSTM PREDICTION
    # ========================================

    print("\n[4] Running LSTM traffic prediction...")

    prediction = predict_traffic()

    predicted_speed = prediction["predicted_speed"]
    predicted_congestion = prediction["congestion"]

    print(
        f"Predicted future average speed: "
        f"{predicted_speed:.2f}"
    )

    print(
        f"Predicted congestion: "
        f"{predicted_congestion}"
    )

    # ========================================
    # EDGE-SPECIFIC PREDICTION
    # ========================================

    print(
        "\n[4.1] Calculating edge-specific predictions..."
    )

    edge_prediction_factor = {}

    for edge_id, group in traffic.groupby("road"):

        # Ignore SUMO internal edges
        if edge_id.startswith(":"):
            continue

        speeds = group["speed"]

        # Historical average speed
        historical_speed = speeds.mean()

        # Most recent observed speed
        recent_speed = speeds.iloc[-1]

        # Recent traffic trend
        trend = recent_speed - historical_speed

        # Apply limited trend adjustment
        edge_predicted_speed = (
            predicted_speed + (trend * 0.3)
        )

        # Prevent unrealistic values
        edge_predicted_speed = max(
            0.1,
            edge_predicted_speed
        )

        edge_prediction_factor[
            edge_id
        ] = edge_predicted_speed

    print("Edge-specific predictions calculated.")

    # ========================================
    # EMERGENCY VEHICLE
    # ========================================

    source = "A0"
    destination = "C2"

    print("\n[5] EMERGENCY VEHICLE DETECTED")

    print(f"Starting point: {source}")
    print(f"Destination: {destination}")

    # ========================================
    # APPLY PREDICTION TO ROUTE COST
    # ========================================

    print("\n[6] Applying predicted traffic...")

    for u, v, data in G.edges(data=True):

        current_speed = data["traffic_speed"]

        # --------------------------------
        # EDGE-SPECIFIC PREDICTION
        # --------------------------------

        edge_id = data["edge_id"]

        if edge_id in edge_prediction_factor:

            edge_prediction = (
                edge_prediction_factor[edge_id]
            )

        else:

            edge_prediction = predicted_speed

        # Combine current speed and prediction
        predicted_edge_speed = (
            current_speed * 0.7
            + edge_prediction * 0.3
        )

        data["predicted_speed"] = (
            predicted_edge_speed
        )

        # Avoid division by zero
        if predicted_edge_speed <= 0:
            predicted_edge_speed = 1

        # --------------------------------
        # TRAVEL TIME
        # --------------------------------

        travel_time = (
            data["length"] /
            predicted_edge_speed
        )

        # --------------------------------
        # PREDICTED TRAFFIC PENALTY
        # --------------------------------

        if predicted_edge_speed < 10:

            predicted_penalty = 3

        elif predicted_edge_speed < 12:

            predicted_penalty = 2

        else:

            predicted_penalty = 1

        # --------------------------------
        # VEHICLE DENSITY PENALTY
        # --------------------------------

        vehicles = data["vehicles"]

        vehicle_penalty = (
            1.0 + (vehicles / 100)
        )

        # Limit penalty
        vehicle_penalty = min(
            vehicle_penalty,
            2.0
        )

        # --------------------------------
        # COMBINED TRAFFIC PENALTY
        # --------------------------------

        final_penalty = (
            data["penalty"] * 0.5
            + predicted_penalty * 0.3
            + vehicle_penalty * 0.2
        )

        data["predicted_penalty"] = (
            predicted_penalty
        )

        data["vehicle_penalty"] = (
            vehicle_penalty
        )

        data["final_penalty"] = (
            final_penalty
        )

        # --------------------------------
        # FINAL SMART ROUTE COST
        # --------------------------------

        data["cost"] = (
            travel_time * final_penalty
        )

    # ========================================
    # FIND OPTIMAL ROUTE
    # ========================================

    print("\n[7] Optimizing route...")

    route = nx.shortest_path(
        G,
        source=source,
        target=destination,
        weight="cost"
    )

    # ========================================
    # DISPLAY RESULT
    # ========================================

    print("\n========================================")
    print("     AI OPTIMAL EMERGENCY ROUTE")
    print("========================================")

    print(" → ".join(route))

    print("\nRoad conditions:")

    total_cost = 0

    road_conditions = []

    for i in range(len(route) - 1):

        u = route[i]
        v = route[i + 1]

        data = G[u][v]

        cost = data["cost"]

        total_cost += cost

        road_info = {
            "road": data["edge_id"],
            "current_speed": data["traffic_speed"],
            "predicted_speed": data["predicted_speed"],
            "vehicles": data["vehicles"],
            "current_penalty": data["penalty"],
            "predicted_penalty": data["predicted_penalty"],
            "final_penalty": data["final_penalty"],
            "vehicle_density_penalty": data["vehicle_penalty"],
            "cost": cost
        }

        road_conditions.append(road_info)

        print(f"\nRoad: {data['edge_id']}")
        print(
            f"Current Speed: "
            f"{data['traffic_speed']:.2f}"
        )
        print(
            f"Predicted Speed: "
            f"{data['predicted_speed']:.2f}"
        )
        print(
            f"Vehicles: "
            f"{data['vehicles']}"
        )
        print(
            f"Current Traffic Penalty: "
            f"{data['penalty']}"
        )
        print(
            f"Predicted Traffic Penalty: "
            f"{data['predicted_penalty']}"
        )
        print(
            f"Final Combined Penalty: "
            f"{data['final_penalty']:.2f}"
        )
        print(
            f"Vehicle Density Penalty: "
            f"{data['vehicle_penalty']:.2f}"
        )
        print(
            f"Route Cost: "
            f"{cost:.2f}"
        )

    print("\n========================================")
    print(
        f"TOTAL ROUTE COST: "
        f"{total_cost:.2f}"
    )
    print("========================================")

    print("\n🚑 Emergency vehicle should follow:")

    print(" → ".join(route))

    print(
        "\nAI route optimization completed successfully!"
    )

    # ========================================
    # ROUTE COMPARISON
    # ========================================

    print("\n========================================")
    print("       ROUTE COMPARISON")
    print("========================================")

    all_routes = list(
        nx.all_simple_paths(
            G,
            source=source,
            target=destination
        )
    )

    route_results = []

    for possible_route in all_routes:

        route_cost = 0

        for i in range(len(possible_route) - 1):

            u = possible_route[i]
            v = possible_route[i + 1]

            route_cost += G[u][v]["cost"]

        route_results.append(
            (possible_route, route_cost)
        )

    route_results.sort(
        key=lambda x: x[1]
    )

    route_comparison = []

    for index, (
        possible_route,
        cost
    ) in enumerate(
        route_results,
        start=1
    ):

        route_text = " → ".join(
            possible_route
        )

        print(
            f"{index}. "
            f"{route_text} "
            f"| Cost: {cost:.2f}"
        )

        route_comparison.append({
            "rank": index,
            "route": route_text,
            "cost": cost
        })

    # ========================================
    # RETURN DATA FOR DASHBOARD
    # ========================================

    return {
        "route": route,
        "total_cost": total_cost,
        "predicted_speed": predicted_speed,
        "predicted_congestion": predicted_congestion,
        "road_conditions": road_conditions,
        "route_comparison": route_comparison
    }


# ========================================
# RUN DIRECTLY FROM TERMINAL
# ========================================

if __name__ == "__main__":
    optimize_route()