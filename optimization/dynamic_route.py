import argparse
import os
import sys
import time
import xml.etree.ElementTree as ET

import networkx as nx
import traci


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NETWORK_FILE = os.path.join(BASE_DIR, "simulation", "network.net.xml")
SUMO_CONFIG = os.path.join(BASE_DIR, "simulation", "simulation.sumocfg")


def load_network():
    """Build a directed graph from the SUMO network."""
    graph = nx.DiGraph()
    root = ET.parse(NETWORK_FILE).getroot()

    for edge in root.findall("edge"):
        edge_id = edge.get("id")
        from_node = edge.get("from")
        to_node = edge.get("to")
        lane = edge.find("lane")

        if not edge_id or edge_id.startswith(":") or not from_node or not to_node:
            continue
        if lane is None:
            continue

        graph.add_edge(
            from_node,
            to_node,
            edge_id=edge_id,
            length=float(lane.get("length", 1.0)),
            max_speed=float(lane.get("speed", 13.89)),
        )

    return graph


def build_edge_index(graph):
    """Map SUMO edge IDs to graph endpoints and edge attributes."""
    return {
        data["edge_id"]: (u, v, data)
        for u, v, data in graph.edges(data=True)
    }


def update_live_costs(graph, edge_index):
    """Update every road's cost from the current SUMO traffic state."""
    for edge_id, (u, v, data) in edge_index.items():
        try:
            speed = traci.edge.getLastStepMeanSpeed(edge_id)
            vehicles = traci.edge.getLastStepVehicleNumber(edge_id)
        except traci.TraCIException:
            speed = 0.0
            vehicles = 0

        if speed <= 0.1:
            speed = min(data["max_speed"], 0.1)

        travel_time = data["length"] / speed
        density_penalty = min(2.5, 1.0 + vehicles / 20.0)

        # A live cost combines travel time and current traffic pressure.
        data["live_speed"] = speed
        data["live_vehicles"] = vehicles
        data["live_cost"] = travel_time * density_penalty


def edge_route_from_nodes(graph, nodes):
    """Convert a node path into SUMO edge IDs."""
    return [graph[nodes[i]][nodes[i + 1]]["edge_id"] for i in range(len(nodes) - 1)]


def find_dynamic_route(graph, edge_index, vehicle_id, destination_node):
    """Calculate the best live route from the vehicle's current edge."""
    current_edge = traci.vehicle.getRoadID(vehicle_id)

    if not current_edge or current_edge.startswith(":"):
        return None
    if current_edge not in edge_index:
        return None

    _, current_to, _ = edge_index[current_edge]

    if current_to == destination_node:
        return [current_edge]

    if current_to not in graph or destination_node not in graph:
        return None

    try:
        nodes = nx.shortest_path(
            graph,
            source=current_to,
            target=destination_node,
            weight="live_cost",
        )
    except nx.NetworkXNoPath:
        return None

    return [current_edge] + edge_route_from_nodes(graph, nodes)


def reroute_vehicle(graph, edge_index, vehicle_id, destination_node):
    """Recalculate and apply a route using only current SUMO traffic."""
    if vehicle_id not in traci.vehicle.getIDList():
        return None

    route = find_dynamic_route(graph, edge_index, vehicle_id, destination_node)
    if not route:
        return None

    current_route = list(traci.vehicle.getRoute(vehicle_id))
    if route == current_route:
        return route

    traci.vehicle.setRoute(vehicle_id, route)
    return route


def choose_vehicle(vehicle_id=None):
    """Return a requested active vehicle or the first active vehicle."""
    vehicles = list(traci.vehicle.getIDList())
    if vehicle_id and vehicle_id in vehicles:
        return vehicle_id
    return vehicles[0] if vehicles else None


def run_dynamic_optimization(
    vehicle_id=None,
    destination="C2",
    reroute_interval=10,
    max_steps=300,
    gui=False,
):
    """Run SUMO and continuously reroute a vehicle as traffic changes."""
    graph = load_network()
    edge_index = build_edge_index(graph)

    sumo_binary = "sumo-gui" if gui else "sumo"
    traci.start([sumo_binary, "-c", SUMO_CONFIG])

    print("\n========================================")
    print(" DYNAMIC REAL-TIME ROUTE OPTIMIZATION")
    print("========================================")
    print(f"Destination node: {destination}")
    print(f"Reroute interval: {reroute_interval} simulation seconds")

    last_reroute = -reroute_interval
    selected_vehicle = None
    previous_route = None
    reroute_count = 0

    try:
        for step in range(max_steps):
            traci.simulationStep()
            simulation_time = traci.simulation.getTime()

            if selected_vehicle not in traci.vehicle.getIDList():
                selected_vehicle = choose_vehicle(vehicle_id)
                if selected_vehicle:
                    print(f"Tracking vehicle: {selected_vehicle}")
                    previous_route = None

            if not selected_vehicle:
                continue

            if simulation_time - last_reroute < reroute_interval:
                continue

            update_live_costs(graph, edge_index)
            route = reroute_vehicle(
                graph,
                edge_index,
                selected_vehicle,
                destination,
            )

            if route:
                reroute_count += 1
                last_reroute = simulation_time
                route_changed = route != previous_route
                previous_route = route

                current_edge = traci.vehicle.getRoadID(selected_vehicle)
                current_speed = traci.vehicle.getSpeed(selected_vehicle)

                print(
                    f"Step {step:03d} | time={simulation_time:.0f}s | "
                    f"vehicle={selected_vehicle} | edge={current_edge} | "
                    f"speed={current_speed:.2f} m/s | "
                    f"route={'CHANGED' if route_changed else 'UNCHANGED'}"
                )
                print("   " + " -> ".join(route))
            else:
                last_reroute = simulation_time
                print(
                    f"Step {step:03d} | no route found for "
                    f"{selected_vehicle} to {destination}"
                )

    finally:
        traci.close()

    print("\n========================================")
    print(f"Dynamic reroutes performed: {reroute_count}")
    print("SUMO simulation completed")
    print("========================================")


def main():
    parser = argparse.ArgumentParser(
        description="Dynamic real-time SUMO route optimization"
    )
    parser.add_argument(
        "--vehicle-id",
        default=None,
        help="Vehicle to optimize. If omitted, the first active vehicle is used.",
    )
    parser.add_argument(
        "--destination",
        default="C2",
        help="Destination SUMO node (default: C2).",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=10,
        help="Seconds between live rerouting decisions (default: 10).",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=300,
        help="Maximum SUMO simulation steps (default: 300).",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Run with sumo-gui instead of sumo.",
    )
    args = parser.parse_args()

    if args.interval <= 0:
        raise ValueError("--interval must be greater than 0")
    if args.steps <= 0:
        raise ValueError("--steps must be greater than 0")

    run_dynamic_optimization(
        vehicle_id=args.vehicle_id,
        destination=args.destination,
        reroute_interval=args.interval,
        max_steps=args.steps,
        gui=args.gui,
    )


if __name__ == "__main__":
    main()
