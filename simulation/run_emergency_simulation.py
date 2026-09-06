import argparse
import os
import sys
import traci

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG = os.path.join(BASE_DIR, "simulation", "emergency.sumocfg")


def main():
    parser = argparse.ArgumentParser(description="Run the SUMO emergency vehicle scenario with live rerouting")
    parser.add_argument("--gui", action="store_true", help="Run with sumo-gui")
    parser.add_argument("--interval", type=int, default=5, help="Seconds between emergency rerouting decisions")
    parser.add_argument("--steps", type=int, default=300, help="Maximum simulation steps")
    parser.add_argument("--vehicle-id", default="emergency_0", help="Emergency vehicle ID")
    parser.add_argument("--destination", default="C2", help="Destination node")
    args = parser.parse_args()
    if args.interval <= 0 or args.steps <= 0:
        raise ValueError("interval and steps must be greater than 0")

    # Import the reusable live-routing engine from the project.
    from optimization.dynamic_route import run_dynamic_optimization
    # The engine accepts an explicit SUMO config through the environment variable.
    os.environ["SMART_ROUTE_SUMO_CONFIG"] = CONFIG
    run_dynamic_optimization(
        vehicle_id=args.vehicle_id,
        destination=args.destination,
        reroute_interval=args.interval,
        max_steps=args.steps,
        gui=args.gui,
        emergency=True,
    )


if __name__ == "__main__":
    main()
