import csv
import traci

# SUMO configuration
sumo_cmd = [
    "sumo",
    "-c",
    "simulation/simulation.sumocfg"
]

print("Starting SUMO...", flush=True)

# Connect Python to SUMO
traci.start(sumo_cmd)

print("Connected to SUMO!", flush=True)

# Create CSV file
with open("data/traffic.csv", "w", newline="") as file:

    writer = csv.writer(file)

    # CSV column names
    writer.writerow([
        "time",
        "vehicle_id",
        "speed",
        "edge",
        "x",
        "y"
    ])

    # Run simulation
    while traci.simulation.getMinExpectedNumber() > 0:

        traci.simulationStep()

        current_time = traci.simulation.getTime()

        # Get all vehicles currently in SUMO
        vehicle_ids = traci.vehicle.getIDList()

        for vehicle_id in vehicle_ids:

            speed = traci.vehicle.getSpeed(vehicle_id)

            edge = traci.vehicle.getRoadID(vehicle_id)

            x, y = traci.vehicle.getPosition(vehicle_id)

            writer.writerow([
                current_time,
                vehicle_id,
                speed,
                edge,
                x,
                y
            ])

        # Show progress every 10 seconds
        if int(current_time) % 10 == 0:
            print(
                f"Time: {current_time:.0f}s | "
                f"Vehicles: {len(vehicle_ids)}",
                flush=True
            )

# Close TraCI
traci.close()

print("Simulation finished!", flush=True)
print("Traffic data saved to data/traffic.csv", flush=True)