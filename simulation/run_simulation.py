import os
import sys
import csv
import traci


# --------------------------------------------------
# PROJECT PATH
# --------------------------------------------------

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


# --------------------------------------------------
# SUMO CONFIG
# --------------------------------------------------

SUMO_CONFIG = os.path.join(
    BASE_DIR,
    "simulation",
    "simulation.sumocfg"
)


# --------------------------------------------------
# SUMO TOOLS
# --------------------------------------------------

if "SUMO_HOME" in os.environ:
    tools = os.path.join(
        os.environ["SUMO_HOME"],
        "tools"
    )
    sys.path.append(tools)


sumo_binary = "sumo"


# --------------------------------------------------
# OUTPUT FILE
# --------------------------------------------------

OUTPUT_FILE = os.path.join(
    BASE_DIR,
    "data",
    "sumo_traffic.csv"
)


# --------------------------------------------------
# START SUMO
# --------------------------------------------------

traci.start(
    [
        sumo_binary,
        "-c",
        SUMO_CONFIG
    ]
)

print(" SUMO started successfully")


# --------------------------------------------------
# CREATE CSV
# --------------------------------------------------

with open(
    OUTPUT_FILE,
    "w",
    newline=""
) as csv_file:

    writer = csv.writer(csv_file)

    writer.writerow([
        "time",
        "vehicle_id",
        "road",
        "speed"
    ])


    # --------------------------------------------------
    # RUN SIMULATION
    # --------------------------------------------------

    step = 0

    while step < 100:

        traci.simulationStep()

        vehicles = traci.vehicle.getIDList()

        print(
            f"Step: {step} | "
            f"Vehicles: {len(vehicles)}"
        )


        # --------------------------------------------------
        # SAVE EVERY VEHICLE
        # --------------------------------------------------

        for vehicle_id in vehicles:

            speed = traci.vehicle.getSpeed(
                vehicle_id
            )

            road = traci.vehicle.getRoadID(
                vehicle_id
            )

            writer.writerow([
                step,
                vehicle_id,
                road,
                round(speed, 2)
            ])


            # Print first vehicle for monitoring
            if vehicle_id == vehicles[0]:

                print(
                    f"   Vehicle: {vehicle_id} | "
                    f"Road: {road} | "
                    f"Speed: {speed:.2f}"
                )


        step += 1


# --------------------------------------------------
# CLOSE SUMO
# --------------------------------------------------

traci.close()

print(" SUMO simulation completed")
print(f" Traffic data saved to: {OUTPUT_FILE}")