import traci
import os
import sys
from pathlib import Path
from kpis import KpiCollector


# Set SUMO_HOME environment variable
if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

# Ensure working directory is the script's directory (with SUMO config file)
os.chdir(Path(__file__).resolve().parent)

# detectors
entry_detectors = [f"e1_{j}" for j in range(26)]
exit_detectors = [f"e1_{j}" for j in range(26, 52)]
print(f"Entry detectors: {len(entry_detectors)}")
print(f"Exit detectors: {len(exit_detectors)}")

warmup = 100  # seconds to ignore at start
measurement_time = 3600  # total simulation time

# Start SUMO with configuration (static lights from the net file)
sumoBinary = "sumo-gui"  # or "sumo" for command line
sumoCmd = [
    sumoBinary,
    "-c",
    "ff_heterogeneous.sumocfg",
    "--additional-files",
    "detectors.add.xml",
    "--emission-output",
    "emissions_static.xml",
]

traci.start(sumoCmd)

# KPI collector instance
collector = KpiCollector(
    entry_detectors,
    exit_detectors,
    warmup=warmup,
    measurement_time=measurement_time,
    stop_speed=0.1,
    emissions_file="emissions_static.xml",
)

# Run simulation for 3600 steps using default/static traffic lights
step = 0
while step < measurement_time:
    traci.simulationStep()
    t = traci.simulation.getTime()
    # snapshot of currently present vehicles to avoid using non-existent API 'exists'
    try:
        existing_vehicles = set(traci.vehicle.getIDList())
    except Exception:
        existing_vehicles = set()

    # delegate KPI processing for this timestep to the KPI collector
    collector.process_step(t, existing_vehicles)

    step += 1

traci.close()

# finalize and save KPIs
collector.save_kpis("kpis_static.json")
print("Saved KPIs to kpis_static.json")
