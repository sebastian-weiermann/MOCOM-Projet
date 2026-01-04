import traci
import os
import sys
from pathlib import Path


# Set SUMO_HOME environment variable
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

# Ensure working directory is the script's directory (with SUMO config file)
os.chdir(Path(__file__).resolve().parent)

junctions = ["C2", "D2", "D3"]

# Start SUMO with configuration
sumoBinary = "sumo-gui"  # or "sumo" for command line
sumoCmd = [sumoBinary, "-c", "ff_heterogeneous.sumocfg", "--additional-files", "ff_bus.rou.xml, ff_bicycle.rou.xml"]

traci.start(sumoCmd)

# Run simulation for 3600 steps
step = 0
while step < 3600:
    traci.simulationStep()
    step += 1
    
# Get vehicle information
vehicle_ids = traci.vehicle.getIDList()

for veh_id in vehicle_ids:
    speed = traci.vehicle.getSpeed(veh_id)
    position = traci.vehicle.getPosition(veh_id)
    waiting_time = traci.vehicle.getWaitingTime(veh_id)
    
    print(f"Vehicle {veh_id}: Speed={speed:.2f}, Wait={waiting_time:.1f}s")

traci.close()