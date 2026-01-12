import traci
import json
import xml.etree.ElementTree as ET
import math
import statistics
import os
import sys
from pathlib import Path
from collections import defaultdict


# Set SUMO_HOME environment variable
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

# Ensure working directory is the script's directory (with SUMO config file)
os.chdir(Path(__file__).resolve().parent)

junctions = ["C2", "D2", "D3"]

# KPI containers
entry_detectors = [f"e1_{j}" for j in range(26)]
exit_detectors  = [f"e1_{j}" for j in range(26, 52)]
print(f"Entry detectors: {len(entry_detectors)}")
print(f"Exit detectors: {len(exit_detectors)}")

veh_entry_time = {}      # vehID -> (entry timestamp, vehicleType)
travel_times = []        # list of travel times (s)
travel_times_by_mode = defaultdict(list)
waiting_times = []      # list of accumulated waiting times (s)
waiting_times_by_mode = defaultdict(list)
veh_in_subsection = set()
wait_inside = {}        # vehID -> accumulated stopped time inside subsection (s)
# throughput counters
throughput_count = 0
throughput_by_mode = defaultdict(int)

# measured vehicles for emissions and detailed per-vehicle KPIs
measured_vehicles = set()
measured_vehicle_type = {}
# stop tracking
veh_stopped = set()
stopped_count = 0
stopped_by_mode = defaultdict(int)


# speed (m/s) below which we consider the vehicle stopped
STOP_SPEED = 0.1

warmup = 100     # seconds to ignore at start
measurement_time = 3600  # total simulation time

# Start SUMO with configuration
sumoBinary = "sumo"  # or "sumo" for command lin
sumoCmd = [sumoBinary, "-c", "ff_heterogeneous.sumocfg",
           "--additional-files", "ff_bus.rou.xml, ff_bicycle.rou.xml, static_program.add.xml, detectors.add.xml",
           "--emission-output", "emissions.xml"]

traci.start(sumoCmd)

# Run simulation for 3600 steps
step = 0
while step < measurement_time:
    traci.simulationStep()
    t = traci.simulation.getTime()
    # snapshot of currently present vehicles to avoid using non-existent API 'exists'
    try:
        existing_vehicles = set(traci.vehicle.getIDList())
    except Exception:
        existing_vehicles = set()

    for det in entry_detectors:
        try:
            ids = traci.inductionloop.getLastStepVehicleIDs(det)
        except Exception:
            ids = []
        for vid in ids:
            # first time we see this vehicle at an entry detector -> mark entry
            if vid not in veh_entry_time:
                # try to get vehicle type now; fallback to 'unknown'
                try:
                    vtype = traci.vehicle.getVehicleClass(vid)
                except Exception:
                    vtype = 'unknown'
                # record entry time and init in-subsection wait accumulator
                veh_entry_time[vid] = (t, vtype)
                wait_inside[vid] = 0.0
                veh_in_subsection.add(vid)
    
    # record exits and compute KPIs
    for det in exit_detectors:
        try:
            ids = traci.inductionloop.getLastStepVehicleIDs(det)
        except Exception:
            ids = []
        for vid in ids:
            et = veh_entry_time.pop(vid, None)
            if et is not None:
                entry_t, vtype = et
                if t >= warmup:
                    travel = t - entry_t
                    travel_times.append(travel)
                    travel_times_by_mode[vtype].append(travel)
                    # mark measured vehicle for emissions/post-processing
                    measured_vehicles.add(vid)
                    measured_vehicle_type[vid] = vtype
                    # throughput counts
                    throughput_count += 1
                    throughput_by_mode[vtype] += 1
                    # use sampled stopped-time inside subsection as waiting time
                    w = wait_inside.pop(vid, 0.0)
                    waiting_times.append(w)
                    waiting_times_by_mode[vtype].append(w)
                    # stopped counters
                    if vid in veh_stopped:
                        stopped_count += 1
                        stopped_by_mode[vtype] += 1
                        veh_stopped.discard(vid)
                if vid in veh_in_subsection:
                    veh_in_subsection.remove(vid)

    # sample per-vehicle stopped time while inside subsection
    for vid in list(veh_in_subsection):
        if vid not in existing_vehicles:
            # vehicle left or was teleported; cleanup tracking
            veh_in_subsection.discard(vid)
            wait_inside.pop(vid, None)
            continue
        try:
            sp = traci.vehicle.getSpeed(vid)
        except Exception:
            sp = 0.0
        if sp <= STOP_SPEED:
            wait_inside[vid] = wait_inside.get(vid, 0.0) + 1.0
            veh_stopped.add(vid)

    step += 1

traci.close()

# Compute final KPIs
def _percentile(data, p):
    if not data:
        return None
    s = sorted(data)
    k = (len(s) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return s[int(k)]
    d0 = s[int(f)] * (c - k)
    d1 = s[int(c)] * (k - f)
    return d0 + d1

global_stats = {}
if travel_times:
    global_stats['count'] = len(travel_times)
    global_stats['mean_s'] = statistics.mean(travel_times)
    global_stats['std_s'] = statistics.pstdev(travel_times)
    global_stats['median_s'] = statistics.median(travel_times)
    global_stats['p90_s'] = _percentile(travel_times, 90)
    global_stats['min_s'] = min(travel_times)
    global_stats['max_s'] = max(travel_times)
else:
    global_stats['count'] = 0

mode_stats = {}
for mode, vals in travel_times_by_mode.items():
    if not vals:
        mode_stats[mode] = {'count': 0}
        continue
    mode_stats[mode] = {
        'count': len(vals),
        'mean_s': statistics.mean(vals),
        'std_s': statistics.pstdev(vals),
        'median_s': statistics.median(vals),
        'p90_s': _percentile(vals, 90),
        'min_s': min(vals),
        'max_s': max(vals),
    }

# waiting-time stats (global and per-mode)
waiting_global = {}
if waiting_times:
    waiting_global['count'] = len(waiting_times)
    waiting_global['total_s'] = sum(waiting_times)
    waiting_global['mean_s'] = statistics.mean(waiting_times)
    waiting_global['median_s'] = statistics.median(waiting_times)
    waiting_global['p90_s'] = _percentile(waiting_times, 90)
    waiting_global['min_s'] = min(waiting_times)
    waiting_global['max_s'] = max(waiting_times)
else:
    waiting_global['count'] = 0
    waiting_global['total_s'] = 0

waiting_mode_stats = {}
for mode, vals in waiting_times_by_mode.items():
    if not vals:
        waiting_mode_stats[mode] = {'count': 0}
        continue
    waiting_mode_stats[mode] = {
        'total_s': sum(vals),
        'count': len(vals),
        'mean_s': statistics.mean(vals),
        'median_s': statistics.median(vals),
        'p90_s': _percentile(vals, 90),
        'min_s': min(vals),
        'max_s': max(vals),
    }

# assemble KPIs
kpis = {'global_travel_time': global_stats, 'per_mode_travel_time': mode_stats}
# attach waiting stats to KPIs
kpis['global_waiting_time'] = waiting_global
kpis['per_mode_waiting_time'] = waiting_mode_stats
# throughput KPIs (normalize to measurement window length in seconds)
measurement_duration = max(1, measurement_time - warmup)
throughput_vph = throughput_count * 3600.0 / measurement_duration
throughput_mode_stats = {}
for mode, cnt in throughput_by_mode.items():
    throughput_mode_stats[mode] = {
        'count': cnt,
        'vph': cnt * 3600.0 / measurement_duration,
    }
kpis['throughput'] = {
    'total_count': throughput_count,
    'total_vph': throughput_vph,
    'per_mode': throughput_mode_stats,
}

# stopped / stop-rate KPIs
stop_rate = (stopped_count / throughput_count) if throughput_count else None
stopped_mode_stats = {}
for mode, cnt in stopped_by_mode.items():
    denom = len(travel_times_by_mode.get(mode, []))
    stopped_mode_stats[mode] = {
        'stopped_count': cnt,
        'stopped_fraction': (cnt / denom) if denom else None,
    }
kpis['stops'] = {
    'stopped_vehicles': stopped_count,
    'stop_rate_fraction': stop_rate,
    'per_mode': stopped_mode_stats,
}

# parse emissions.xml (if present) and sum emissions for measured vehicles
total_co2 = None
total_fuel = None
per_mode_emissions = {}
if os.path.exists('emissions.xml'):
    # We'll collect the last-seen (most recent) CO2 and fuel values per vehicle
    veh_last = {}
    try:
        tree = ET.parse('emissions.xml')
        root = tree.getroot()
        # emissions file often contains <timestep> elements which then contain <vehicle> entries
        for ts in root.findall('timestep'):
            for veh in ts.findall('vehicle'):
                vid = veh.get('id')
                attrib = {k.lower(): v for k, v in veh.attrib.items()}
                co2 = None
                fuel = None
                for key in ('co2', 'co2_emission', 'co2emission', 'co2emitted'):
                    if key in attrib:
                        try:
                            co2 = float(attrib[key])
                            break
                        except Exception:
                            pass
                for key in ('fuel', 'fuelconsumption', 'fuel_consumption'):
                    if key in attrib:
                        try:
                            fuel = float(attrib[key])
                            break
                        except Exception:
                            pass
                if co2 is None and fuel is None:
                    # skip vehicles without useful attributes in this record
                    continue
                entry = veh_last.setdefault(vid, {'co2': None, 'fuel': None})
                # store latest values (timesteps are ordered, so overwrite is fine)
                if co2 is not None:
                    entry['co2'] = co2
                if fuel is not None:
                    entry['fuel'] = fuel

        # now aggregate for only measured vehicles
        total_co2 = 0.0
        total_fuel = 0.0
        for vid in measured_vehicles:
            vals = veh_last.get(vid)
            if not vals:
                continue
            co2 = vals.get('co2')
            fuel = vals.get('fuel')
            if co2 is not None:
                total_co2 += co2
                vtype = measured_vehicle_type.get(vid, 'unknown')
                pm = per_mode_emissions.setdefault(vtype, {'co2': 0.0, 'fuel': 0.0, 'count': 0})
                pm['co2'] += co2
                pm['count'] += 1
            if fuel is not None:
                total_fuel += fuel
                vtype = measured_vehicle_type.get(vid, 'unknown')
                pm = per_mode_emissions.setdefault(vtype, {'co2': 0.0, 'fuel': 0.0, 'count': 0})
                pm['fuel'] += fuel
    except Exception:
        total_co2 = None
        total_fuel = None

if total_co2 is not None or total_fuel is not None:
    kpis['emissions'] = {
        'total_co2_g': total_co2,
        'total_fuel_l': total_fuel,
        'per_mode': per_mode_emissions,
    }
else:
    kpis['emissions'] = {
        'total_co2_g': None,
        'total_fuel_l': None,
        'per_mode': {},
    }

# Save KPIs
with open('kpis_subsection3.json', 'w') as jf:
    json.dump(kpis, jf, indent=2)
print('Saved KPIs to kpis_subsection3.json')