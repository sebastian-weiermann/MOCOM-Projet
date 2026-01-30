import traci
import math
import statistics
import xml.etree.ElementTree as ET
import os
import json
from collections import defaultdict


class KpiCollector:
    def __init__(self, entry_detectors, exit_detectors, warmup=100, measurement_time=3600, stop_speed=0.1, emissions_file='emissions.xml'):
        self.entry_detectors = list(entry_detectors)
        self.exit_detectors = list(exit_detectors)
        self.warmup = warmup
        self.measurement_time = measurement_time
        self.stop_speed = stop_speed
        self.emissions_file = emissions_file

        # KPI containers (internal state)
        self.veh_entry_time = {}      # vehID -> (entry timestamp, vehicleType)
        self.travel_times = []        # list of travel times (s)
        self.travel_times_by_mode = defaultdict(list)
        self.waiting_times = []      # list of accumulated waiting times (s)
        self.waiting_times_by_mode = defaultdict(list)
        self.veh_in_subsection = set()
        self.wait_inside = {}        # vehID -> accumulated stopped time inside subsection (s)
        # throughput counters
        self.throughput_count = 0
        self.throughput_by_mode = defaultdict(int)

        # measured vehicles for emissions and detailed per-vehicle KPIs
        self.measured_vehicles = set()
        self.measured_vehicle_type = {}
        # stop tracking
        self.veh_stopped = set()
        self.stopped_count = 0
        self.stopped_by_mode = defaultdict(int)

    def process_step(self, t, existing_vehicles):
        """Process detectors and per-vehicle sampling for one simulation step.

        Updates internal KPI containers.
        """
        # record entries from entry detectors
        for det in self.entry_detectors:
            try:
                ids = traci.inductionloop.getLastStepVehicleIDs(det)
            except Exception:
                ids = []
            for vid in ids:
                if vid not in self.veh_entry_time:
                    try:
                        vtype = traci.vehicle.getVehicleClass(vid)
                    except Exception:
                        vtype = 'unknown'
                    self.veh_entry_time[vid] = (t, vtype)
                    self.wait_inside[vid] = 0.0
                    self.veh_in_subsection.add(vid)

        # record exits and compute KPIs
        for det in self.exit_detectors:
            try:
                ids = traci.inductionloop.getLastStepVehicleIDs(det)
            except Exception:
                ids = []
            for vid in ids:
                et = self.veh_entry_time.pop(vid, None)
                if et is not None:
                    entry_t, vtype = et
                    if t >= self.warmup:
                        travel = t - entry_t
                        self.travel_times.append(travel)
                        self.travel_times_by_mode[vtype].append(travel)
                        self.measured_vehicles.add(vid)
                        self.measured_vehicle_type[vid] = vtype
                        self.throughput_count += 1
                        self.throughput_by_mode[vtype] += 1
                        w = self.wait_inside.pop(vid, 0.0)
                        self.waiting_times.append(w)
                        self.waiting_times_by_mode[vtype].append(w)
                        if vid in self.veh_stopped:
                            self.stopped_count += 1
                            self.stopped_by_mode[vtype] += 1
                            self.veh_stopped.discard(vid)
                    if vid in self.veh_in_subsection:
                        self.veh_in_subsection.remove(vid)

        # sample per-vehicle stopped time while inside subsection
        for vid in list(self.veh_in_subsection):
            if vid not in existing_vehicles:
                self.veh_in_subsection.discard(vid)
                self.wait_inside.pop(vid, None)
                continue
            try:
                sp = traci.vehicle.getSpeed(vid)
            except Exception:
                sp = 0.0
            if sp <= self.stop_speed:
                self.wait_inside[vid] = self.wait_inside.get(vid, 0.0) + 1.0
                self.veh_stopped.add(vid)

    @staticmethod
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

    def finalize_kpis(self):
        # travel-time stats
        global_stats = {}
        if self.travel_times:
            global_stats['count'] = len(self.travel_times)
            global_stats['mean_s'] = statistics.mean(self.travel_times)
            global_stats['std_s'] = statistics.pstdev(self.travel_times)
            global_stats['median_s'] = statistics.median(self.travel_times)
            global_stats['p90_s'] = self._percentile(self.travel_times, 90)
            global_stats['min_s'] = min(self.travel_times)
            global_stats['max_s'] = max(self.travel_times)
        else:
            global_stats['count'] = 0

        mode_stats = {}
        for mode, vals in self.travel_times_by_mode.items():
            if not vals:
                mode_stats[mode] = {'count': 0}
                continue
            mode_stats[mode] = {
                'count': len(vals),
                'mean_s': statistics.mean(vals),
                'std_s': statistics.pstdev(vals),
                'median_s': statistics.median(vals),
                'p90_s': self._percentile(vals, 90),
                'min_s': min(vals),
                'max_s': max(vals),
            }

        # waiting-time stats
        waiting_global = {}
        if self.waiting_times:
            waiting_global['count'] = len(self.waiting_times)
            waiting_global['total_s'] = sum(self.waiting_times)
            waiting_global['mean_s'] = statistics.mean(self.waiting_times)
            waiting_global['median_s'] = statistics.median(self.waiting_times)
            waiting_global['p90_s'] = self._percentile(self.waiting_times, 90)
            waiting_global['min_s'] = min(self.waiting_times)
            waiting_global['max_s'] = max(self.waiting_times)
        else:
            waiting_global['count'] = 0
            waiting_global['total_s'] = 0

        waiting_mode_stats = {}
        for mode, vals in self.waiting_times_by_mode.items():
            if not vals:
                waiting_mode_stats[mode] = {'count': 0}
                continue
            waiting_mode_stats[mode] = {
                'total_s': sum(vals),
                'count': len(vals),
                'mean_s': statistics.mean(vals),
                'median_s': statistics.median(vals),
                'p90_s': self._percentile(vals, 90),
                'min_s': min(vals),
                'max_s': max(vals),
            }

        # assemble KPIs
        kpis = {'global_travel_time': global_stats, 'per_mode_travel_time': mode_stats}
        kpis['global_waiting_time'] = waiting_global
        kpis['per_mode_waiting_time'] = waiting_mode_stats

        # throughput KPIs
        measurement_duration = max(1, self.measurement_time - self.warmup)
        throughput_vph = self.throughput_count * 3600.0 / measurement_duration
        throughput_mode_stats = {}
        for mode, cnt in self.throughput_by_mode.items():
            throughput_mode_stats[mode] = {
                'count': cnt,
                'vph': cnt * 3600.0 / measurement_duration,
            }
        kpis['throughput'] = {
            'total_count': self.throughput_count,
            'total_vph': throughput_vph,
            'per_mode': throughput_mode_stats,
        }

        # stopped / stop-rate KPIs
        stop_rate = (self.stopped_count / self.throughput_count) if self.throughput_count else None
        stopped_mode_stats = {}
        for mode, cnt in self.stopped_by_mode.items():
            denom = len(self.travel_times_by_mode.get(mode, []))
            stopped_mode_stats[mode] = {
                'stopped_count': cnt,
                'stopped_fraction': (cnt / denom) if denom else None,
            }
        kpis['stops'] = {
            'stopped_vehicles': self.stopped_count,
            'stop_rate_fraction': stop_rate,
            'per_mode': stopped_mode_stats,
        }

        # parse emissions file (if present) and sum emissions for measured vehicles
        total_co2 = None
        total_fuel = None
        per_mode_emissions = {}
        if os.path.exists(self.emissions_file):
            veh_last = {}
            try:
                tree = ET.parse(self.emissions_file)
                root = tree.getroot()
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
                            continue
                        entry = veh_last.setdefault(vid, {'co2': None, 'fuel': None})
                        if co2 is not None:
                            entry['co2'] = co2
                        if fuel is not None:
                            entry['fuel'] = fuel

                total_co2 = 0.0
                total_fuel = 0.0
                for vid in self.measured_vehicles:
                    vals = veh_last.get(vid)
                    if not vals:
                        continue
                    co2 = vals.get('co2')
                    fuel = vals.get('fuel')
                    if co2 is not None:
                        total_co2 += co2
                        vtype = self.measured_vehicle_type.get(vid, 'unknown')
                        pm = per_mode_emissions.setdefault(vtype, {'co2': 0.0, 'fuel': 0.0, 'count': 0})
                        pm['co2'] += co2
                        pm['count'] += 1
                    if fuel is not None:
                        total_fuel += fuel
                        vtype = self.measured_vehicle_type.get(vid, 'unknown')
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

        return kpis

    def save_kpis(self, filename='kpis_subsection3.json'):
        kpis = self.finalize_kpis()
        with open(filename, 'w') as jf:
            json.dump(kpis, jf, indent=2)
