import json
from pathlib import Path
import math

# Paths (adjust if needed)
STATIC_PATH = Path('kpis_static.json')
ACTUATED_PATH = Path('kpis_actuated.json')
RL_PATH = Path('kpis_subsection.json')

STATIC_AUG_PATH = Path('kpis_static_augmented.json')
ACTUATED_AUG_PATH = Path('kpis_actuated_augmented.json')
RL_AUG_PATH = Path('kpis_subsection_augmented.json')

# Load KPI JSONs
with open(STATIC_PATH, 'r') as f:
    kpi_static = json.load(f)
with open(ACTUATED_PATH, 'r') as f:
    kpi_actuated = json.load(f)
with open(RL_PATH, 'r') as f:
    kpi_rl = json.load(f)

with open(STATIC_AUG_PATH, 'r') as f:
    kpi_static_aug = json.load(f)
with open(ACTUATED_AUG_PATH, 'r') as f:
    kpi_actuated_aug = json.load(f)
with open(RL_AUG_PATH, 'r') as f:
    kpi_rl_aug = json.load(f)

def get_val(d, path):
    cur = d
    for p in path.split('/'):
        cur = cur[p]
    return cur

def fmt_val(x):
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "--"
    if isinstance(x, (int,)) or (isinstance(x, float) and float(x).is_integer()):
        return f"{int(x)}"
    return f"{x:.2f}"

def best_index(values, direction):
    # direction: 'min' or 'max'
    valid = [(i, v) for i, v in enumerate(values) if v is not None and not (isinstance(v, float) and (math.isnan(v) or math.isinf(v)))]
    if not valid:
        return None
    if direction == 'min':
        return min(valid, key=lambda t: t[1])[0]
    return max(valid, key=lambda t: t[1])[0]

# Define rows: (label, path, direction, source)
# source: 'base' or 'aug'
rows = [
    ("Throughput (vph)", "throughput/total_vph", "max", "base"),
    ("Stop rate (fraction)", "stops/stop_rate_fraction", "min", "base"),

    ("Per-mode throughput (bus)", "throughput/per_mode/bus/vph", "max", "base"),
    ("Per-mode throughput (bicycle)", "throughput/per_mode/bicycle/vph", "max", "base"),
    ("Per-mode throughput (passenger)", "throughput/per_mode/passenger/vph", "max", "base"),

    ("Per-mode stopped fraction (bus)", "stops/per_mode/bus/stopped_fraction", "min", "base"),
    ("Per-mode stopped fraction (bicycle)", "stops/per_mode/bicycle/stopped_fraction", "min", "base"),
    ("Per-mode stopped fraction (passenger)", "stops/per_mode/passenger/stopped_fraction", "min", "base"),

    ("Per-mode CO$_2$ (bus, g)", "emissions/per_mode/bus/co2", "min", "base"),
    ("Per-mode CO$_2$ (bicycle, g)", "emissions/per_mode/bicycle/co2", "min", "base"),
    ("Per-mode CO$_2$ (passenger, g)", "emissions/per_mode/passenger/co2", "min", "base"),

    ("Per-mode fuel (bus, L)", "emissions/per_mode/bus/fuel", "min", "base"),
    ("Per-mode fuel (bicycle, L)", "emissions/per_mode/bicycle/fuel", "min", "base"),
    ("Per-mode fuel (passenger, L)", "emissions/per_mode/passenger/fuel", "min", "base"),

    ("Mean travel time per person (s)", "global_travel_time/mean_s_per_person", "min", "aug"),
    ("Mean waiting time per person (s)", "global_waiting_time/mean_s_per_person", "min", "aug"),
    ("CO$_2$ per person (g)", "emissions/total_co2_g_per_person", "min", "aug"),
    ("Fuel per person (L)", "emissions/total_fuel_l_per_person", "min", "aug"),
]

# Choose source dicts
def select_sources(source):
    if source == "aug":
        return kpi_static_aug, kpi_actuated_aug, kpi_rl_aug
    return kpi_static, kpi_actuated, kpi_rl

# Build LaTeX table
lines = []
lines.append("\\begin{table}[htbp]")
lines.append("\\centering")
lines.append("\\caption{Summary of additional KPIs (static vs. actuated vs. RL).}")
lines.append("\\label{tab:kpi_summary}")
lines.append("\\renewcommand{\\arraystretch}{1.15}")
lines.append("\\begin{tabular}{lccc}")
lines.append("\\hline")
lines.append("\\textbf{KPI} & \\textbf{Static} & \\textbf{Actuated} & \\textbf{RL} \\\\")
lines.append("\\hline")

for label, path, direction, source in rows:
    ks, ka, kr = select_sources(source)
    vals = [get_val(ks, path), get_val(ka, path), get_val(kr, path)]
    best_i = best_index(vals, direction)

    formatted = []
    for i, v in enumerate(vals):
        s = fmt_val(v)
        if best_i is not None and i == best_i and s != "--":
            s = f"\\textbf{{{s}}}"
        formatted.append(s)

    lines.append(f"{label} & {formatted[0]} & {formatted[1]} & {formatted[2]} \\\\")

lines.append("\\hline")
lines.append("\\end{tabular}")
lines.append("\\end{table}")

latex_table = "\n".join(lines)
print(latex_table)