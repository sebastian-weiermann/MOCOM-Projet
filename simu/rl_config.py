import os

# Controlled junctions
CONTROLLED_TLS = ["C2", "D2", "D3"]

# Phase indices (confirmed)
PHASE_NS_GREEN = 0
PHASE_NS_YELLOW = 1
PHASE_EW_GREEN = 2
PHASE_EW_YELLOW = 3

# Control timing
ACTION_INTERVAL = 5   # seconds between decisions
MIN_GREEN = 10        # minimum green time before switching
YELLOW_DURATION = 3   # duration to hold yellow before switching

# State normalization
QUEUE_NORM = 20.0     # normalization factor for queue lengths
TIME_NORM = 60.0      # normalization factor for time-since-switch

# Reward shaping
SWITCH_PENALTY = 1.0  # penalty for switching phases

# Incoming edges per controlled junction (verify if your network differs)
TLS_IN_EDGES = {
	"C2": {"N": "C3C2", "S": "C1C2", "E": "D2C2", "W": "B2C2"},
	"D2": {"N": "D3D2", "S": "D1D2", "E": "E2D2", "W": "C2D2"},
	"D3": {"N": "D4D3", "S": "D2D3", "E": "E3D3", "W": "C3D3"},
}

# Priority weights (bus on C2 NS; bikes on D2/D3 NS)
PRIORITY_WEIGHTS = {
	"C2": {"NS": 2.0, "EW": 1.0},
	"D2": {"NS": 1.5, "EW": 1.0},
	"D3": {"NS": 1.5, "EW": 1.0},
}

# Model paths
MODEL_DIR = "models_rl"
MODEL_LOAD_PATH = os.path.join(MODEL_DIR, "policy_latest.pth")
MODEL_SAVE_PATH = os.path.join(MODEL_DIR, "policy_latest.pth")
