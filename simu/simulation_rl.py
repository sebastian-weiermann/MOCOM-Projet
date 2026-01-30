import traci
import json
import os
import random
from typing import Any, Dict
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from kpis import KpiCollector


def groupe04():
    """Self-contained RL traffic-light controller for group 04.
    
    This function manages all RL control logic, configuration, and online learning
    metrics internally. Can be called from any simulation loop.
    
    State and learning metrics persist across calls via a function attribute.
    """
    # Persist state on the function itself to keep everything self-contained
    if not hasattr(groupe04, "state"):
        groupe04.state = {"initialized": False}
    rl_state: Dict[str, Any] = groupe04.state
    
    # ===== CONFIGURATION (all internalized) =====
    # Controlled junctions
    CONTROLLED_TLS = ["C2", "D2", "D3"]
    
    # Phase indices
    PHASE_NS_GREEN = 0
    PHASE_NS_YELLOW = 1
    PHASE_EW_GREEN = 2
    PHASE_EW_YELLOW = 3
    
    # Control timing
    ACTION_INTERVAL = 5
    MIN_GREEN = 10
    YELLOW_DURATION = 3
    
    # State normalization
    QUEUE_NORM = 20.0
    TIME_NORM = 60.0
    
    # Reward shaping
    SWITCH_PENALTY = 1.0
    
    # Incoming edges per controlled junction
    TLS_IN_EDGES = {
        "C2": {"N": "C3C2", "S": "C1C2", "E": "D2C2", "W": "B2C2"},
        "D2": {"N": "D3D2", "S": "D1D2", "E": "E2D2", "W": "C2D2"},
        "D3": {"N": "D4D3", "S": "D2D3", "E": "E3D3", "W": "C3D3"},
    }
    
    # Priority weights
    PRIORITY_WEIGHTS = {
        "C2": {"NS": 2.0, "EW": 1.0},
        "D2": {"NS": 1.5, "EW": 1.0},
        "D3": {"NS": 1.5, "EW": 1.0},
    }
    
    # Model paths
    MODEL_DIR = "models_rl"
    MODEL_LOAD_PATH = os.path.join(MODEL_DIR, "policy_latest.pth")
    
    # Training/learning configuration
    ONLINE_LEARNING = True
    TRAIN_MODE = ONLINE_LEARNING
    EPSILON_START = 1.0
    EPSILON_MIN = 0.05
    EPSILON_DECAY = 0.9985
    GAMMA = 0.99
    LEARNING_RATE = 5e-4
    BATCH_SIZE = 64
    REPLAY_CAPACITY = 50000
    TARGET_UPDATE_STEPS = 200
    TRAINING_UPDATE_FREQ = 4
    LOAD_MODEL = True
    
    # Online learning metrics tracking
    TRACK_METRICS = True
    METRIC_INTERVAL = 100  # Track every N steps
    REWARD_WINDOW_SIZE = 100

    class QNetwork(nn.Module):
        def __init__(self, input_dim, output_dim):
            super().__init__()
            self.fc1 = nn.Linear(input_dim, 64)
            self.fc2 = nn.Linear(64, output_dim)

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            return self.fc2(x)

    class ReplayBuffer:
        def __init__(self, capacity):
            from collections import deque

            self.buffer = deque(maxlen=capacity)

        def push(self, state, action, reward, next_state, done):
            self.buffer.append((state, action, reward, next_state, done))

        def sample(self, batch_size):
            batch = random.sample(self.buffer, batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)
            return (
                torch.tensor(np.array(states), dtype=torch.float32),
                torch.tensor(actions, dtype=torch.int64),
                torch.tensor(rewards, dtype=torch.float32),
                torch.tensor(np.array(next_states), dtype=torch.float32),
                torch.tensor(dones, dtype=torch.float32),
            )

        def __len__(self):
            return len(self.buffer)

    def _init_rl():
        """Initialize RL networks and state."""
        policy = QNetwork(input_dim=6, output_dim=2)
        target = QNetwork(input_dim=6, output_dim=2)
        target.load_state_dict(policy.state_dict())
        optimizer = optim.Adam(policy.parameters(), lr=LEARNING_RATE)
        buffer = ReplayBuffer(REPLAY_CAPACITY)

        per_tls = {}
        for tls_id in CONTROLLED_TLS:
            current_phase = traci.trafficlight.getPhase(tls_id)
            per_tls[tls_id] = {
                "last_state": None,
                "last_action": None,
                "last_switch_step": 0,
                "last_decision_step": -ACTION_INTERVAL,
                "pending_target": None,
                "yellow_remaining": 0,
                "current_phase": current_phase,
            }

        rl_state.clear()
        rl_state.update({
            "initialized": True,
            "policy": policy,
            "target": target,
            "optimizer": optimizer,
            "buffer": buffer,
            "epsilon": EPSILON_START,
            "steps": 0,
            "decision_count": 0,
            "losses": [],
            "per_tls": per_tls,
            # Online learning metrics
            "learning_metrics": {
                "timesteps": [],
                "rewards": [],
                "avg_rewards": [],
                "epsilon": [],
                "avg_loss": [],
            },
            "reward_window": [],
        })
        
        # Load pre-trained model if available
        if LOAD_MODEL and os.path.exists(MODEL_LOAD_PATH):
            rl_state["policy"].load_state_dict(torch.load(MODEL_LOAD_PATH))
            rl_state["target"].load_state_dict(rl_state["policy"].state_dict())

    def _current_green_dir(phase):
        if phase in (PHASE_NS_GREEN, PHASE_NS_YELLOW):
            return 0  # NS
        return 1  # EW

    def _get_state(tls_id, step):
        edges = TLS_IN_EDGES[tls_id]
        q_n = traci.edge.getLastStepHaltingNumber(edges["N"]) / QUEUE_NORM
        q_s = traci.edge.getLastStepHaltingNumber(edges["S"]) / QUEUE_NORM
        q_e = traci.edge.getLastStepHaltingNumber(edges["E"]) / QUEUE_NORM
        q_w = traci.edge.getLastStepHaltingNumber(edges["W"]) / QUEUE_NORM
        phase = traci.trafficlight.getPhase(tls_id)
        phase_is_ns = 1.0 if _current_green_dir(phase) == 0 else 0.0
        time_since = min(1.0, (step - rl_state["per_tls"][tls_id]["last_switch_step"]) / TIME_NORM)
        return [q_n, q_s, q_e, q_w, phase_is_ns, time_since]

    def _compute_reward(tls_id, action, current_phase):
        edges = TLS_IN_EDGES[tls_id]
        ns_halt = (
            traci.edge.getLastStepHaltingNumber(edges["N"]) +
            traci.edge.getLastStepHaltingNumber(edges["S"])
        )
        ew_halt = (
            traci.edge.getLastStepHaltingNumber(edges["E"]) +
            traci.edge.getLastStepHaltingNumber(edges["W"])
        )
        weights = PRIORITY_WEIGHTS.get(tls_id, {"NS": 1.0, "EW": 1.0})
        reward = -(weights["NS"] * ns_halt + weights["EW"] * ew_halt)

        current_action = 0 if _current_green_dir(current_phase) == 0 else 1
        if action != current_action:
            reward -= SWITCH_PENALTY
        return reward

    def _set_phase_with_yellow(tls_id, target_phase, step):
        current_phase = traci.trafficlight.getPhase(tls_id)
        if current_phase == target_phase:
            return
        # go to yellow for current direction
        if _current_green_dir(current_phase) == 0:
            traci.trafficlight.setPhase(tls_id, PHASE_NS_YELLOW)
        else:
            traci.trafficlight.setPhase(tls_id, PHASE_EW_YELLOW)
        rl_state["per_tls"][tls_id]["yellow_remaining"] = YELLOW_DURATION
        rl_state["per_tls"][tls_id]["pending_target"] = target_phase

    # Initialize on first call
    if not rl_state.get("initialized", False):
        _init_rl()

    # Get current simulation step from SUMO
    step = int(traci.simulation.getTime())

    # update yellow timers and apply pending targets
    for tls_id in CONTROLLED_TLS:
        tls_state = rl_state["per_tls"][tls_id]
        if tls_state["yellow_remaining"] > 0:
            tls_state["yellow_remaining"] -= 1
            if tls_state["yellow_remaining"] == 0 and tls_state["pending_target"] is not None:
                traci.trafficlight.setPhase(tls_id, tls_state["pending_target"])
                tls_state["last_switch_step"] = step
                tls_state["pending_target"] = None

    # decision and learning step
    for tls_id in CONTROLLED_TLS:
        tls_state = rl_state["per_tls"][tls_id]

        if tls_state["yellow_remaining"] > 0:
            continue

        if step - tls_state["last_decision_step"] < ACTION_INTERVAL:
            continue

        current_phase = traci.trafficlight.getPhase(tls_id)
        current_action = 0 if _current_green_dir(current_phase) == 0 else 1

        obs = _get_state(tls_id, step)

        if tls_state["last_state"] is not None and tls_state["last_action"] is not None:
            reward = _compute_reward(tls_id, tls_state["last_action"], current_phase)
            rl_state["buffer"].push(
                tls_state["last_state"],
                tls_state["last_action"],
                reward,
                obs,
                False,
            )

        # epsilon-greedy action selection
        if TRAIN_MODE and random.random() < rl_state["epsilon"]:
            action = random.randint(0, 1)
        else:
            with torch.no_grad():
                q_vals = rl_state["policy"](torch.tensor(obs, dtype=torch.float32))
                action = int(torch.argmax(q_vals).item())

        # enforce minimum green
        if (step - tls_state["last_switch_step"]) < MIN_GREEN:
            action = current_action

        target_phase = PHASE_NS_GREEN if action == 0 else PHASE_EW_GREEN
        if target_phase != current_phase:
            _set_phase_with_yellow(tls_id, target_phase, step)

        tls_state["last_state"] = obs
        tls_state["last_action"] = action
        tls_state["last_decision_step"] = step

    # DQN update: training happens online during evaluation
    rl_state["decision_count"] = rl_state.get("decision_count", 0) + 1
    
    if TRAIN_MODE and rl_state["decision_count"] % TRAINING_UPDATE_FREQ == 0 and len(rl_state["buffer"]) >= BATCH_SIZE:
        states, actions, rewards, next_states, dones = rl_state["buffer"].sample(BATCH_SIZE)

        q_values = rl_state["policy"](states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q = rl_state["target"](next_states).max(1)[0]
            target_q = rewards + GAMMA * (1 - dones) * next_q

        loss = nn.MSELoss()(q_values, target_q)
        rl_state["optimizer"].zero_grad()
        loss.backward()
        rl_state["optimizer"].step()

        # Track loss for metrics
        if "losses" not in rl_state:
            rl_state["losses"] = []
        rl_state["losses"].append(loss.item())

        rl_state["steps"] += 1
        if rl_state["steps"] % TARGET_UPDATE_STEPS == 0:
            rl_state["target"].load_state_dict(rl_state["policy"].state_dict())

        if rl_state["epsilon"] > EPSILON_MIN:
            rl_state["epsilon"] *= EPSILON_DECAY
    
    # Track online learning metrics
    if TRACK_METRICS and TRAIN_MODE and step % METRIC_INTERVAL == 0:
        if len(rl_state["buffer"]) > 0:
            # Get recent rewards from buffer
            recent_count = min(len(rl_state["buffer"]), 20)
            recent_rewards = [rl_state["buffer"].buffer[i][2] for i in range(-recent_count, 0)]
            current_reward = sum(recent_rewards) / len(recent_rewards)
            
            rl_state["reward_window"].append(current_reward)
            if len(rl_state["reward_window"]) > REWARD_WINDOW_SIZE:
                rl_state["reward_window"].pop(0)
            
            # Compute average loss
            avg_loss = 0.0
            if len(rl_state.get("losses", [])) > 0:
                recent_losses = rl_state["losses"][-20:]
                avg_loss = sum(recent_losses) / len(recent_losses)
            
            rl_state["learning_metrics"]["timesteps"].append(step)
            rl_state["learning_metrics"]["rewards"].append(current_reward)
            rl_state["learning_metrics"]["avg_rewards"].append(
                sum(rl_state["reward_window"]) / len(rl_state["reward_window"])
            )
            rl_state["learning_metrics"]["epsilon"].append(rl_state.get("epsilon", 0.0))
            rl_state["learning_metrics"]["avg_loss"].append(avg_loss)

    return


def save_learning_metrics():
    """Save and plot online learning metrics. Call after simulation completes."""
    if not hasattr(groupe04, "state"):
        return
    
    state: Dict[str, Any] = groupe04.state
    if not state.get("initialized", False):
        return
    
    learning_metrics = state.get("learning_metrics", {})
    if not learning_metrics.get("timesteps"):
        return
    
    # Save metrics to JSON
    with open("online_learning_metrics.json", "w") as f:
        json.dump(learning_metrics, f, indent=2)
    print("Saved online learning metrics to online_learning_metrics.json")
    
    # Create plot
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle('Online Learning Performance During Evaluation', fontsize=16)
    
    # Plot 1: Reward progression
    axes[0].plot(learning_metrics["timesteps"], learning_metrics["rewards"], 
                 'b-', alpha=0.3, linewidth=1, label='Instant Reward')
    axes[0].set_xlabel('Simulation Step')
    axes[0].set_ylabel('Average Reward')
    axes[0].set_title('Reward Progression')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Epsilon decay
    axes[1].plot(learning_metrics["timesteps"], learning_metrics["epsilon"], 
                 'r-', linewidth=2)
    axes[1].set_xlabel('Simulation Step')
    axes[1].set_ylabel('Epsilon')
    axes[1].set_title('Exploration Rate During Evaluation')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('online_learning_progress.png', dpi=150, bbox_inches='tight')
    print("Saved online learning plot to online_learning_progress.png")
    plt.close()
    
    # Save online-learned model
    MODEL_DIR = "models_rl"
    MODEL_LOAD_PATH = os.path.join(MODEL_DIR, "policy_latest.pth")
    torch.save(state["policy"].state_dict(), 
               MODEL_LOAD_PATH.replace('.pth', '_eval_online.pth'))
    print(f"Saved online-learned model to {MODEL_LOAD_PATH.replace('.pth', '_eval_online.pth')}")

# Main simulation (only runs when script is executed directly)
if __name__ == "__main__":
    # Entry and exit detectors for KPI collection
    entry_detectors = {"e1_0", "e1_1", "e1_2", "e1_3"}
    exit_detectors = {"e1_44", "e1_45", "e1_46", "e1_47"}
    warmup = 900  # 15 min
    measurement_time = 3600  # 1h
    
    # Start SUMO with configuration
    sumoBinary = "sumo-gui"  # or "sumo" for command line
    sumoCmd = [sumoBinary, "-c", "ff_heterogeneous.sumocfg",
               "--additional-files", "detectors.add.xml",
               "--emission-output", "emissions_rl.xml"]

    traci.start(sumoCmd)

    # KPI collector instance
    collector = KpiCollector(
        entry_detectors,
        exit_detectors,
        warmup=warmup,
        measurement_time=measurement_time,
        stop_speed=0.1,
        emissions_file="emissions_rl.xml",
    )

    # Run simulation for 3600 steps
    step = 0
    while step < measurement_time:
        traci.simulationStep()
        t = traci.simulation.getTime()
        
        # snapshot of currently present vehicles
        try:
            existing_vehicles = set(traci.vehicle.getIDList())
        except Exception:
            existing_vehicles = set()
        
        # call RL traffic-light controller (fully self-contained)
        groupe04()

        # delegate KPI processing for this timestep to the KPI collector
        collector.process_step(t, existing_vehicles)

        step += 1

    traci.close()

    # finalize and save KPIs
    collector.save_kpis('kpis_subsection.json')
    print('Saved KPIs to kpis_subsection.json')

    # Save and plot online learning metrics
    save_learning_metrics()