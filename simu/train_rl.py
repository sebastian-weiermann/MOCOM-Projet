import traci
import json
import xml.etree.ElementTree as ET
import math
import statistics
import os
import sys
import random
from typing import Any, Dict, List
from pathlib import Path
from collections import defaultdict
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from rl_model import QNetwork, ReplayBuffer
from rl_config import (
    ACTION_INTERVAL,
    CONTROLLED_TLS,
    MIN_GREEN,
    MODEL_DIR,
    MODEL_LOAD_PATH,
    MODEL_SAVE_PATH,
    PHASE_EW_GREEN,
    PHASE_EW_YELLOW,
    PHASE_NS_GREEN,
    PHASE_NS_YELLOW,
    PRIORITY_WEIGHTS,
    QUEUE_NORM,
    SWITCH_PENALTY,
    TIME_NORM,
    TLS_IN_EDGES,
    YELLOW_DURATION,
)


# Set SUMO_HOME environment variable
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

# Ensure working directory is the script's directory (with SUMO config file)
os.chdir(Path(__file__).resolve().parent)

junctions = ["C2", "D2", "D3"]

# Training configuration
TRAIN_EPISODES = 50
TRAIN_MODE = True
EPSILON_START = 1.0
EPSILON_MIN = 0.1
EPSILON_DECAY = 0.97  # per episode decay
GAMMA = 0.99
LEARNING_RATE = 5e-4
BATCH_SIZE = 64
REPLAY_CAPACITY = 50000
TARGET_UPDATE_STEPS = 200

MODEL_SAVE_EVERY = 5
LOAD_MODEL = False
SAVE_MODEL = True

SAVE_KPIS_EVERY_EPISODE = False

SIMULATION_TIME = 3600  # total simulation time in seconds
BEST_MODEL_SAVE_PATH = os.path.join(MODEL_DIR, "policy_best.pth")
BEST_AVG_REWARD = float("-inf")


RL_STATE: Dict[str, Any] = {
    "initialized": False,
}


def reset_episode_state():
    """Reset per-episode TLS tracking without reinitializing the networks."""
    if not RL_STATE.get("initialized", False):
        return
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
    RL_STATE["per_tls"] = per_tls


def maybe_load_model():
    if LOAD_MODEL and os.path.exists(MODEL_LOAD_PATH):
        RL_STATE["policy"].load_state_dict(torch.load(MODEL_LOAD_PATH))
        RL_STATE["target"].load_state_dict(RL_STATE["policy"].state_dict())


def maybe_save_model(episode):
    if not SAVE_MODEL:
        return
    os.makedirs(MODEL_DIR, exist_ok=True)
    if episode % MODEL_SAVE_EVERY == 0 or episode == TRAIN_EPISODES:
        torch.save(RL_STATE["policy"].state_dict(), MODEL_SAVE_PATH)


def maybe_save_best_model(avg_reward):
    global BEST_AVG_REWARD
    if not SAVE_MODEL:
        return
    if avg_reward > BEST_AVG_REWARD:
        os.makedirs(MODEL_DIR, exist_ok=True)
        BEST_AVG_REWARD = avg_reward
        torch.save(RL_STATE["policy"].state_dict(), BEST_MODEL_SAVE_PATH)
        print(
            f"Saved best model to {BEST_MODEL_SAVE_PATH} (avg reward {avg_reward:.2f})"
        )


def groupe04():
    """RL traffic-light controller for group 04 (training version)."""
    global RL_STATE, step

    def _init_rl():
        global RL_STATE
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

        RL_STATE.clear()
        RL_STATE.update({
            "initialized": True,
            "policy": policy,
            "target": target,
            "optimizer": optimizer,
            "buffer": buffer,
            "epsilon": EPSILON_START,
            "steps": 0,
            "per_tls": per_tls,
        })

    def _current_green_dir(phase):
        if phase in (PHASE_NS_GREEN, PHASE_NS_YELLOW):
            return 0  # NS
        return 1  # EW

    def _get_state(tls_id):
        edges = TLS_IN_EDGES[tls_id]
        q_n = traci.edge.getLastStepHaltingNumber(edges["N"]) / QUEUE_NORM
        q_s = traci.edge.getLastStepHaltingNumber(edges["S"]) / QUEUE_NORM
        q_e = traci.edge.getLastStepHaltingNumber(edges["E"]) / QUEUE_NORM
        q_w = traci.edge.getLastStepHaltingNumber(edges["W"]) / QUEUE_NORM
        phase = traci.trafficlight.getPhase(tls_id)
        phase_is_ns = 1.0 if _current_green_dir(phase) == 0 else 0.0
        time_since = min(1.0, (step - RL_STATE["per_tls"][tls_id]["last_switch_step"]) / TIME_NORM)
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

    def _set_phase_with_yellow(tls_id, target_phase):
        current_phase = traci.trafficlight.getPhase(tls_id)
        if current_phase == target_phase:
            return
        # go to yellow for current direction
        if _current_green_dir(current_phase) == 0:
            traci.trafficlight.setPhase(tls_id, PHASE_NS_YELLOW)
        else:
            traci.trafficlight.setPhase(tls_id, PHASE_EW_YELLOW)
        RL_STATE["per_tls"][tls_id]["yellow_remaining"] = YELLOW_DURATION
        RL_STATE["per_tls"][tls_id]["pending_target"] = target_phase

    if not RL_STATE["initialized"]:
        _init_rl()
        return

    # update yellow timers and apply pending targets
    for tls_id in CONTROLLED_TLS:
        tls_state = RL_STATE["per_tls"][tls_id]
        if tls_state["yellow_remaining"] > 0:
            tls_state["yellow_remaining"] -= 1
            if tls_state["yellow_remaining"] == 0 and tls_state["pending_target"] is not None:
                traci.trafficlight.setPhase(tls_id, tls_state["pending_target"])
                tls_state["last_switch_step"] = step
                tls_state["pending_target"] = None

    # decision and learning step
    for tls_id in CONTROLLED_TLS:
        tls_state = RL_STATE["per_tls"][tls_id]

        if tls_state["yellow_remaining"] > 0:
            continue

        if step - tls_state["last_decision_step"] < ACTION_INTERVAL:
            continue

        current_phase = traci.trafficlight.getPhase(tls_id)
        current_action = 0 if _current_green_dir(current_phase) == 0 else 1

        state = _get_state(tls_id)

        if tls_state["last_state"] is not None and tls_state["last_action"] is not None:
            reward = _compute_reward(tls_id, tls_state["last_action"], current_phase)
            RL_STATE["buffer"].push(
                tls_state["last_state"],
                tls_state["last_action"],
                reward,
                state,
                False,
            )

        # epsilon-greedy action selection
        if TRAIN_MODE and random.random() < RL_STATE["epsilon"]:
            action = random.randint(0, 1)
        else:
            with torch.no_grad():
                q_vals = RL_STATE["policy"](torch.tensor(state, dtype=torch.float32))
                action = int(torch.argmax(q_vals).item())

        # enforce minimum green
        if (step - tls_state["last_switch_step"]) < MIN_GREEN:
            action = current_action

        target_phase = PHASE_NS_GREEN if action == 0 else PHASE_EW_GREEN
        if target_phase != current_phase:
            _set_phase_with_yellow(tls_id, target_phase)

        tls_state["last_state"] = state
        tls_state["last_action"] = action
        tls_state["last_decision_step"] = step

    # DQN update
    if TRAIN_MODE and len(RL_STATE["buffer"]) >= BATCH_SIZE:
        states, actions, rewards, next_states, dones = RL_STATE["buffer"].sample(BATCH_SIZE)

        q_values = RL_STATE["policy"](states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q = RL_STATE["target"](next_states).max(1)[0]
            target_q = rewards + GAMMA * (1 - dones) * next_q

        loss = nn.MSELoss()(q_values, target_q)
        RL_STATE["optimizer"].zero_grad()
        loss.backward()
        RL_STATE["optimizer"].step()

        RL_STATE["steps"] += 1
        if RL_STATE["steps"] % TARGET_UPDATE_STEPS == 0:
            RL_STATE["target"].load_state_dict(RL_STATE["policy"].state_dict())

    return


# Start SUMO with configuration
sumoBinary = "sumo"  # or "sumo" for command line
sumoCmd = [sumoBinary, "-c", "ff_heterogeneous.sumocfg",
           "--additional-files", "detectors.add.xml",
           "--emission-output", "emissions.xml",
           "--no-warnings"]

# Training metrics tracking
training_metrics = {
    "episodes": [],
    "total_rewards": [],
    "avg_rewards": [],
    "epsilon_values": [],
}

for episode in range(1, TRAIN_EPISODES + 1):
    traci.start(sumoCmd)

    # initialize RL (networks) and reset per-episode state
    if not RL_STATE.get("initialized", False):
        groupe04()
        maybe_load_model()
    reset_episode_state()

    # Run simulation
    step = 0
    episode_total_reward = 0.0
    while step < SIMULATION_TIME:
        traci.simulationStep()
        # call RL traffic-light controller (implemented in `groupe04`)
        groupe04()
        step += 1

    traci.close()

    # Calculate episode reward from buffer (sum of recent rewards)
    if len(RL_STATE["buffer"]) > 0:
        recent_transitions = min(len(RL_STATE["buffer"]), 100)
        recent_rewards = [RL_STATE["buffer"].buffer[i][2] for i in range(-recent_transitions, 0)]
        episode_reward = sum(recent_rewards)
        avg_reward = episode_reward / len(recent_rewards)
    else:
        episode_reward = 0.0
        avg_reward = 0.0

    # Track metrics
    training_metrics["episodes"].append(episode)
    training_metrics["total_rewards"].append(episode_reward)
    training_metrics["avg_rewards"].append(avg_reward)
    training_metrics["epsilon_values"].append(RL_STATE["epsilon"])

    maybe_save_best_model(avg_reward)

    # Decay epsilon after each episode
    if RL_STATE["epsilon"] > EPSILON_MIN:
        RL_STATE["epsilon"] *= EPSILON_DECAY

    print(f"Episode {episode}/{TRAIN_EPISODES} - Total Reward: {episode_reward:.2f}, Avg Reward: {avg_reward:.2f}, Epsilon: {RL_STATE['epsilon']:.3f}")

    maybe_save_model(episode)

# Save training metrics
with open("training_metrics.json", "w") as f:
    json.dump(training_metrics, f, indent=2)
print("Saved training metrics to training_metrics.json")

# Create training plots
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
fig.suptitle('RL Training Progress', fontsize=16)

# Plot 1: Average Reward
axes[0].plot(training_metrics["episodes"], training_metrics["avg_rewards"], 'b-', linewidth=2)
axes[0].set_xlabel('Episode')
axes[0].set_ylabel('Average Reward')
axes[0].set_title('Average Reward per Episode')
axes[0].grid(True, alpha=0.3)

# Plot 2: Epsilon Decay
axes[1].plot(training_metrics["episodes"], training_metrics["epsilon_values"], 'r-', linewidth=2)
axes[1].set_xlabel('Episode')
axes[1].set_ylabel('Epsilon')
axes[1].set_title('Exploration Rate (Epsilon) Decay')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_progress.png', dpi=150, bbox_inches='tight')
print("Saved training plot to training_progress.png")
plt.show()
