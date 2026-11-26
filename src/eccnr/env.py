"""Sonapse-LAB-compatible multi-task embodied environment for CORTEX-ECCNR.

Matches the 155-dim sensory observation and 5-dim motor action contract
(DRIVE, TURN, REACH, HEIGHT, GRIP in [-1, 1]) from Sonapse-LAB, while
supporting deterministic state cloning for counterfactual perturbation tests.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

OBS_DIM = 155
ACTION_DIM = 5
DRIVE, TURN, REACH, HEIGHT, GRIP = range(5)
ACTION_NAMES = ("drive", "turn", "reach", "height", "grip")

TASKS = (
    "navigation",
    "dynamic_obstacles",
    "memory_cue",
    "resource_allocation",
    "sequential",
)

# 6-class base task one-hot slot (compatible with Sonapse-LAB BASE_TASKS length=6)
_BASE_SLOT_DIM = 6
_RETINA_DIM = 135  # 135 + 8 (prop) + 6 (onehot) + 4 (cue) + 2 (reward, t/max_t) = 155
_CUE_DIM = 4


def govern_action(value: np.ndarray, max_drive: float = 1.0) -> np.ndarray:
    """Validate and clip 5-coordinate action vector."""
    action = np.asarray(value, dtype=np.float32)
    if action.shape != (ACTION_DIM,) or not np.isfinite(action).all():
        raise ValueError("action must be finite shape (5,)")
    action = np.clip(action, -1.0, 1.0).copy()
    action[DRIVE] = float(np.clip(action[DRIVE], -max_drive, max_drive))
    return action


@dataclass
class StepResult:
    obs: np.ndarray
    reward: float
    done: bool
    info: Dict[str, Any]


class ECCNRWorld:
    """Deterministic multi-task 2D+height continuous world with state cloning."""

    def __init__(self, max_steps: int = 80, seed: int = 42):
        self.max_steps = int(max_steps)
        self.base_seed = int(seed)
        self.rng = np.random.default_rng(seed)
        self.task: str = "navigation"
        self.t: int = 0
        self.last_reward: float = 0.0
        self.x: float = 0.0
        self.y: float = 0.0
        self.th: float = 0.0
        self.v: float = 0.0
        self.w: float = 0.0
        self.reach: float = 0.35
        self.height: float = 0.25
        self.grip: float = 0.0
        self.energy: float = 1.0
        self.stamina: float = 1.0
        self.target: Tuple[float, float] = (3.0, 0.0)
        self.decoy: Tuple[float, float] = (-3.0, 0.0)
        self.gate_pos: Tuple[float, float] = (1.5, 0.0)
        self.gate_triggered: bool = False
        self.cue_discovered: bool = False
        self.cue_code: int = 0
        self.obstacles: List[Dict[str, float]] = []
        self.resources: List[Dict[str, float]] = []
        self.cumulative_reward: float = 0.0
        self.collisions: int = 0
        self.subgoals_completed: int = 0

    def reset(self, task: str = "navigation", seed: Optional[int] = None) -> np.ndarray:
        if task not in TASKS:
            raise ValueError(f"Unknown task {task!r}; expected one of {TASKS}")
        self.task = task
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        else:
            self.rng = np.random.default_rng(self.base_seed)

        self.t = 0
        self.last_reward = 0.0
        self.cumulative_reward = 0.0
        self.collisions = 0
        self.subgoals_completed = 0
        self.x, self.y = 0.0, 0.0
        self.th = float(self.rng.uniform(-math.pi, math.pi))
        self.v, self.w = 0.0, 0.0
        self.reach, self.height, self.grip = 0.35, 0.25, 0.0
        self.energy, self.stamina = 1.0, 1.0
        self.gate_triggered = False
        self.cue_discovered = False
        self.cue_code = int(self.rng.integers(0, 2))

        ang = float(self.rng.uniform(-math.pi, math.pi))
        dist = float(self.rng.uniform(2.5, 4.2))
        self.target = (dist * math.cos(ang), dist * math.sin(ang))
        self.decoy = (-self.target[0], -self.target[1])
        self.gate_pos = (0.5 * self.target[0] + float(self.rng.uniform(-0.6, 0.6)),
                         0.5 * self.target[1] + float(self.rng.uniform(-0.6, 0.6)))

        self.obstacles = []
        n_obs = 4 if task == "dynamic_obstacles" else 2
        for i in range(n_obs):
            oa = float(self.rng.uniform(-math.pi, math.pi))
            od = float(self.rng.uniform(1.2, 3.5))
            self.obstacles.append({
                "x": od * math.cos(oa),
                "y": od * math.sin(oa),
                "vx": float(self.rng.uniform(-0.08, 0.08)),
                "vy": float(self.rng.uniform(-0.08, 0.08)),
                "r": 0.45,
            })

        self.resources = []
        n_res = 4 if task == "resource_allocation" else 2
        for _ in range(n_res):
            ra = float(self.rng.uniform(-math.pi, math.pi))
            rd = float(self.rng.uniform(1.0, 3.8))
            self.resources.append({"x": rd * math.cos(ra), "y": rd * math.sin(ra), "collected": 0.0})

        return self.observe()

    def clone_state(self) -> Dict[str, Any]:
        """Exact deterministic state snapshot for counterfactual perturbation."""
        return {
            "task": self.task,
            "t": self.t,
            "last_reward": self.last_reward,
            "x": self.x,
            "y": self.y,
            "th": self.th,
            "v": self.v,
            "w": self.w,
            "reach": self.reach,
            "height": self.height,
            "grip": self.grip,
            "energy": self.energy,
            "stamina": self.stamina,
            "target": self.target,
            "decoy": self.decoy,
            "gate_pos": self.gate_pos,
            "gate_triggered": self.gate_triggered,
            "cue_discovered": self.cue_discovered,
            "cue_code": self.cue_code,
            "obstacles": copy.deepcopy(self.obstacles),
            "resources": copy.deepcopy(self.resources),
            "cumulative_reward": self.cumulative_reward,
            "collisions": self.collisions,
            "subgoals_completed": self.subgoals_completed,
            "rng_state": copy.deepcopy(self.rng.bit_generator.state),
        }

    def restore_state(self, snap: Dict[str, Any]) -> None:
        self.task = snap["task"]
        self.t = snap["t"]
        self.last_reward = snap["last_reward"]
        self.x, self.y, self.th = snap["x"], snap["y"], snap["th"]
        self.v, self.w = snap["v"], snap["w"]
        self.reach, self.height, self.grip = snap["reach"], snap["height"], snap["grip"]
        self.energy, self.stamina = snap["energy"], snap["stamina"]
        self.target = tuple(snap["target"])
        self.decoy = tuple(snap["decoy"])
        self.gate_pos = tuple(snap["gate_pos"])
        self.gate_triggered = snap["gate_triggered"]
        self.cue_discovered = snap["cue_discovered"]
        self.cue_code = snap["cue_code"]
        self.obstacles = copy.deepcopy(snap["obstacles"])
        self.resources = copy.deepcopy(snap["resources"])
        self.cumulative_reward = snap["cumulative_reward"]
        self.collisions = snap["collisions"]
        self.subgoals_completed = snap["subgoals_completed"]
        self.rng.bit_generator.state = copy.deepcopy(snap["rng_state"])

    def _dist(self, px: float, py: float) -> float:
        return math.hypot(px - self.x, py - self.y)

    def step(self, raw_action: np.ndarray) -> StepResult:
        max_drive = 0.5 + 0.5 * self.stamina
        action = govern_action(raw_action, max_drive=max_drive)
        self.t += 1

        prev_goal = self.target if (self.task != "memory_cue" or self.cue_code == 0) else self.decoy
        prev_dist = self._dist(*prev_goal)

        # Physics integration
        self.v = 0.65 * self.v + 0.35 * float(action[DRIVE]) * 0.28
        self.w = 0.55 * self.w + 0.45 * float(action[TURN]) * 0.45
        self.th = math.atan2(math.sin(self.th + self.w), math.cos(self.th + self.w))
        self.x = float(np.clip(self.x + self.v * math.cos(self.th), -6.0, 6.0))
        self.y = float(np.clip(self.y + self.v * math.sin(self.th), -6.0, 6.0))
        self.reach = float(np.clip(0.575 + 0.325 * action[REACH], 0.25, 0.90))
        self.height = float(np.clip(0.375 + 0.375 * action[HEIGHT], 0.0, 0.75))
        self.grip = float(action[GRIP])

        # Energy & stamina dynamics
        effort = 0.004 * abs(float(action[DRIVE])) + 0.002 * abs(float(action[TURN]))
        self.energy = float(np.clip(self.energy - effort, 0.0, 1.0))
        self.stamina = float(np.clip(self.stamina - 0.5 * effort + 0.001, 0.1, 1.0))

        # Move dynamic obstacles
        for obs in self.obstacles:
            obs["x"] += obs["vx"]
            obs["y"] += obs["vy"]
            if abs(obs["x"]) > 4.8:
                obs["vx"] *= -1.0
            if abs(obs["y"]) > 4.8:
                obs["vy"] *= -1.0

        # Reward shaping & task logic
        reward = -0.01  # step time cost
        goal = self.target if (self.task != "memory_cue" or self.cue_code == 0) else self.decoy
        new_dist = self._dist(*goal)
        reward += 0.35 * (prev_dist - new_dist)

        # Collisions
        for obs in self.obstacles:
            if self._dist(obs["x"], obs["y"]) < obs["r"] + 0.25:
                reward -= 0.25
                self.collisions += 1

        # Resources
        for res in self.resources:
            if res["collected"] == 0.0 and self._dist(res["x"], res["y"]) < 0.65:
                res["collected"] = 1.0
                self.energy = float(np.clip(self.energy + 0.25, 0.0, 1.0))
                self.stamina = float(np.clip(self.stamina + 0.25, 0.0, 1.0))
                reward += 0.40 if self.task == "resource_allocation" else 0.15

        # Sequential subgoals
        done = False
        success = False
        if self.task == "sequential":
            if not self.cue_discovered and self.t >= 4:
                self.cue_discovered = True
                self.subgoals_completed += 1
                reward += 0.20
            if not self.gate_triggered and self._dist(*self.gate_pos) < 0.85:
                self.gate_triggered = True
                self.subgoals_completed += 1
                reward += 0.50
            if self.gate_triggered and new_dist < 0.65:
                self.subgoals_completed += 1
                reward += 1.50
                success = True
                done = True
        else:
            if new_dist < 0.65:
                reward += 1.25
                success = True
                done = True
            elif self.task == "memory_cue":
                wrong = self.decoy if self.cue_code == 0 else self.target
                if self._dist(*wrong) < 0.65:
                    reward -= 1.0
                    done = True

        if self.task == "resource_allocation" and self.energy <= 0.05:
            reward -= 0.5
            done = True

        if self.t >= self.max_steps:
            done = True

        self.last_reward = float(reward)
        self.cumulative_reward += float(reward)
        return StepResult(
            obs=self.observe(),
            reward=float(reward),
            done=done,
            info={
                "success": success,
                "dist_to_goal": new_dist,
                "collisions": self.collisions,
                "energy": self.energy,
                "subgoals_completed": self.subgoals_completed,
                "t": self.t,
            },
        )

    def observe(self) -> np.ndarray:
        """155-dim observation matching Sonapse-LAB ComplexWorld contract."""
        # Retina: 45 rays x 3 channels (goal/cue, obstacle, resource) = 135
        ret = np.zeros(_RETINA_DIM, dtype=np.float32)
        goal = self.target if (self.task != "memory_cue" or self.cue_code == 0) else self.decoy
        # In memory_cue, goal beacon hides after t > 12 so agent must remember cue
        show_goal = (self.task != "memory_cue") or (self.t <= 12)
        if show_goal:
            gx, gy = goal
            if self.task == "sequential" and not self.gate_triggered:
                gx, gy = self.gate_pos
            ang = math.atan2(gy - self.y, gx - self.x) - self.th
            ang = math.atan2(math.sin(ang), math.cos(ang))
            ray = int(np.clip(((ang + math.pi) / (2 * math.pi)) * 45, 0, 44))
            d = self._dist(gx, gy)
            ret[ray * 3] = float(np.clip(1.0 / (1.0 + 0.4 * d), 0.0, 1.0))

        for obs in self.obstacles:
            ang = math.atan2(obs["y"] - self.y, obs["x"] - self.x) - self.th
            ang = math.atan2(math.sin(ang), math.cos(ang))
            ray = int(np.clip(((ang + math.pi) / (2 * math.pi)) * 45, 0, 44))
            d = self._dist(obs["x"], obs["y"])
            ret[ray * 3 + 1] = max(ret[ray * 3 + 1], float(np.clip(1.0 / (1.0 + 0.6 * d), 0.0, 1.0)))

        for res in self.resources:
            if res["collected"] == 0.0:
                ang = math.atan2(res["y"] - self.y, res["x"] - self.x) - self.th
                ang = math.atan2(math.sin(ang), math.cos(ang))
                ray = int(np.clip(((ang + math.pi) / (2 * math.pi)) * 45, 0, 44))
                d = self._dist(res["x"], res["y"])
                ret[ray * 3 + 2] = max(ret[ray * 3 + 2], float(np.clip(1.0 / (1.0 + 0.5 * d), 0.0, 1.0)))

        prop = np.array([
            (self.reach - 0.25) / 0.65,
            self.height / 0.75,
            self.grip,
            self.energy,
            self.v / 0.4,
            self.w / 0.5,
            math.sin(self.th),
            math.cos(self.th),
        ], dtype=np.float32)

        onehot = np.zeros(_BASE_SLOT_DIM, dtype=np.float32)
        onehot[TASKS.index(self.task)] = 1.0

        cue = np.zeros(_CUE_DIM, dtype=np.float32)
        if self.task != "memory_cue" or self.t <= 12:
            cue[self.cue_code] = 1.0
        cue[2] = self.stamina
        cue[3] = 1.0 if self.gate_triggered else 0.0

        tail = np.array([self.last_reward, self.t / max(1, self.max_steps)], dtype=np.float32)
        obs = np.concatenate([ret, prop, onehot, cue, tail]).astype(np.float32)
        assert obs.shape == (OBS_DIM,)
        return obs
