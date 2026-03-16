import time
import numpy as np

from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.enums import DroneModel, Physics

# Environment
env = CtrlAviary(
    drone_model=DroneModel.CF2X,
    num_drones=1,
    physics=Physics.PYB,
    gui=True
)

# PID controller
ctrl = DSLPIDControl(drone_model=DroneModel.CF2X)

obs, info = env.reset()

# Waypoints
targets = [
    np.array([0.0, 0.0, 1.0]),   # takeoff
    np.array([1.0, 0.0, 1.0]),   # move +x
    np.array([1.0, 1.0, 1.0]),   # move +y
    np.array([0.0, 1.0, 1.0]),   # move -x
    np.array([0.0, 0.0, 1.0]),   # back home
]

target_idx = 0
steps_per_target = 240 * 3  # 3 seconds at 240 Hz
step_count = 0

for i in range(240 * 20):
    # Current state vector for drone 0
    state = obs["0"]["state"] if isinstance(obs, dict) else obs[0]
    cur_pos = state[0:3]
    cur_quat = state[3:7]
    cur_vel = state[10:13]
    cur_ang_vel = state[13:16]

    target_pos = targets[target_idx]
    target_rpy = np.array([0.0, 0.0, 0.0])
    target_vel = np.array([0.0, 0.0, 0.0])
    target_rpy_rates = np.array([0.0, 0.0, 0.0])

    rpm, _, _ = ctrl.computeControl(
        control_timestep=env.CTRL_TIMESTEP,
        cur_pos=cur_pos,
        cur_quat=cur_quat,
        cur_vel=cur_vel,
        cur_ang_vel=cur_ang_vel,
        target_pos=target_pos,
        target_rpy=target_rpy,
        target_vel=target_vel,
        target_rpy_rates=target_rpy_rates,
    )

    action = {"0": rpm} if isinstance(obs, dict) else np.array([rpm])
    obs, reward, terminated, truncated, info = env.step(action)

    step_count += 1
    if step_count >= steps_per_target:
        step_count = 0
        target_idx = (target_idx + 1) % len(targets)

    if terminated or truncated:
        obs, info = env.reset()

    time.sleep(env.CTRL_TIMESTEP)

env.close()
