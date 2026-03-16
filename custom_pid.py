import time
import numpy as np

from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics


class PID:
    def __init__(self, kp, ki, kd, dt, integral_limit=None):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt

        self.integral = 0.0
        self.prev_error = 0.0
        self.integral_limit = integral_limit

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0

    def update(self, error):
        self.integral += error * self.dt
        if self.integral_limit is not None:
            self.integral = np.clip(
                self.integral,
                -self.integral_limit,
                self.integral_limit
            )

        derivative = (error - self.prev_error) / self.dt
        self.prev_error = error

        return self.kp * error + self.ki * self.integral + self.kd * derivative


env = CtrlAviary(
    drone_model=DroneModel.CF2X,
    num_drones=1,
    physics=Physics.PYB,
    gui=True
)

obs, info = env.reset()
dt = env.CTRL_TIMESTEP

# Position PID
pid_x = PID(kp=0.4, ki=0.0, kd=0.15, dt=dt)
pid_y = PID(kp=0.4, ki=0.0, kd=0.15, dt=dt)
pid_z = PID(kp=1.2, ki=0.0, kd=0.4, dt=dt)

# Attitude PID
pid_roll = PID(kp=80.0, ki=0.0, kd=20.0, dt=dt)
pid_pitch = PID(kp=80.0, ki=0.0, kd=20.0, dt=dt)
pid_yaw = PID(kp=30.0, ki=0.0, kd=5.0, dt=dt)

target_pos = np.array([1.0, 1.0, 1.0])
target_yaw = 0.0

# Approximate hover RPM
hover_rpm = 14500.0

for _ in range(5000):
    state = obs[0]

    # Common state layout in gym-pybullet-drones
    pos = state[0:3]
    quat = state[3:7]
    rpy = state[7:10]
    vel = state[10:13]
    ang_vel = state[13:16]

    # ---------------------------
    # 1) Position PID
    # ---------------------------
    err_pos = target_pos - pos

    ux = pid_x.update(err_pos[0])
    uy = pid_y.update(err_pos[1])
    uz = pid_z.update(err_pos[2])

    # Very simple mapping:
    # x error -> desired pitch
    # y error -> desired roll
    # z error -> collective thrust
    desired_roll = np.clip(-uy, -0.3, 0.3)
    desired_pitch = np.clip(ux, -0.3, 0.3)
    desired_yaw = target_yaw

    collective = hover_rpm + 3000.0 * uz
    collective = np.clip(collective, 10000.0, 22000.0)

    # ---------------------------
    # 2) Attitude PID
    # ---------------------------
    err_roll = desired_roll - rpy[0]
    err_pitch = desired_pitch - rpy[1]
    err_yaw = desired_yaw - rpy[2]

    u_roll = pid_roll.update(err_roll)
    u_pitch = pid_pitch.update(err_pitch)
    u_yaw = pid_yaw.update(err_yaw)

    # ---------------------------
    # 3) Motor mixing
    # X configuration approximate mixer
    # ---------------------------
    m1 = collective + u_roll - u_pitch + u_yaw
    m2 = collective - u_roll - u_pitch - u_yaw
    m3 = collective - u_roll + u_pitch + u_yaw
    m4 = collective + u_roll + u_pitch - u_yaw

    action = np.array([[m1, m2, m3, m4]], dtype=np.float32)
    action = np.clip(action, 0.0, 25000.0)

    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        obs, info = env.reset()
        pid_x.reset(); pid_y.reset(); pid_z.reset()
        pid_roll.reset(); pid_pitch.reset(); pid_yaw.reset()

    time.sleep(dt)

env.close()
