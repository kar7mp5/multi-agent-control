import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# -----------------------
# Simple Multi-Agent Swarm (Top View)
# -----------------------
# Model: point-mass kinematics
#   x_{t+1} = x_t + v_t * dt
# Control:
#   v_i = v_goal(i) + v_consensus(i) + v_repulsion(i)
# - leader goes to goal
# - others follow via consensus
# - repulsion prevents collisions

def build_ring_neighbors(n: int):
    """Ring graph neighbors: i connected to i-1 and i+1 (mod n)."""
    nbrs = []
    for i in range(n):
        nbrs.append([(i - 1) % n, (i + 1) % n])
    return nbrs

def simulate():
    # ----- parameters -----
    np.random.seed(0)
    N = 10
    leader = 0

    dt = 0.05
    steps_per_frame = 2  # sim steps per animation frame

    # gains
    k_goal = 1.2          # leader attraction to goal
    k_cons = 0.8          # consensus / formation keeping
    k_rep = 0.04          # collision repulsion strength

    v_max = 1.8           # velocity limit
    rep_radius = 0.7      # start repulsion if closer than this
    damping = 0.15        # mild damping (helps stability)

    # initial positions (spread out)
    X = np.random.uniform(low=-4.0, high=-2.0, size=(N, 2))
    V = np.zeros((N, 2))

    # goal
    goal = np.array([4.0, 3.0])

    # neighbors graph
    neighbors = build_ring_neighbors(N)

    # ----- plotting -----
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-6, 6)
    ax.set_ylim(-6, 6)
    ax.grid(True, alpha=0.3)

    scat = ax.scatter(X[:, 0], X[:, 1], s=60)
    leader_dot = ax.scatter([X[leader, 0]], [X[leader, 1]], s=120, marker="*", zorder=3)
    goal_dot = ax.scatter([goal[0]], [goal[1]], s=120, marker="X", zorder=3)

    # draw neighbor edges (updated each frame)
    lines = []
    for i in range(N):
        for j in neighbors[i]:
            if j > i:  # avoid duplicates
                (ln,) = ax.plot([X[i, 0], X[j, 0]], [X[i, 1], X[j, 1]], alpha=0.25)
                lines.append((i, j, ln))

    title = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top")

    def step_dynamics():
        nonlocal X, V

        # consensus term: pull toward neighbor average
        V_cons = np.zeros_like(V)
        for i in range(N):
            nbr = neighbors[i]
            x_avg = np.mean(X[nbr], axis=0)
            V_cons[i] = k_cons * (x_avg - X[i])

        # goal term: only leader pulled to goal
        V_goal = np.zeros_like(V)
        V_goal[leader] = k_goal * (goal - X[leader])

        # repulsion: pairwise
        V_rep = np.zeros_like(V)
        for i in range(N):
            for j in range(N):
                if j == i:
                    continue
                d = X[i] - X[j]
                dist = np.linalg.norm(d) + 1e-9
                if dist < rep_radius:
                    # repulsive magnitude grows when closer
                    mag = k_rep * (1.0 / dist - 1.0 / rep_radius) / (dist**2)
                    V_rep[i] += mag * (d / dist)

        # combine + damping
        V = (1.0 - damping) * V + (V_goal + V_cons + V_rep)

        # clamp speed
        speeds = np.linalg.norm(V, axis=1) + 1e-9
        scale = np.minimum(1.0, v_max / speeds)
        V = V * scale[:, None]

        # integrate
        X = X + V * dt

    def update(frame_idx: int):
        for _ in range(steps_per_frame):
            step_dynamics()

        scat.set_offsets(X)
        leader_dot.set_offsets([X[leader]])
        goal_dot.set_offsets([goal])

        # update edges
        for (i, j, ln) in lines:
            ln.set_data([X[i, 0], X[j, 0]], [X[i, 1], X[j, 1]])

        # stop condition (leader near goal)
        dist_to_goal = np.linalg.norm(X[leader] - goal)
        title.set_text(f"frame={frame_idx:04d} | leader->goal dist={dist_to_goal:.2f}")
        return (scat, leader_dot, goal_dot, title, *[ln for (_, _, ln) in lines])

    ani = FuncAnimation(fig, update, frames=600, interval=30, blit=True)
    ani.save('animation.mp4', fps=30)
    plt.show()

if __name__ == "__main__":
    simulate()
