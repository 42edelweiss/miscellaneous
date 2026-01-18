import numpy as np
import matplotlib as mpl
import os

# définir ffmpeg AVANT tout import animation
mpl.rcParams["animation.ffmpeg_path"] = r"C:\Program Files\IHMC CmapTools\bin\ffmpeg.exe"

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter


# =========================
# (A) TES INPUTS ICI
# =========================
sigma = 10.0
rho   = 28.0
beta  = 8.0 / 3.0

# initial condition
x0, y0, z0 = 0.1, 0.0, 0.0

# simulation
dt = 0.01
t_max = 80.0          # ↑ plus grand = vidéo plus longue (simulation)

# export
fps = 60
dpi = 180
outfile = "lorenz_simulation.mp4"

# animation pacing
points_per_frame = 6   # ↑ augmente => moins de frames => export plus rapide
elev = 25
azim_per_frame = 0.20  # rotation caméra (deg / frame)

# =========================
# (B) LORENZ + RK4
# =========================
def f(state):
    x, y, z = state
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return np.array([dx, dy, dz], dtype=float)

def rk4_step(s, h):
    k1 = f(s)
    k2 = f(s + 0.5*h*k1)
    k3 = f(s + 0.5*h*k2)
    k4 = f(s + h*k3)
    return s + (h/6.0)*(k1 + 2*k2 + 2*k3 + k4)

n = int(t_max / dt)
traj = np.zeros((n, 3), dtype=float)
traj[0] = np.array([x0, y0, z0], dtype=float)

for i in range(n - 1):
    traj[i + 1] = rk4_step(traj[i], dt)

x, y, z = traj[:, 0], traj[:, 1], traj[:, 2]

# axis ranges (stable)
pad = 0.05
def expand(lo, hi, p=pad):
    r = hi - lo
    return lo - p*r, hi + p*r

xmin, xmax = expand(x.min(), x.max())
ymin, ymax = expand(y.min(), y.max())
zmin, zmax = expand(z.min(), z.max())

# =========================
# (C) FIGURE LAYOUT
# =========================
fig = plt.figure(figsize=(10, 6))
fig.patch.set_facecolor("black")

gs = fig.add_gridspec(3, 2, width_ratios=[2.2, 1.0], wspace=0.05, hspace=0.10)

ax3d = fig.add_subplot(gs[:, 0], projection="3d")
ax_xy = fig.add_subplot(gs[0, 1])
ax_xz = fig.add_subplot(gs[1, 1])
ax_yz = fig.add_subplot(gs[2, 1])

for ax in [ax3d, ax_xy, ax_xz, ax_yz]:
    ax.set_facecolor("black")

# 3D style
ax3d.set_axis_off()
ax3d.set_xlim(xmin, xmax)
ax3d.set_ylim(ymin, ymax)
ax3d.set_zlim(zmin, zmax)

# 2D style
def style_2d(ax, xlab, ylab):
    ax.tick_params(colors="white", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("white")
    ax.set_xlabel(xlab, color="white", fontsize=9)
    ax.set_ylabel(ylab, color="white", fontsize=9)

style_2d(ax_xy, "x", "y")
style_2d(ax_xz, "x", "z")
style_2d(ax_yz, "y", "z")

ax_xy.set_xlim(xmin, xmax); ax_xy.set_ylim(ymin, ymax)
ax_xz.set_xlim(xmin, xmax); ax_xz.set_ylim(zmin, zmax)
ax_yz.set_xlim(ymin, ymax); ax_yz.set_ylim(zmin, zmax)

# =========================
# (D) LINES
# =========================
main3d_line, = ax3d.plot([], [], [], lw=1.2)
xy_line, = ax_xy.plot([], [], lw=1.0)
xz_line, = ax_xz.plot([], [], lw=1.0)
yz_line, = ax_yz.plot([], [], lw=1.0)

main3d_line.set_color("yellow")
xy_line.set_color("cyan")
xz_line.set_color("magenta")
yz_line.set_color("orange")

# optional small label (still "simulation", not a full lecture)
fig.text(
    0.02, 0.97,
    rf"Lorenz: $\sigma={sigma:g},\ \rho={rho:g},\ \beta={beta:.4g}$   IC=({x0:g},{y0:g},{z0:g})   dt={dt:g}",
    color="white", fontsize=10, ha="left", va="top"
)

# =========================
# (E) ANIMATION
# =========================
frames = n // points_per_frame

def init():
    main3d_line.set_data([], [])
    main3d_line.set_3d_properties([])
    xy_line.set_data([], [])
    xz_line.set_data([], [])
    yz_line.set_data([], [])
    ax3d.view_init(elev=elev, azim=0)
    return main3d_line, xy_line, xz_line, yz_line

def update(k):
    i = min((k + 1) * points_per_frame, n)

    main3d_line.set_data(x[:i], y[:i])
    main3d_line.set_3d_properties(z[:i])

    xy_line.set_data(x[:i], y[:i])
    xz_line.set_data(x[:i], z[:i])
    yz_line.set_data(y[:i], z[:i])

    ax3d.view_init(elev=elev, azim=azim_per_frame * k)
    return main3d_line, xy_line, xz_line, yz_line

anim = FuncAnimation(fig, update, frames=frames, init_func=init, interval=1000/fps, blit=False)

# =========================
# (F) EXPORT MP4
# =========================
frames_dir = "frames_lorenz"
os.makedirs(frames_dir, exist_ok=True)

# Sauvegarde chaque frame en PNG
for k in range(frames):
    update(k)
    fig.savefig(f"{frames_dir}/frame_{k:05d}.png", dpi=dpi)

print("Frames saved in:", frames_dir)
print("Saved:", outfile)
