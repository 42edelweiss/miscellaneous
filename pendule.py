import numpy as np
import matplotlib as mpl
import os

# définir ffmpeg AVANT tout import animation (si tu veux exporter MP4)
mpl.rcParams["animation.ffmpeg_path"] = r"C:\Program Files\IHMC CmapTools\bin\ffmpeg.exe"

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter


# =========================
# (A) TES INPUTS ICI
# =========================
g = 9.81      # gravité
ell = 1.0     # longueur ℓ
a = 0.35      # amortissement a>0

# conditions initiales
theta0 = np.deg2rad(70)  # angle initial (rad)
omega0 = 0.0             # vitesse angulaire initiale (rad/s)

# simulation
dt = 0.002
t_max = 20.0

# animation / export
fps = 60
dpi = 180
outfile = "pendulum_simulation.mp4"

points_per_frame = 6   # ↑ augmente => moins de frames => export plus rapide

# export frames PNG
frames_dir = "frames_pendulum"


# =========================
# (B) PENDULE + RK4
# =========================
def f(state):
    # state = [theta, omega]
    theta, omega = state
    dtheta = omega
    domega = -a * omega - (g / ell) * np.sin(theta)
    return np.array([dtheta, domega], dtype=float)

def rk4_step(s, h):
    k1 = f(s)
    k2 = f(s + 0.5*h*k1)
    k3 = f(s + 0.5*h*k2)
    k4 = f(s + h*k3)
    return s + (h/6.0)*(k1 + 2*k2 + 2*k3 + k4)

n = int(t_max / dt) + 1
t = np.linspace(0, t_max, n)

traj = np.zeros((n, 2), dtype=float)
traj[0] = np.array([theta0, omega0], dtype=float)

for i in range(n - 1):
    traj[i + 1] = rk4_step(traj[i], dt)

theta = traj[:, 0]
omega = traj[:, 1]

# position bob (pivot à l'origine)
x = ell * np.sin(theta)
y = -ell * np.cos(theta)

# unwrapping/affichage (optionnel) : ici on garde theta tel quel
# si tu veux éviter les sauts à ±pi dans le portrait de phase :
# theta_phase = np.unwrap(theta)
theta_phase = theta


# =========================
# (C) FIGURE LAYOUT
# =========================
fig = plt.figure(figsize=(10, 6))
fig.patch.set_facecolor("black")

gs = fig.add_gridspec(2, 2, width_ratios=[1.2, 1.4], wspace=0.20, hspace=0.25)

ax_pend = fig.add_subplot(gs[:, 0])   # pendule (grand)
ax_theta = fig.add_subplot(gs[0, 1])  # theta(t)
ax_phase = fig.add_subplot(gs[1, 1])  # phase (theta, omega)

for ax in [ax_pend, ax_theta, ax_phase]:
    ax.set_facecolor("black")
    ax.tick_params(colors="white", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("white")

# Pendule axis style
ax_pend.set_aspect("equal", adjustable="box")
ax_pend.set_xlim(-1.25*ell, 1.25*ell)
ax_pend.set_ylim(-1.25*ell, 0.35*ell)
ax_pend.set_xticks([])
ax_pend.set_yticks([])
ax_pend.set_title("Pendule amorti", color="white", fontsize=11, pad=8)

# theta(t)
ax_theta.set_title(r"$\theta(t)$", color="white", fontsize=10, pad=6)
ax_theta.set_xlabel("t (s)", color="white", fontsize=9)
ax_theta.set_ylabel(r"$\theta$ (rad)", color="white", fontsize=9)
ax_theta.set_xlim(t.min(), t.max())

# phase
ax_phase.set_title(r"Portrait de phase $(\theta,\omega)$", color="white", fontsize=10, pad=6)
ax_phase.set_xlabel(r"$\theta$ (rad)", color="white", fontsize=9)
ax_phase.set_ylabel(r"$\omega$ (rad/s)", color="white", fontsize=9)

# limites dynamiques (avec un peu de marge)
pad = 0.08
def expand(lo, hi, p=pad):
    r = hi - lo
    if r == 0:
        r = 1.0
    return lo - p*r, hi + p*r

th_lo, th_hi = expand(theta_phase.min(), theta_phase.max())
om_lo, om_hi = expand(omega.min(), omega.max())

ax_theta.set_ylim(expand(theta.min(), theta.max()))
ax_phase.set_xlim(th_lo, th_hi)
ax_phase.set_ylim(om_lo, om_hi)

# Info label
fig.text(
    0.02, 0.97,
    rf"Pendule: $g={g:g}$, $\ell={ell:g}$, $a={a:g}$   IC: $\theta_0={theta0:.3f}$ rad, $\omega_0={omega0:.3f}$ rad/s   dt={dt:g}",
    color="white", fontsize=10, ha="left", va="top"
)


# =========================
# (D) OBJETS À ANIMER
# =========================
# pendule : tige + bob + trace
rod_line, = ax_pend.plot([], [], lw=2.0)
bob_point, = ax_pend.plot([], [], marker="o", markersize=10)
trace_line, = ax_pend.plot([], [], lw=1.0)

rod_line.set_color("white")
bob_point.set_color("red")
trace_line.set_color("cyan")

# time series / phase
theta_line, = ax_theta.plot([], [], lw=1.2)
phase_line, = ax_phase.plot([], [], lw=1.2)
theta_line.set_color("yellow")
phase_line.set_color("magenta")

# petit pivot
ax_pend.plot([0], [0], marker="o", markersize=4, color="white")


# =========================
# (E) ANIMATION
# =========================
frames = n // points_per_frame

def init():
    rod_line.set_data([], [])
    bob_point.set_data([], [])
    trace_line.set_data([], [])
    theta_line.set_data([], [])
    phase_line.set_data([], [])
    return rod_line, bob_point, trace_line, theta_line, phase_line

def update(k):
    i = min((k + 1) * points_per_frame, n)

    # pendule courant
    xb, yb = x[i-1], y[i-1]
    rod_line.set_data([0, xb], [0, yb])
    bob_point.set_data([xb], [yb])

    # trace (traj du bob)
    trace_line.set_data(x[:i], y[:i])

    # theta(t)
    theta_line.set_data(t[:i], theta[:i])

    # phase
    phase_line.set_data(theta_phase[:i], omega[:i])

    return rod_line, bob_point, trace_line, theta_line, phase_line

anim = FuncAnimation(
    fig, update, frames=frames, init_func=init,
    interval=1000/fps, blit=False
)

# =========================
# (F) EXPORT : FRAMES PNG (comme ton script)
# =========================
os.makedirs(frames_dir, exist_ok=True)

for k in range(frames):
    update(k)
    fig.savefig(f"{frames_dir}/frame_{k:05d}.png", dpi=dpi)

print("Frames saved in:", frames_dir)

# =========================
# (G) (OPTION) EXPORT MP4
# =========================
# Si tu veux vraiment un mp4 directement, décommente :
# writer = FFMpegWriter(fps=fps, bitrate=1800)
# anim.save(outfile, writer=writer, dpi=dpi)
# print("Saved:", outfile)
