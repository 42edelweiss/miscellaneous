"""
DEMO "Vandermonde interpolation" pour ton site (Python)
======================================================

Ce script génère 3 assets prêts à être intégrés sur un site :
  1) video_frame_by_frame.mp4  : interpolation qui se construit (degré augmente)
  2) video_sensitivity.mp4     : p(y) vs p(y+ε) (instabilité)
  3) video_compare_nodes.mp4   : noeuds équidistants vs Chebyshev (split-screen)

✅ Sortie: dossier ./site_assets/
✅ Dépendances: numpy, matplotlib
✅ Vidéo MP4: nécessite ffmpeg installé sur ta machine

Installation:
  pip install numpy matplotlib

ffmpeg (Windows):
  - installe ffmpeg, puis vérifie: ffmpeg -version

Lance:
  python vandermonde_site_demo.py
"""

import os
import shutil
import subprocess
import numpy as np
import matplotlib.pyplot as plt


# --------------------------
# Réglages généraux
# --------------------------
SEED = 42
OUT_DIR = "site_assets"
FRAMES_DIR = os.path.join(OUT_DIR, "_frames")
FPS = 18
DPI = 160

NMAX = 20              # degré max (NMAX+1 points)
NG = 2000              # points pour courbe fine
Y_LIM = (-60, 10)      # fixe pour voir l'explosion
EPS_NOISE = 1e-10      # pour la sensibilité

# Style sobre (site)
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.spines.top": False,
    "axes.spines.right": False
})


# --------------------------
# Outils numériques
# --------------------------
def vandermonde_increasing(x: np.ndarray, deg: int) -> np.ndarray:
    """V = [1, x, x^2, ..., x^deg]"""
    return np.vander(x, N=deg + 1, increasing=True)

def poly_eval(c: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Évalue p(x)=c0+c1 x+...+cN x^N par Horner."""
    y = np.zeros_like(x, dtype=float)
    for ck in reversed(c):
        y = y * x + ck
    return y

def interp_coeffs_vandermonde(xi: np.ndarray, yi: np.ndarray, deg: int):
    """Résout Vc=y et retourne (c, cond(V))."""
    V = vandermonde_increasing(xi, deg)
    c = np.linalg.solve(V, yi)
    condV = np.linalg.cond(V)
    return c, condV

def chebyshev_nodes_on_01(N: int) -> np.ndarray:
    """
    N+1 noeuds de Chebyshev (Lobatto) sur [0,1]
      x_k = 0.5*(1 - cos(k*pi/N)), k=0..N
    """
    k = np.arange(N + 1)
    return 0.5 * (1.0 - np.cos(np.pi * k / N))

def ensure_dirs():
    os.makedirs(OUT_DIR, exist_ok=True)
    if os.path.exists(FRAMES_DIR):
        shutil.rmtree(FRAMES_DIR)
    os.makedirs(FRAMES_DIR, exist_ok=True)

def run_ffmpeg(frames_pattern: str, out_mp4: str, fps: int = FPS):
    """
    Crée un mp4 depuis des frames.
    frames_pattern ex: site_assets/_frames/fbf_%03d.png
    """
    cmd = [
        "ffmpeg", "-y",
        "-r", str(fps),
        "-i", frames_pattern,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        out_mp4
    ]
    subprocess.run(cmd, check=True)

def save_frame(path: str):
    plt.tight_layout()
    plt.savefig(path, dpi=DPI)
    plt.close()


# --------------------------
# 1) Frame-by-frame (noeuds équidistants)
# --------------------------
def make_video_frame_by_frame(rng: np.random.Generator):
    x_all = np.linspace(0.0, 1.0, NMAX + 1)
    y_all = rng.random(NMAX + 1)
    xg = np.linspace(0.0, 1.0, NG)

    for k in range(1, NMAX + 1):
        xi = x_all[:k+1]
        yi = y_all[:k+1]

        c, condV = interp_coeffs_vandermonde(xi, yi, k)
        yg = poly_eval(c, xg)
        yy = poly_eval(c, xi)
        err_nodes = np.max(np.abs(yi - yy))

        plt.figure(figsize=(9.6, 5.4))
        plt.plot(xg, yg, linewidth=2.2, label=r"$p_k(x)$")
        plt.plot(xi, yi, "o", markersize=6, label="points")
        plt.xlim(0, 1)
        plt.ylim(*Y_LIM)
        plt.title(f"Interpolation (construction) — k={k} | cond(V)={condV:.2e} | err(noeuds)={err_nodes:.2e}")
        plt.xlabel("x"); plt.ylabel("y")
        plt.legend(loc="best")

        frame_path = os.path.join(FRAMES_DIR, f"fbf_{k:03d}.png")
        save_frame(frame_path)

    out_mp4 = os.path.join(OUT_DIR, "video_frame_by_frame.mp4")
    run_ffmpeg(os.path.join(FRAMES_DIR, "fbf_%03d.png"), out_mp4)
    return out_mp4


# --------------------------
# 2) Sensibilité: p(y) vs p(y+eps)
# --------------------------
def make_video_sensitivity(rng: np.random.Generator):
    # On fixe N pour que la démo soit "choc"
    N = 15
    x = np.linspace(0.0, 1.0, N + 1)
    y = rng.random(N + 1)
    y2 = y + EPS_NOISE * rng.standard_normal(N + 1)

    xg = np.linspace(0.0, 1.0, NG)

    c1, condV = interp_coeffs_vandermonde(x, y, N)
    c2, _ = interp_coeffs_vandermonde(x, y2, N)

    y1g = poly_eval(c1, xg)
    y2g = poly_eval(c2, xg)
    diff_inf = np.max(np.abs(y2g - y1g))

    # Frames: on "révèle" progressivement la 2e courbe (effet storytelling)
    steps = 30
    for t in range(steps):
        alpha = t / (steps - 1)

        plt.figure(figsize=(9.6, 5.4))
        plt.plot(xg, y1g, linewidth=2.2, label=r"$p_N(y)$")
        plt.plot(xg, (1-alpha)*y1g + alpha*y2g, "--", linewidth=2.2, label=r"$p_N(y+\varepsilon)$")
        plt.plot(x, y, "o", markersize=5, label="points")
        plt.xlim(0, 1)
        plt.ylim(*Y_LIM)
        plt.title(
            f"Sensibilité — N={N} | cond(V)={condV:.2e} | ||Δp||∞={diff_inf:.2e} (ε={EPS_NOISE:.1e})"
        )
        plt.xlabel("x"); plt.ylabel("y")
        plt.legend(loc="best")

        frame_path = os.path.join(FRAMES_DIR, f"sens_{t:03d}.png")
        save_frame(frame_path)

    out_mp4 = os.path.join(OUT_DIR, "video_sensitivity.mp4")
    run_ffmpeg(os.path.join(FRAMES_DIR, "sens_%03d.png"), out_mp4)
    return out_mp4


# --------------------------
# 3) Comparaison: équidistants vs Chebyshev (split-screen)
# --------------------------
def make_video_compare_nodes(rng: np.random.Generator):
    N = 20
    x_eq = np.linspace(0.0, 1.0, N + 1)
    x_ch = chebyshev_nodes_on_01(N)

    # Même "fonction cible" (plus propre que rand pour comparer)
    # Choisis une fonction qui peut montrer Runge-like
    def f(x):
        return 1.0 / (1.0 + 25.0 * (2*x - 1.0)**2)

    y_eq = f(x_eq)
    y_ch = f(x_ch)
    xg = np.linspace(0.0, 1.0, NG)
    yg_true = f(xg)

    c_eq, cond_eq = interp_coeffs_vandermonde(x_eq, y_eq, N)
    c_ch, cond_ch = interp_coeffs_vandermonde(x_ch, y_ch, N)

    yg_eq = poly_eval(c_eq, xg)
    yg_ch = poly_eval(c_ch, xg)

    # Frames: on augmente N visuellement ? Ici on fait un “fade-in” + infos
    steps = 36
    for t in range(steps):
        alpha = t / (steps - 1)

        fig, axs = plt.subplots(1, 2, figsize=(12.8, 5.4), sharey=True)

        # Équidistants
        axs[0].plot(xg, yg_true, linewidth=1.8, label="f(x)")
        axs[0].plot(xg, (1-alpha)*yg_true + alpha*yg_eq, linewidth=2.2, label="interp")
        axs[0].plot(x_eq, y_eq, "o", markersize=4, label="noeuds")
        axs[0].set_title(f"Équidistants — N={N}\ncond(V)={cond_eq:.2e}")
        axs[0].set_xlim(0, 1)
        axs[0].set_xlabel("x")
        axs[0].set_ylabel("y")
        axs[0].grid(True, alpha=0.25)

        # Chebyshev
        axs[1].plot(xg, yg_true, linewidth=1.8, label="f(x)")
        axs[1].plot(xg, (1-alpha)*yg_true + alpha*yg_ch, linewidth=2.2, label="interp")
        axs[1].plot(x_ch, y_ch, "o", markersize=4, label="noeuds")
        axs[1].set_title(f"Chebyshev — N={N}\ncond(V)={cond_ch:.2e}")
        axs[1].set_xlim(0, 1)
        axs[1].set_xlabel("x")
        axs[1].grid(True, alpha=0.25)

        # Légende unique
        handles, labels = axs[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)

        fig.suptitle("Interpolation polynomiale: choix des nœuds (comparaison)", y=0.98)

        frame_path = os.path.join(FRAMES_DIR, f"cmp_{t:03d}.png")
        plt.tight_layout(rect=[0, 0.06, 1, 0.94])
        plt.savefig(frame_path, dpi=DPI)
        plt.close(fig)

    out_mp4 = os.path.join(OUT_DIR, "video_compare_nodes.mp4")
    run_ffmpeg(os.path.join(FRAMES_DIR, "cmp_%03d.png"), out_mp4)
    return out_mp4


def main():
    ensure_dirs()
    rng = np.random.default_rng(SEED)

    print("1) Génération: video_frame_by_frame.mp4 ...")
    p1 = make_video_frame_by_frame(rng)
    print("   ->", p1)

    print("2) Génération: video_sensitivity.mp4 ...")
    p2 = make_video_sensitivity(rng)
    print("   ->", p2)

    print("3) Génération: video_compare_nodes.mp4 ...")
    p3 = make_video_compare_nodes(rng)
    print("   ->", p3)

    # Nettoyage frames (optionnel: commente si tu veux garder les png)
    shutil.rmtree(FRAMES_DIR, ignore_errors=True)

    print("\n✅ Terminé. Tu peux mettre ces fichiers sur ton site :")
    print(" - site_assets/video_frame_by_frame.mp4")
    print(" - site_assets/video_sensitivity.mp4")
    print(" - site_assets/video_compare_nodes.mp4")
    print("\nAstuce site: utilise une balise <video controls autoplay muted loop>.")


if __name__ == "__main__":
    main()
