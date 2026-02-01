import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

# ----------------------------
# Style "article scientifique"
# ----------------------------
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "font.size": 10,
    "axes.linewidth": 0.8,
})

# ----------------------------
# Champ de "pseudorange" (distance)
# ----------------------------
def pseudorange(X, Y, cx, cy):
    return np.sqrt((X - cx)**2 + (Y - cy)**2)

# ----------------------------
# "Position estimée" = minimisation des écarts aux rayons (conceptuel)
# (Pour une animation pédagogique, on fait simple et stable)
# ----------------------------
def estimate_xy(centers, radii, grid_X, grid_Y):
    # Score: somme des erreurs absolues entre distances et rayons
    score = np.zeros_like(grid_X)
    for (cx, cy), r in zip(centers, radii):
        score += np.abs(np.sqrt((grid_X - cx)**2 + (grid_Y - cy)**2) - r)
    # Minimum sur la grille
    idx = np.unravel_index(np.argmin(score), score.shape)
    return grid_X[idx], grid_Y[idx]

# ----------------------------
# Paramètres visuels
# ----------------------------
x = np.linspace(-6, 6, 450)
y = np.linspace(-6, 6, 450)
X, Y = np.meshgrid(x, y)

levels = np.linspace(0.5, 6.0, 12)  # niveaux de contours
lw = 0.55  # épaisseur des lignes

fig, ax = plt.subplots(figsize=(6.2, 4.0))
ax.set_aspect("equal")
ax.axis("off")

# ----------------------------
# Timeline (en frames)
# ----------------------------
fps = 30

# Durées en secondes (ajuste si tu veux)
T1, T2, T3, T4 = 3.5, 4.0, 4.5, 4.0
N1, N2, N3, N4 = int(T1*fps), int(T2*fps), int(T3*fps), int(T4*fps)
N_total = N1 + N2 + N3 + N4

# "Vraie" position (juste pour ancrer l’histoire)
true_pos = np.array([1.2, -0.8])

# ----------------------------
# Helpers de dessin
# ----------------------------
def draw_contours(cx, cy):
    F = pseudorange(X, Y, cx, cy)
    ax.contour(X, Y, F, levels=levels, colors="black", linewidths=lw)

def draw_satellite(cx, cy):
    # Petit symbole simple (carré + croix), style gravure
    ax.plot([cx], [cy], marker="s", markersize=4, color="black")
    ax.plot([cx-0.25, cx+0.25], [cy, cy], color="black", linewidth=0.8)
    ax.plot([cx, cx], [cy-0.25, cy+0.25], color="black", linewidth=0.8)

def draw_text(title, subtitle=""):
    ax.text(-5.8, 5.4, title, ha="left", va="top", color="black")
    if subtitle:
        ax.text(-5.8, 4.95, subtitle, ha="left", va="top", color="black")

def draw_ellipse(center, a, b, angle_rad, n=240):
    t = np.linspace(0, 2*np.pi, n)
    # ellipse avant rotation
    ex = a*np.cos(t)
    ey = b*np.sin(t)
    # rotation
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    rx = c*ex - s*ey
    ry = s*ex + c*ey
    ax.plot(center[0] + rx, center[1] + ry, color="black", linewidth=1.0)

# ----------------------------
# Animation (4 actes)
# ----------------------------
def update(frame):
    ax.clear()
    ax.set_aspect("equal")
    ax.set_xlim(-6, 6)
    ax.set_ylim(-4.2, 6)
    ax.axis("off")

    # Acte I: 1 sphère (incertitude)
    if frame < N1:
        t = frame / max(1, (N1-1))
        # Satellite tourne légèrement autour
        cx = -2.8 + 0.6*np.cos(2*np.pi*t)
        cy =  1.4 + 0.6*np.sin(2*np.pi*t)
        draw_contours(cx, cy)
        draw_satellite(cx, cy)
        # True pos (optionnel, discret)
        ax.plot(true_pos[0], true_pos[1], marker="o", markersize=3, color="black")
        draw_text("ACTE I — Une mesure", "Une sphère (distance) ⇒ infinité de positions possibles")

    # Acte II: 2 sphères (cercle d'intersection)
    elif frame < N1 + N2:
        f = frame - N1
        t = f / max(1, (N2-1))
        # Deux satellites se rapprochent / s’écartent
        d = 2.8 - 1.2*np.cos(np.pi*t)  # commence séparé, se rapproche au milieu
        c1 = (-d, 1.1)
        c2 = ( d, 1.1)

        draw_contours(*c1)
        draw_contours(*c2)
        draw_satellite(*c1)
        draw_satellite(*c2)

        # Highlight visuel "zone d'intersection" via un niveau commun
        r = np.linalg.norm(true_pos - np.array(c1))
        # une courbe de niveau proche du rayon "vrai"
        F1 = pseudorange(X, Y, *c1)
        F2 = pseudorange(X, Y, *c2)
        ax.contour(X, Y, np.abs(F1-r) + np.abs(F2-r), levels=[0.25], colors="black", linewidths=1.1)

        ax.plot(true_pos[0], true_pos[1], marker="o", markersize=3, color="black")
        draw_text("ACTE II — Deux mesures", "Deux sphères ⇒ intersection (cercle) ⇒ ambiguïté reste")

    # Acte III: 3 sphères (point)
    elif frame < N1 + N2 + N3:
        f = frame - (N1 + N2)
        t = f / max(1, (N3-1))

        # Trois satellites : la 3e se déplace pour "résoudre" l'ambiguïté
        c1 = (-2.3, 1.0)
        c2 = ( 2.3, 1.0)
        c3 = ( 0.0, 3.4 - 1.2*np.cos(np.pi*t))  # descend un peu puis remonte

        centers = [c1, c2, c3]
        radii = [np.linalg.norm(true_pos - np.array(c)) for c in centers]

        for c in centers:
            draw_contours(*c)
            draw_satellite(*c)

        # Estimation simple (minimisation sur grille) pour afficher un point qui "converge"
        est = estimate_xy(centers, radii, X, Y)

        # On anime une interpolation visuelle vers l'estimation (pour effet pédagogique)
        est_vis = (1-t)*np.array([-0.5, -0.2]) + t*np.array(est)
        ax.plot(est_vis[0], est_vis[1], marker="o", markersize=5, color="black")
        ax.plot(true_pos[0], true_pos[1], marker="o", markersize=3, color="black")

        draw_text("ACTE III — Trois mesures", "Trois sphères ⇒ intersection ⇒ position (quasi) déterminée")

    # Acte IV: bruit + ellipses (DOP visuel)
    else:
        f = frame - (N1 + N2 + N3)
        t = f / max(1, (N4-1))

        # Géométrie change : satellites "alignés" (mauvais) → "bien répartis" (bon)
        # Interpolation entre deux géométries
        bad = [(-3.5, 2.5), (0.0, 2.6), (3.5, 2.7)]
        good = [(-3.0, 0.8), (2.8, 3.3), (0.0, 4.8)]

        centers = []
        for cb, cg in zip(bad, good):
            centers.append(((1-t)*cb[0] + t*cg[0], (1-t)*cb[1] + t*cg[1]))

        # Dessin des contours
        for c in centers:
            draw_contours(*c)
            draw_satellite(*c)

        # Nuage d'estimations (erreur) — plus grand quand géométrie mauvaise
        # sigma diminue avec t
        sigma = (1-t)*0.55 + t*0.18
        rng = np.random.default_rng(1234 + f)  # stable-ish
        samples = true_pos + rng.normal(0, sigma, size=(180, 2))

        # afficher nuage discret (points)
        ax.plot(samples[:, 0], samples[:, 1], linestyle="", marker=".", markersize=1.2, color="black")

        # ellipse d'erreur (approx: a,b)
        a = (1-t)*1.25 + t*0.45
        b = (1-t)*0.55 + t*0.28
        angle = (1-t)*0.15 + t*0.85  # rotation pour montrer "forme"
        draw_ellipse(true_pos, a, b, angle)

        ax.plot(true_pos[0], true_pos[1], marker="o", markersize=4, color="black")
        draw_text("ACTE IV — Erreur & géométrie", "Bruit + géométrie satellites ⇒ ellipse d'erreur (DOP)")

# ----------------------------
# Render video
# ----------------------------
ani = FuncAnimation(fig, update, frames=N_total, interval=1000/fps)

out_mp4 = "gps_math_animation.mp4"
writer = FFMpegWriter(fps=fps, bitrate=2200)

ani.save(out_mp4, writer=writer)
print(f"✅ Vidéo créée: {out_mp4}")
