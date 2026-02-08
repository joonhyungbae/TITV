"""
08b_dag_clean.py
Clean, publication-quality Directed Acyclic Graph for causal identification.
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np

FIG_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(FIG_DIR, exist_ok=True)


def draw_dag():
    """Draw clean causal DAG suitable for journal publication."""

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.set_xlim(-1.8, 11.8)
    ax.set_ylim(-1.5, 8.5)
    ax.set_aspect('equal')
    ax.axis('off')

    # ── Node positions ──
    nodes = {
        'U': (5.0, 7.2),    # Unobserved artistic capacity
        'S': (1.8, 4.2),    # Network stability (treatment)
        'P': (8.2, 4.2),    # Career plateau (outcome)
        'A': (5.0, 1.5),    # Cumulative achievement
        'T': (0.5, 1.5),    # Career time
        'N': (9.5, 1.5),    # Network size
    }

    # ── Node styling ──
    box_w, box_h = 2.4, 1.4
    hw, hh = box_w / 2, box_h / 2

    styles = {
        'U': dict(boxstyle='round,pad=0.3', facecolor='#FDE8E8',
                  edgecolor='#C0392B', linewidth=2, linestyle='dashed'),
        'S': dict(boxstyle='round,pad=0.3', facecolor='#D4EFDF',
                  edgecolor='#1E8449', linewidth=2.5),
        'P': dict(boxstyle='round,pad=0.3', facecolor='#FDEBD0',
                  edgecolor='#D35400', linewidth=2.5),
        'A': dict(boxstyle='round,pad=0.3', facecolor='#D6EAF8',
                  edgecolor='#2C3E50', linewidth=2.0),
        'T': dict(boxstyle='round,pad=0.3', facecolor='#D6EAF8',
                  edgecolor='#2C3E50', linewidth=2.0),
        'N': dict(boxstyle='round,pad=0.3', facecolor='#D6EAF8',
                  edgecolor='#2C3E50', linewidth=2.0),
    }

    labels = {
        'U': ('Unobserved\nArtistic Capacity\n$(U)$', 15),
        'S': ('Network\nStability\n$(S_t)$', 16.5),
        'P': ('Career\nPlateau\n$(P_t)$', 16.5),
        'A': ('Cumulative\nAchievement $(A_t)$', 14.25),
        'T': ('Career\nTime $(t)$', 14.25),
        'N': ('Network\nSize $(N_t)$', 14.25),
    }

    # Draw nodes
    for key, (cx, cy) in nodes.items():
        bbox = FancyBboxPatch(
            (cx - hw, cy - hh), box_w, box_h,
            **styles[key]
        )
        ax.add_patch(bbox)
        label_text, fontsize = labels[key]
        ax.text(cx, cy, label_text, ha='center', va='center',
                fontsize=fontsize, fontweight='bold', color='#2C3E50')

    # ── Box edge intersection ──
    def edge_point(node_key, toward_key, pad=0.12):
        """
        Compute the point on the box edge of `node_key` facing `toward_key`,
        plus a small outward padding.
        """
        cx, cy = nodes[node_key]
        tx, ty = nodes[toward_key]
        dx, dy = tx - cx, ty - cy
        length = np.hypot(dx, dy)
        if length == 0:
            return (cx, cy)
        ux, uy = dx / length, dy / length

        tx_edge = (hw / abs(ux)) if abs(ux) > 1e-10 else float('inf')
        ty_edge = (hh / abs(uy)) if abs(uy) > 1e-10 else float('inf')
        t = min(tx_edge, ty_edge)

        return (cx + ux * (t + pad), cy + uy * (t + pad))

    # ── Arrow helper ──
    def arrow(start_key, end_key, color='#7F8C8D', lw=1.5, ls='-',
              rad=0.08, label='', label_kw=None,
              head_length=0.8, head_width=0.45, pad=0.04):
        s = edge_point(start_key, end_key, pad=pad)
        e = edge_point(end_key, start_key, pad=pad)

        ax.annotate(
            '', xy=e, xytext=s,
            arrowprops=dict(
                arrowstyle=f'->,head_length={head_length},head_width={head_width}',
                color=color, lw=lw,
                linestyle=ls,
                shrinkA=0, shrinkB=0,
                connectionstyle=f'arc3,rad={rad}'
            )
        )
        if label:
            kw = label_kw or {}
            t = kw.get('t', 0.5)
            dx, dy = kw.get('dx', 0), kw.get('dy', 0)
            sc = nodes[start_key]
            ec = nodes[end_key]
            mx = sc[0] + (ec[0] - sc[0]) * t + dx
            my = sc[1] + (ec[1] - sc[1]) * t + dy
            ax.text(mx, my, label, fontsize=12.75, color=color,
                    ha='center', va='center', fontstyle='italic',
                    bbox=dict(boxstyle='round,pad=0.12', fc='white',
                              ec='none', alpha=0.9))

    # ── Edges ──

    # Core causal path: S → P (green, thick) — curves upward
    arrow('S', 'P', color='#1E8449', lw=4.0, rad=-0.3,
          head_length=1.0, head_width=0.55,
          label='Causal effect of interest',
          label_kw={'t': 0.5, 'dx': 0, 'dy': 1.0})

    # Reverse causality: P → S (blue) — curves downward
    arrow('P', 'S', color='#2471A3', lw=2.8,
          rad=-0.3,
          head_length=0.9, head_width=0.5,
          label='Reverse causality\n(blocked by $S_{t-2}$ lag)',
          label_kw={'t': 0.5, 'dx': 0, 'dy': -1.0})

    # Unmeasured confounding: U → S, U → P (red, dashed)
    arrow('U', 'S', color='#C0392B', lw=2.5, rad=0.1,
          head_length=0.9, head_width=0.5,
          label='Capacity\n→ network',
          label_kw={'t': 0.45, 'dx': -0.55, 'dy': 0})
    arrow('U', 'P', color='#C0392B', lw=2.5, rad=-0.1,
          head_length=0.9, head_width=0.5,
          label='Capacity\n→ stagnation',
          label_kw={'t': 0.45, 'dx': 0.55, 'dy': 0})

    # Measured covariates (gray)
    arrow('A', 'S', color='#7F8C8D', lw=1.8, rad=0.08,
          head_length=0.7, head_width=0.4)
    arrow('A', 'P', color='#7F8C8D', lw=1.8, rad=-0.08,
          head_length=0.7, head_width=0.4)
    arrow('T', 'S', color='#7F8C8D', lw=1.8, rad=0.0,
          head_length=0.7, head_width=0.4)
    arrow('T', 'A', color='#7F8C8D', lw=1.8, rad=0.0,
          head_length=0.7, head_width=0.4)
    # Network size → Stability (back-door: more institutions → lower stability)
    arrow('N', 'S', color='#7F8C8D', lw=1.8, rad=-0.15,
          head_length=0.7, head_width=0.4)
    # Network size → Plateau (protective effect, HR = 0.69)
    arrow('N', 'P', color='#7F8C8D', lw=1.8, rad=0.0,
          head_length=0.7, head_width=0.4)

    # ── Legend ──
    legend_items = [
        mpatches.Patch(facecolor='#D4EFDF', edgecolor='#1E8449',
                       label='Treatment ($S_t$: network stability)'),
        mpatches.Patch(facecolor='#FDEBD0', edgecolor='#D35400',
                       label='Outcome ($P_t$: career plateau)'),
        mpatches.Patch(facecolor='#FDE8E8', edgecolor='#C0392B',
                       linestyle='dashed',
                       label='Unobserved confounder ($U$)'),
        mpatches.Patch(facecolor='#D6EAF8', edgecolor='#2C3E50',
                       label='Measured covariates'),
    ]
    ax.legend(handles=legend_items, loc='lower center',
              bbox_to_anchor=(0.5, 0.02), ncol=2,
              fontsize=12.75, frameon=True, fancybox=True,
              borderpad=0.6, handlelength=1.5,
              edgecolor='#BDC3C7')

    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'fig1_causal_dag.png')
    plt.savefig(path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none',
                pad_inches=0.5)
    plt.close()
    print(f"DAG saved to {path}")
    return path


if __name__ == '__main__':
    draw_dag()
