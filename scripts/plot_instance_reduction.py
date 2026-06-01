#!/usr/bin/env python3
"""Gera 3 gráficos de redução de instâncias e F1-Score."""

import csv
import pathlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

CSV_PATH = pathlib.Path(__file__).parent.parent / "results" / "EP04_CUDA_benchmark_raw.csv"
FIGS_PATH = pathlib.Path(__file__).parent.parent / "results" / "figs"

MODES = {"seq": "Sequencial", "omp": "OpenMP 12t", "cuda": "CUDA (GPU 1-NN)"}
COLORS = {"seq": "#4C72B0", "omp": "#DD8452", "cuda": "#55A868"}

# ---------------------------------------------------------------------------
# Leitura
# ---------------------------------------------------------------------------
rows = []
with open(CSV_PATH, newline="") as f:
    for r in csv.DictReader(f):
        rows.append(r)

datasets_ordered = []
seen = set()
for r in rows:
    d = r["Dataset"]
    if d not in seen:
        datasets_ordered.append(d)
        seen.add(d)

data = {}
for r in rows:
    ds = r["Dataset"]
    mode = r["Mode"]
    threads = r["Threads"]
    if mode == "omp" and threads != "12":
        continue
    data[(ds, mode)] = {
        "reducao": float(r["Reducao_pct"]),
        "f1": float(r["F1"]),
        "N": int(r["N"]),
        "selected": round(int(r["N"]) * (1 - float(r["Reducao_pct"]) / 100)),
    }

datasets = datasets_ordered
x = np.arange(len(datasets))
width = 0.25
offsets = {"seq": -width, "omp": 0, "cuda": width}

# ---------------------------------------------------------------------------
# Fig 6 — Redução em porcentagem
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(14, 5))

for mode, label in MODES.items():
    vals = [data.get((ds, mode), {}).get("reducao", 0) for ds in datasets]
    bars = ax.bar(x + offsets[mode], vals, width * 0.9,
                  label=label, color=COLORS[mode], alpha=0.85, zorder=3)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{v:.1f}%", ha="center", va="bottom", fontsize=6.5, color=COLORS[mode])

Ns = [data.get((ds, "seq"), {}).get("N", "") for ds in datasets]
ax.set_xticks(x)
ax.set_xticklabels([f"{ds}\n(N={n})" for ds, n in zip(datasets, Ns)], fontsize=9)
ax.set_ylabel("Instâncias removidas (%)", fontsize=11)
ax.set_xlabel("Dataset", fontsize=11)
ax.set_ylim(0, 35)
ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
ax.set_axisbelow(True)
ax.legend(fontsize=9)
ax.set_title("Redução de Instâncias por Dataset (%)", fontsize=12)
fig.tight_layout()
fig.savefig(FIGS_PATH / "fig6_reducao_pct.png", dpi=150)
print("Salvo: fig6_reducao_pct.png")
plt.close()

# ---------------------------------------------------------------------------
# Fig 7 — Tamanho do dataset após redução (valores absolutos)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(14, 5))

for mode, label in MODES.items():
    selected = [data.get((ds, mode), {}).get("selected", 0) for ds in datasets]
    original = [data.get((ds, "seq"), {}).get("N", 0) for ds in datasets]
    bars = ax.bar(x + offsets[mode], selected, width * 0.9,
                  label=label, color=COLORS[mode], alpha=0.85, zorder=3)
    for bar, v in zip(bars, selected):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10,
                str(v), ha="center", va="bottom", fontsize=6.5, color=COLORS[mode])

# Linha de N original
orig_vals = [data.get((ds, "seq"), {}).get("N", 0) for ds in datasets]
ax.plot(x, orig_vals, "s--", color="#888", linewidth=1.5, markersize=5,
        label="N original", zorder=4)
for xi, v in zip(x, orig_vals):
    ax.text(xi, v + 30, str(v), ha="center", va="bottom", fontsize=6.5, color="#888")

ax.set_xticks(x)
ax.set_xticklabels(datasets, fontsize=9)
ax.set_ylabel("Instâncias selecionadas", fontsize=11)
ax.set_xlabel("Dataset", fontsize=11)
ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
ax.set_axisbelow(True)
ax.legend(fontsize=9)
ax.set_title("Tamanho do Dataset Após Redução (instâncias absolutas)", fontsize=12)
fig.tight_layout()
fig.savefig(FIGS_PATH / "fig7_reducao_absoluta.png", dpi=150)
print("Salvo: fig7_reducao_absoluta.png")
plt.close()

# ---------------------------------------------------------------------------
# Fig 8 — F1-Score: Sequencial × OpenMP × CUDA
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(14, 5))

for mode, label in MODES.items():
    f1s = [data.get((ds, mode), {}).get("f1", 0) for ds in datasets]
    bars = ax.bar(x + offsets[mode], f1s, width * 0.9,
                  label=label, color=COLORS[mode], alpha=0.85, zorder=3)
    for bar, v in zip(bars, f1s):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                f"{v:.3f}", ha="center", va="bottom", fontsize=6.5, color=COLORS[mode])

ax.set_xticks(x)
ax.set_xticklabels([f"{ds}\n(N={n})" for ds, n in zip(datasets, Ns)], fontsize=9)
ax.set_ylabel("F1-Score", fontsize=11)
ax.set_xlabel("Dataset", fontsize=11)
ax.set_ylim(0.6, 1.05)
ax.axhline(0.9, linestyle="--", color="#aaa", linewidth=1, zorder=0)
ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
ax.set_axisbelow(True)
ax.legend(fontsize=9)
ax.set_title("F1-Score: Sequencial × OpenMP 12t × CUDA", fontsize=12)
fig.tight_layout()
fig.savefig(FIGS_PATH / "fig8_f1_paralelo.png", dpi=150)
print("Salvo: fig8_f1_paralelo.png")
plt.close()
