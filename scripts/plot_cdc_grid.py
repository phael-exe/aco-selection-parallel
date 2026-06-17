#!/usr/bin/env python3
"""Gera os gráficos do benchmark CDC (grade OpenMP threads×escalonador + CUDA block size).

Lê results/EP05_CDC_grid_results.csv e escreve PNGs em results/figs/cdc_*.png:
  - cdc_cuda_blocksize.png  : tempo + throughput (GFLOPS) por block size
  - cdc_omp_tempo.png       : tempo por nº de threads (static × dynamic)
  - cdc_omp_speedup.png     : escalabilidade dentro da leva (speedup vs 2 threads)
  - cdc_overview.png        : melhor CUDA × melhor OpenMP (tempo)

Usa só csv + matplotlib + numpy (sem pandas), no mesmo estilo de plot_instance_reduction.py.
"""
import argparse
import csv
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_CSV = ROOT / "results" / "EP05_CDC_grid_results.csv"
FIGS = ROOT / "results" / "figs"

C_CUDA = "#55A868"
C_STATIC = "#4C72B0"
C_DYN = "#DD8452"
C_GFLOPS = "#C44E52"
C_IDEAL = "#999999"


def load(csv_path):
    cuda, omp = [], []
    with open(csv_path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["status"] != "OK":
                continue
            if r["modo"] == "CUDA":
                cuda.append((int(r["block_size"]), float(r["tempo_aco_ms"]), float(r["gflops"])))
            elif r["modo"] == "OpenMP":
                omp.append((r["escalonador"], int(r["threads"]),
                            float(r["tempo_aco_ms"]), float(r["gflops"])))
    cuda.sort(key=lambda x: x[0])
    return cuda, omp


def fig_cuda(cuda):
    bs = [x[0] for x in cuda]
    t = [x[1] / 1000.0 for x in cuda]
    g = [x[2] for x in cuda]
    x = np.arange(len(bs))
    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    bars = ax1.bar(x, t, color=C_CUDA, alpha=0.85)
    ax1.set_xlabel("Block size (threads por bloco)")
    ax1.set_ylabel("Tempo ACO (s)", color=C_CUDA)
    ax1.set_xticks(x)
    ax1.set_xticklabels(bs)
    for b, v in zip(bars, t):
        ax1.text(b.get_x() + b.get_width() / 2, v, f"{v:.0f}s", ha="center", va="bottom", fontsize=9)
    ax2 = ax1.twinx()
    ax2.plot(x, g, "o-", color=C_GFLOPS, linewidth=2)
    ax2.set_ylabel("Throughput eval 1-NN (GFLOPS)", color=C_GFLOPS)
    ax1.set_title("CUDA — tempo e throughput por block size (CDC, N=253.680, 22 iters)")
    fig.tight_layout()
    fig.savefig(FIGS / "cdc_cuda_blocksize.png", dpi=130)
    plt.close(fig)


def _omp_by_sched(omp, sched):
    return sorted([(t, tm, g) for (s, t, tm, g) in omp if s == sched], key=lambda x: x[0])


def fig_omp_tempo(omp):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for sched, color in [("static", C_STATIC), ("dynamic", C_DYN)]:
        pts = _omp_by_sched(omp, sched)
        xs = [p[0] for p in pts]
        ys = [p[1] / 60000.0 for p in pts]
        ax.plot(xs, ys, "o-", color=color, linewidth=2, label=sched)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks([2, 4, 8, 16, 32])
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_xlabel("Threads OpenMP")
    ax.set_ylabel("Tempo ACO (min, escala log)")
    ax.set_title("OpenMP — tempo por nº de threads (CDC, 22 iters)")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGS / "cdc_omp_tempo.png", dpi=130)
    plt.close(fig)


def fig_omp_speedup(omp):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for sched, color in [("static", C_STATIC), ("dynamic", C_DYN)]:
        pts = _omp_by_sched(omp, sched)
        t2 = pts[0][1]
        xs = [p[0] for p in pts]
        ys = [t2 / p[1] for p in pts]
        ax.plot(xs, ys, "o-", color=color, linewidth=2, label=sched)
        for xv, yv in zip(xs, ys):
            ax.text(xv, yv, f"{yv:.1f}x", ha="center", va="bottom", fontsize=8)
    xs = [2, 4, 8, 16, 32]
    ax.plot(xs, [t / 2 for t in xs], "--", color=C_IDEAL, label="ideal (linear)")
    ax.set_xscale("log", base=2)
    ax.set_xticks(xs)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_xlabel("Threads OpenMP")
    ax.set_ylabel("Speedup (vs 2 threads, mesma leva)")
    ax.set_title("OpenMP — escalabilidade dentro da leva (CDC)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGS / "cdc_omp_speedup.png", dpi=130)
    plt.close(fig)


def fig_overview(cuda, omp):
    best_cuda = min(cuda, key=lambda x: x[1])
    labels = [f"CUDA\nbs={best_cuda[0]}"]
    vals = [best_cuda[1] / 1000.0]
    colors = [C_CUDA]
    for sched, color in [("static", C_STATIC), ("dynamic", C_DYN)]:
        t32 = [tm for (s, t, tm, g) in omp if s == sched and t == 32]
        if t32:
            labels.append(f"OpenMP 32t\n{sched}")
            vals.append(t32[0] / 1000.0)
            colors.append(color)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(labels, vals, color=colors, alpha=0.85)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.0f}s", ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("Tempo ACO (s)")
    ax.set_title("Melhor CUDA × melhor OpenMP (CDC, 22 iters)")
    fig.tight_layout()
    fig.savefig(FIGS / "cdc_overview.png", dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(DEFAULT_CSV))
    args = ap.parse_args()
    FIGS.mkdir(parents=True, exist_ok=True)
    cuda, omp = load(args.csv)
    if cuda:
        fig_cuda(cuda)
        if omp:
            fig_overview(cuda, omp)
    if omp:
        fig_omp_tempo(omp)
        fig_omp_speedup(omp)
    print(f"Figuras geradas em {FIGS}/cdc_*.png")


if __name__ == "__main__":
    main()
