#!/usr/bin/env python3
"""Gráficos do benchmark padronizado dos 9 datasets (results/baseline_sched_benchmark.csv):
  - baseline_speedup_vs_n.png : speedup (vs seq) de CUDA e OpenMP por tamanho do dataset (crossover)
  - baseline_sched.png        : tempo por escalonador (16 threads) nos datasets >= 1000 linhas
"""
import csv
import pathlib
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
CSV = ROOT / "results" / "baseline_sched_benchmark.csv"
FIGS = ROOT / "results" / "figs"
C_CUDA = "#55A868"; C_OMP = "#4C72B0"; C_STATIC = "#4C72B0"; C_DYN = "#DD8452"

rows = list(csv.DictReader(open(CSV, encoding="utf-8")))
ds_order, seen = [], set()
for r in rows:
    if r["dataset"] not in seen:
        ds_order.append(r["dataset"]); seen.add(r["dataset"])


def get(ds, mode, **kw):
    for r in rows:
        if r["dataset"] == ds and r["mode"] == mode and all(r.get(k) == v for k, v in kw.items()):
            return r
    return None


def fig_speedup():
    Ns, sp_cuda, sp_omp = [], [], []
    for ds in ds_order:
        seq = get(ds, "seq"); cu = get(ds, "cuda")
        st = float(seq["tempo_ms"]); ct = float(cu["tempo_ms"])
        omps = [r for r in rows if r["dataset"] == ds and r["mode"] == "omp" and r["threads"] != "1"]
        bt = min(float(r["tempo_ms"]) for r in omps)
        Ns.append(int(seq["N"])); sp_cuda.append(st / ct); sp_omp.append(st / bt)
    order = np.argsort(Ns)
    Ns = np.array(Ns)[order]; sp_cuda = np.array(sp_cuda)[order]; sp_omp = np.array(sp_omp)[order]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(Ns, sp_omp, "o-", color=C_OMP, linewidth=2, label="OpenMP (melhor config)")
    ax.plot(Ns, sp_cuda, "s-", color=C_CUDA, linewidth=2, label="CUDA (GPU)")
    ax.axhline(1.0, color="#999999", ls="--", label="sequencial (baseline)")
    ax.set_xscale("log")
    ax.set_xlabel("Tamanho do dataset (N, escala log)")
    ax.set_ylabel("Speedup vs sequencial")
    ax.set_title("Speedup por tamanho do dataset (9 bases, 100 iter, i9-14900K + RTX 4090)")
    ax.legend(); ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout(); fig.savefig(FIGS / "baseline_speedup_vs_n.png", dpi=130); plt.close(fig)


def fig_sched():
    scheds = ["static", "static-2", "static-3", "dynamic", "dynamic-2", "dynamic-3"]
    big = [d for d in ds_order if int(get(d, "seq")["N"]) >= 1000]
    # tempo normalizado pelo 'static' de cada dataset (16 threads), media geometrica
    norm = []
    for s in scheds:
        ratios = []
        for d in big:
            r = get(d, "omp", threads="16", schedule=s)
            base = get(d, "omp", threads="16", schedule="static")
            if r and base:
                ratios.append(float(r["tempo_ms"]) / float(base["tempo_ms"]))
        norm.append(np.exp(np.mean(np.log(ratios))) if ratios else np.nan)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = [C_STATIC if s.startswith("static") else C_DYN for s in scheds]
    bars = ax.bar(scheds, norm, color=colors, alpha=0.85)
    ax.axhline(1.0, color="#999999", ls="--")
    for b, v in zip(bars, norm):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Tempo relativo ao static (16 threads)\n(menor = melhor, média geom. 4 bases)")
    ax.set_title("Escalonadores OpenMP — bases ≥ 1000 linhas (16 threads)")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.tight_layout(); fig.savefig(FIGS / "baseline_sched.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    FIGS.mkdir(parents=True, exist_ok=True)
    fig_speedup()
    fig_sched()
    print("Figuras: baseline_speedup_vs_n.png, baseline_sched.png")
