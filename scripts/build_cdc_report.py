#!/usr/bin/env python3
"""Gera o relatório da grade CDC a partir do CSV do runner + baseline T1 do log.

Uso:
    python3 scripts/build_cdc_report.py \
        --csv results/cdc_grid_results.csv \
        --t1-log results_remote/hpc_raw_runs.log \
        --out results/cdc_grid_report.md
"""
import argparse
import csv
import re


def extract_t1_per_iter(log_path):
    """Tempo (ms) por iteração do baseline 1-thread.

    Procura o primeiro bloco '=== Resultado ACO' que contém 'Threads: 1' e
    divide o 'Tempo ACO' pelo nº de 'Iteracoes executadas'. O 1-thread é
    independente do escalonador (~1%), então serve de T1 para todos.
    """
    with open(log_path, encoding="utf-8", errors="ignore") as fh:
        text = fh.read()
    for block in re.split(r"=== Resultado ACO", text):
        if "Threads: 1" not in block:
            continue
        m_t = re.search(r"Tempo ACO:\s*([0-9.eE+]+)\s*ms", block)
        m_i = re.search(r"Iteracoes executadas:\s*(\d+)", block)
        if m_t and m_i and int(m_i.group(1)) > 0:
            return float(m_t.group(1)) / int(m_i.group(1))
    raise ValueError("Nenhum bloco 1-thread ('Threads: 1') encontrado em " + log_path)


def compute_speedup(t1_per_iter, tempo_aco_ms, iters):
    """Speedup normalizado por iteração: (T1/iter) / (Tp/iter)."""
    tempo_aco_ms = float(tempo_aco_ms)
    iters = int(iters)
    if iters <= 0 or tempo_aco_ms <= 0:
        return 0.0
    return t1_per_iter / (tempo_aco_ms / iters)


def build_report(csv_path, t1_log, out_path):
    t1 = extract_t1_per_iter(t1_log)
    with open(csv_path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    cuda = [r for r in rows if r["modo"] == "CUDA"]
    omp = [r for r in rows if r["modo"] == "OpenMP"]

    lines = []
    lines.append("# Relatório — Grade CDC (OpenMP threads×escalonador + CUDA)\n")
    lines.append(f"**Baseline T1 (1 thread):** {t1:,.1f} ms/iteração "
                 f"(de `{t1_log}`)\n")
    lines.append("> Speedup = (T1/iteração) / (Tp/iteração) — normalizado por "
                 "iteração por causa do early stop (K varia). Ressalva: alguns "
                 "laços do ACO têm escalonador fixo; o laço quente (eval 1-NN) "
                 "respeita `OMP_SCHEDULE`.\n")

    lines.append("\n## CUDA (por block size)\n")
    lines.append("| Block size | Tempo ACO (ms) | Eval (ms) | GFLOPS | Speedup | Iter | Status |")
    lines.append("| ---: | ---: | ---: | ---: | ---: | ---: | :--- |")
    for r in cuda:
        sp = compute_speedup(t1, r["tempo_aco_ms"], r["iteracoes"])
        lines.append(f"| {r['block_size']} | {r['tempo_aco_ms']} | {r['tempo_eval_ms']} "
                     f"| {r['gflops']} | {sp:.2f}x | {r['iteracoes']} | {r['status']} |")

    lines.append("\n## OpenMP (threads × escalonador)\n")
    lines.append("| Escalonador | Threads | Tempo ACO (ms) | Eval (ms) | GFLOPS | Speedup | Iter | Status |")
    lines.append("| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |")
    for r in sorted(omp, key=lambda x: (x["escalonador"], int(x["threads"] or 0))):
        sp = compute_speedup(t1, r["tempo_aco_ms"], r["iteracoes"])
        lines.append(f"| {r['escalonador']} | {r['threads']} | {r['tempo_aco_ms']} "
                     f"| {r['tempo_eval_ms']} | {r['gflops']} | {sp:.2f}x "
                     f"| {r['iteracoes']} | {r['status']} |")

    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"Relatório escrito em {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/cdc_grid_results.csv")
    ap.add_argument("--t1-log", default="results_remote/hpc_raw_runs.log")
    ap.add_argument("--out", default="results/cdc_grid_report.md")
    args = ap.parse_args()
    build_report(args.csv, args.t1_log, args.out)


if __name__ == "__main__":
    main()
