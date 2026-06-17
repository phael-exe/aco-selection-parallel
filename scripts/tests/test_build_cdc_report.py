import os
import sys
import tempfile
import textwrap

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from build_cdc_report import extract_t1_per_iter, compute_speedup, build_report


def test_extract_t1_per_iter():
    sample = (
        "=== Resultado ACO (OpenMP) ===\n"
        "Iteracoes executadas: 100/100\n"
        "Tempo ACO: 2.6451e+07 ms\n"
        "Threads: 1\n"
    )
    f = tempfile.NamedTemporaryFile("w", suffix=".log", delete=False)
    f.write(sample)
    f.close()
    t1 = extract_t1_per_iter(f.name)
    os.unlink(f.name)
    # 2.6451e7 / 100 = 264510 ms/iter
    assert abs(t1 - 264510.0) < 1.0, t1


def test_compute_speedup():
    # Tp/iter = metade do T1/iter -> speedup 2x
    # 1322550/10 = 132255 = 264510/2
    s = compute_speedup(264510.0, tempo_aco_ms=1322550.0, iters=10)
    assert abs(s - 2.0) < 1e-6, s


def test_extract_t1_ignores_threads_1x():
    """Fix 1: 'Threads: 16' must NOT be selected as the 1-thread baseline."""
    # First block has Threads: 16 — should be ignored.
    # Second block has Threads: 1 — should be picked.
    sample = textwrap.dedent("""\
        === Resultado ACO (OpenMP) ===
        Iteracoes executadas: 50/100
        Tempo ACO: 9.9999e+08 ms
        Threads: 16

        === Resultado ACO (OpenMP) ===
        Iteracoes executadas: 100/100
        Tempo ACO: 2.6451e+07 ms
        Threads: 1
    """)
    f = tempfile.NamedTemporaryFile("w", suffix=".log", delete=False)
    f.write(sample)
    f.close()
    t1 = extract_t1_per_iter(f.name)
    os.unlink(f.name)
    # Should use the Threads:1 block: 2.6451e7 / 100 = 264510 ms/iter
    assert abs(t1 - 264510.0) < 1.0, (
        f"Expected ~264510.0 (Threads:1 block), got {t1} — "
        "Threads:16 block was incorrectly matched as 1-thread baseline"
    )


def test_report_dedups_erro_then_ok():
    """Fix 2: when a config appears as ERRO then OK, only the OK row is rendered."""
    # Build a minimal T1 log
    t1_log_content = textwrap.dedent("""\
        === Resultado ACO (OpenMP) ===
        Iteracoes executadas: 100/100
        Tempo ACO: 2.6451e+07 ms
        Threads: 1
    """)

    # CSV with duplicate CUDA block_size=64: first ERRO, then OK
    csv_content = textwrap.dedent("""\
        modo,threads,escalonador,block_size,tempo_aco_ms,tempo_eval_ms,gflops,iteracoes,status
        CUDA,,, 64,9999999.0,8888888.0,0.01,5,ERRO
        CUDA,,,64,1234567.0,111111.0,1.23,100,OK
    """)

    with tempfile.TemporaryDirectory() as tmpdir:
        t1_log = os.path.join(tmpdir, "run.log")
        csv_path = os.path.join(tmpdir, "results.csv")
        out_path = os.path.join(tmpdir, "report.md")

        with open(t1_log, "w") as fh:
            fh.write(t1_log_content)
        with open(csv_path, "w") as fh:
            fh.write(csv_content)

        build_report(csv_path, t1_log, out_path)

        with open(out_path, encoding="utf-8") as fh:
            md = fh.read()

    # The OK row's tempo should appear
    assert "1234567.0" in md, f"OK row tempo not found in report:\n{md}"
    # The ERRO row's distinctive tempo must NOT appear
    assert "9999999.0" not in md, f"ERRO row tempo unexpectedly present in report:\n{md}"
    # block_size=64 should appear exactly once as a table row
    count = md.count("| 64 ")
    assert count == 1, (
        f"Expected block_size=64 to appear exactly once in table, got {count} times:\n{md}"
    )


if __name__ == "__main__":
    test_extract_t1_per_iter()
    print("  test_extract_t1_per_iter ... OK")
    test_compute_speedup()
    print("  test_compute_speedup ... OK")
    test_extract_t1_ignores_threads_1x()
    print("  test_extract_t1_ignores_threads_1x ... OK")
    test_report_dedups_erro_then_ok()
    print("  test_report_dedups_erro_then_ok ... OK")
    print("OK: todos os testes passaram")
