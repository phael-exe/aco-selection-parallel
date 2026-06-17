import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from build_cdc_report import extract_t1_per_iter, compute_speedup


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


if __name__ == "__main__":
    test_extract_t1_per_iter()
    test_compute_speedup()
    print("OK: todos os testes passaram")
