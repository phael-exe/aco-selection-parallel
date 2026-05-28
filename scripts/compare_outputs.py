#!/usr/bin/env python3
import sys
import numpy as np


def load_csv(path):
    try:
        return np.loadtxt(path, delimiter=";")
    except ValueError:
        return np.loadtxt(path, delimiter=",")


def compare(file_a, file_b, atol=1e-6):
    a = load_csv(file_a)
    b = load_csv(file_b)

    if a.shape != b.shape:
        print(f"FAIL: shape mismatch {a.shape} vs {b.shape}", file=sys.stderr)
        return False

    if not np.allclose(a, b, atol=atol):
        diff = np.argwhere(~np.isclose(a, b, atol=atol))
        print(f"FAIL: {len(diff)} valores divergentes", file=sys.stderr)
        for idx in diff[:10]:
            key = tuple(idx)
            print(f"  {key}: {a[key]:.8f} vs {b[key]:.8f}", file=sys.stderr)
        return False

    print(f"OK: outputs equivalentes ({a.shape})")
    return True


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Uso: {sys.argv[0]} <arquivo_a> <arquivo_b>", file=sys.stderr)
        sys.exit(2)

    ok = compare(sys.argv[1], sys.argv[2])
    sys.exit(0 if ok else 1)
