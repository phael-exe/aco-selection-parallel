import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from compare_outputs import compare


def _write(path, data, delimiter=";"):
    np.savetxt(path, data, delimiter=delimiter, fmt="%g")


def test_identical_files_pass(tmp_path):
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    data = np.array([[1, 0, 1], [0, 1, 0]])
    _write(a, data)
    _write(b, data)
    assert compare(str(a), str(b)) is True


def test_different_values_fail(tmp_path):
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    _write(a, np.array([[1, 0, 1]]))
    _write(b, np.array([[0, 1, 0]]))
    assert compare(str(a), str(b)) is False


def test_shape_mismatch_fail(tmp_path):
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    _write(a, np.array([[1, 0]]))
    _write(b, np.array([[1, 0, 1]]))
    assert compare(str(a), str(b)) is False


def test_within_tolerance_pass(tmp_path):
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    data = np.array([[1.0, 0.0, 1.0]])
    _write(a, data)
    _write(b, data + 1e-9)  # within atol=1e-6
    assert compare(str(a), str(b)) is True


def test_outside_tolerance_fail(tmp_path):
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    data = np.array([[1.0, 0.0]])
    _write(a, data)
    _write(b, data + 1e-4)  # outside atol=1e-6
    assert compare(str(a), str(b)) is False


def test_comma_delimiter(tmp_path):
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    data = np.array([[1, 0, 1]])
    _write(a, data, delimiter=",")
    _write(b, data, delimiter=",")
    assert compare(str(a), str(b)) is True
