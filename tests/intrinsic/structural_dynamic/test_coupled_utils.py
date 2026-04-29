"""Tests for coupled simulation utility functions.

Validates displacement extraction helpers in
``feniax.simulations.coupled_utils`` against a variety of ``ra`` array
shapes that FENIAX produces.
"""

import pathlib

import numpy as np
import pytest

from feniax.simulations.coupled_utils import (
    _find_ra_file,
    _validate_ra_shape,
    extract_final_displacement,
    extract_full_displacement_history,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def undeformed_coords():
    """Simple 5-node undeformed coordinate array."""
    rng = np.random.default_rng(42)
    return rng.random((5, 3))


@pytest.fixture
def dynamic_ra(undeformed_coords):
    """Simulate a dynamic ``ra`` array: shape (Nt, 3, num_nodes)."""
    num_nodes = undeformed_coords.shape[0]
    nt = 10
    rng = np.random.default_rng(0)
    # Deformed positions slightly different from undeformed
    base = undeformed_coords.T  # (3, num_nodes)
    perturbation = rng.random((nt, 3, num_nodes)) * 0.01
    ra = base[np.newaxis, :, :] + perturbation
    return ra


@pytest.fixture
def static_ra_3d(undeformed_coords):
    """Simulate a static ``ra`` array with multiple load steps: (tn, 3, N)."""
    num_nodes = undeformed_coords.shape[0]
    tn = 3
    rng = np.random.default_rng(1)
    base = undeformed_coords.T
    perturbation = rng.random((tn, 3, num_nodes)) * 0.005
    ra = base[np.newaxis, :, :] + perturbation
    return ra


@pytest.fixture
def static_ra_2d(undeformed_coords):
    """Simulate a static ``ra`` array for a single-step result: (3, N)."""
    num_nodes = undeformed_coords.shape[0]
    rng = np.random.default_rng(2)
    base = undeformed_coords.T  # (3, num_nodes)
    perturbation = rng.random((3, num_nodes)) * 0.005
    return base + perturbation


# ---------------------------------------------------------------------------
# _find_ra_file
# ---------------------------------------------------------------------------


class TestFindRaFile:
    def test_finds_dynamic_ra(self, tmp_path):
        dynamic_dir = tmp_path / "DynamicSystem_s1"
        dynamic_dir.mkdir()
        ra_file = dynamic_dir / "ra.npy"
        ra_file.touch()

        result = _find_ra_file(tmp_path)
        assert result == ra_file

    def test_finds_static_ra(self, tmp_path):
        static_dir = tmp_path / "StaticSystem_s1"
        static_dir.mkdir()
        ra_file = static_dir / "ra.npy"
        ra_file.touch()

        result = _find_ra_file(tmp_path)
        assert result == ra_file

    def test_prefers_dynamic_over_static(self, tmp_path):
        static_dir = tmp_path / "StaticSystem_s1"
        static_dir.mkdir()
        (static_dir / "ra.npy").touch()

        dynamic_dir = tmp_path / "DynamicSystem_s1"
        dynamic_dir.mkdir()
        dynamic_ra = dynamic_dir / "ra.npy"
        dynamic_ra.touch()

        result = _find_ra_file(tmp_path)
        assert result == dynamic_ra

    def test_raises_when_missing(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="No ra.npy found"):
            _find_ra_file(tmp_path)

    def test_finds_nested_ra(self, tmp_path):
        nested = tmp_path / "level1" / "DynamicSystem_s1"
        nested.mkdir(parents=True)
        ra_file = nested / "ra.npy"
        ra_file.touch()

        result = _find_ra_file(tmp_path)
        assert result == ra_file


# ---------------------------------------------------------------------------
# _validate_ra_shape
# ---------------------------------------------------------------------------


class TestValidateRaShape:
    def test_valid_dynamic_3d(self, dynamic_ra, undeformed_coords):
        # Should not raise
        _validate_ra_shape(dynamic_ra, undeformed_coords.shape[0], "dynamic")

    def test_valid_static_3d(self, static_ra_3d, undeformed_coords):
        _validate_ra_shape(static_ra_3d, undeformed_coords.shape[0], "static")

    def test_valid_static_2d(self, static_ra_2d, undeformed_coords):
        _validate_ra_shape(static_ra_2d, undeformed_coords.shape[0], "static")

    def test_raises_on_wrong_node_count(self, dynamic_ra):
        with pytest.raises(ValueError, match="Node count mismatch"):
            _validate_ra_shape(dynamic_ra, expected_nodes=999, solution_type="dynamic")

    def test_raises_on_wrong_coord_axis(self, undeformed_coords):
        # Shape (10, 4, 5) – 4 coordinates instead of 3
        bad_ra = np.zeros((10, 4, undeformed_coords.shape[0]))
        with pytest.raises(ValueError, match="3 spatial coordinates"):
            _validate_ra_shape(bad_ra, undeformed_coords.shape[0], "dynamic")

    def test_raises_on_bad_ndim(self, undeformed_coords):
        bad_ra = np.zeros((10, 3, 5, undeformed_coords.shape[0]))  # 4-D
        with pytest.raises(ValueError, match="ra.ndim"):
            _validate_ra_shape(bad_ra, undeformed_coords.shape[0], "dynamic")


# ---------------------------------------------------------------------------
# extract_final_displacement
# ---------------------------------------------------------------------------


class TestExtractFinalDisplacement:
    def test_dynamic_returns_last_step(self, dynamic_ra, undeformed_coords):
        disp = extract_final_displacement(dynamic_ra, undeformed_coords)
        assert disp.shape == undeformed_coords.shape

        # Expected: last time step transposed minus undeformed
        expected = dynamic_ra[-1].T - undeformed_coords
        np.testing.assert_allclose(disp, expected)

    def test_static_3d_returns_last_step(self, static_ra_3d, undeformed_coords):
        disp = extract_final_displacement(static_ra_3d, undeformed_coords)
        assert disp.shape == undeformed_coords.shape

        expected = static_ra_3d[-1].T - undeformed_coords
        np.testing.assert_allclose(disp, expected)

    def test_static_2d_returns_displacement(self, static_ra_2d, undeformed_coords):
        disp = extract_final_displacement(static_ra_2d, undeformed_coords)
        assert disp.shape == undeformed_coords.shape

        expected = static_ra_2d.T - undeformed_coords
        np.testing.assert_allclose(disp, expected)

    def test_zero_displacement_when_undeformed(self, undeformed_coords):
        # ra = undeformed coords transposed → displacement should be zero
        ra = undeformed_coords.T[np.newaxis, :, :]  # (1, 3, N)
        disp = extract_final_displacement(ra, undeformed_coords)
        np.testing.assert_allclose(disp, np.zeros_like(undeformed_coords), atol=1e-14)

    def test_raises_on_bad_ndim(self, undeformed_coords):
        bad_ra = np.zeros((2, 3, 5, undeformed_coords.shape[0]))
        with pytest.raises(ValueError, match="Unexpected ra.ndim"):
            extract_final_displacement(bad_ra, undeformed_coords)

    def test_raises_on_shape_mismatch(self):
        ra = np.zeros((10, 3, 5))  # 5 nodes
        undeformed = np.zeros((7, 3))  # 7 nodes – mismatch
        with pytest.raises(ValueError, match="Shape mismatch"):
            extract_final_displacement(ra, undeformed)

    def test_nan_replaced_with_zero(self, undeformed_coords):
        ra = undeformed_coords.T[np.newaxis, :, :].copy()
        ra[0, 0, 0] = np.nan
        disp = extract_final_displacement(ra, undeformed_coords)
        assert not np.any(np.isnan(disp))


# ---------------------------------------------------------------------------
# extract_full_displacement_history
# ---------------------------------------------------------------------------


class TestExtractFullDisplacementHistory:
    def test_dynamic_shape(self, dynamic_ra, undeformed_coords):
        hist = extract_full_displacement_history(dynamic_ra, undeformed_coords)
        nt = dynamic_ra.shape[0]
        assert hist.shape == (nt, undeformed_coords.shape[0], 3)

    def test_static_3d_shape(self, static_ra_3d, undeformed_coords):
        hist = extract_full_displacement_history(static_ra_3d, undeformed_coords)
        tn = static_ra_3d.shape[0]
        assert hist.shape == (tn, undeformed_coords.shape[0], 3)

    def test_static_2d_shape(self, static_ra_2d, undeformed_coords):
        hist = extract_full_displacement_history(static_ra_2d, undeformed_coords)
        assert hist.shape == (1, undeformed_coords.shape[0], 3)

    def test_values_dynamic(self, dynamic_ra, undeformed_coords):
        hist = extract_full_displacement_history(dynamic_ra, undeformed_coords)
        # Manually compute expected
        expected = dynamic_ra.transpose(0, 2, 1) - undeformed_coords[np.newaxis, :, :]
        np.testing.assert_allclose(hist, expected)

    def test_values_static_2d(self, static_ra_2d, undeformed_coords):
        hist = extract_full_displacement_history(static_ra_2d, undeformed_coords)
        expected = (static_ra_2d.T - undeformed_coords)[np.newaxis, :, :]
        np.testing.assert_allclose(hist, expected)

    def test_final_step_consistent_with_extract_final(self, dynamic_ra, undeformed_coords):
        """Last step of history must equal extract_final_displacement result."""
        hist = extract_full_displacement_history(dynamic_ra, undeformed_coords)
        final = extract_final_displacement(dynamic_ra, undeformed_coords)
        np.testing.assert_allclose(hist[-1], final)

    def test_raises_on_bad_ndim(self, undeformed_coords):
        bad_ra = np.zeros((2, 3, 5, undeformed_coords.shape[0]))
        with pytest.raises(ValueError, match="Unexpected ra.ndim"):
            extract_full_displacement_history(bad_ra, undeformed_coords)

    def test_nan_replaced_with_zero(self, dynamic_ra, undeformed_coords):
        ra = dynamic_ra.copy()
        ra[0, 0, 0] = np.nan
        hist = extract_full_displacement_history(ra, undeformed_coords)
        assert not np.any(np.isnan(hist))
