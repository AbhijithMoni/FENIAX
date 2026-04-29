"""Utility functions for coupled FSI simulations using FENIAX.

Provides displacement extraction helpers that correctly handle the FENIAX
``ra`` output array for both dynamic and static solutions.

Notes
-----
FENIAX stores node positions in ``ra`` arrays with the following shapes:

* **Dynamic** (``DynamicSystem``): ``(Nt, 3, num_nodes)`` where ``Nt`` is the
  number of saved time steps.
* **Static** (``StaticSystem``): ``(tn, 3, num_nodes)`` where ``tn`` is the
  number of load steps, or ``(3, num_nodes)`` for a single-step result.

All public functions accept either shape variant and return displacements as
``numpy`` arrays with shape ``(num_nodes, 3)`` (final step) or
``(Nt, num_nodes, 3)`` (full history).
"""

from __future__ import annotations

import pathlib

import numpy as np


def _find_ra_file(sol_path: pathlib.Path) -> pathlib.Path:
    """Locate ``ra.npy`` within a FENIAX solution directory.

    Searches recursively for ``ra.npy`` files.  When both ``DynamicSystem``
    and ``StaticSystem`` variants exist the ``DynamicSystem`` file is
    preferred.

    Parameters
    ----------
    sol_path : pathlib.Path
        Root of the FENIAX solution directory (value of
        ``config.driver.sol_path``).

    Returns
    -------
    pathlib.Path
        Path to the located ``ra.npy`` file.

    Raises
    ------
    FileNotFoundError
        When no ``ra.npy`` can be found under *sol_path*.
    """
    ra_files = list(sol_path.glob("**/ra.npy"))

    if not ra_files:
        available = [str(f.relative_to(sol_path)) for f in sol_path.glob("**/*.npy")]
        raise FileNotFoundError(
            f"No ra.npy found under '{sol_path}'.\n"
            f"Available .npy files: {available}"
        )

    # Prefer DynamicSystem over StaticSystem when both are present.
    dynamic_ra = [f for f in ra_files if "DynamicSystem" in f.parts]
    if dynamic_ra:
        return dynamic_ra[0]

    return ra_files[0]


def _validate_ra_shape(
    ra: np.ndarray,
    expected_nodes: int,
    solution_type: str,
) -> None:
    """Validate the shape of a FENIAX ``ra`` array.

    Parameters
    ----------
    ra : np.ndarray
        Array loaded from ``ra.npy``.
    expected_nodes : int
        Number of FEM nodes expected (``undeformed_fem_coords.shape[0]``).
    solution_type : str
        Either ``"dynamic"`` or ``"static"``.

    Raises
    ------
    ValueError
        When *ra* does not conform to the expected shape conventions.
    """
    if ra.ndim not in (2, 3):
        raise ValueError(
            f"Expected ra.ndim to be 2 or 3, got {ra.ndim} (shape={ra.shape})."
        )

    # For both dynamic and static, the last two axes are (3, num_nodes).
    coords_axis = ra.shape[-2]
    nodes_axis = ra.shape[-1]

    if coords_axis != 3:
        raise ValueError(
            f"Expected 3 spatial coordinates per node (second-to-last axis), "
            f"got {coords_axis} (shape={ra.shape})."
        )

    if nodes_axis != expected_nodes:
        raise ValueError(
            f"Node count mismatch for '{solution_type}' solution: "
            f"expected {expected_nodes} nodes, ra contains {nodes_axis} nodes "
            f"(shape={ra.shape})."
        )

    if ra.ndim == 3:
        nt = ra.shape[0]
        print(
            f"[FENIAX] {solution_type} solution validated: "
            f"{nt} time/load steps, {nodes_axis} nodes."
        )
    else:
        print(
            f"[FENIAX] {solution_type} solution validated: "
            f"single step, {nodes_axis} nodes."
        )


def extract_final_displacement(
    ra: np.ndarray,
    undeformed_fem_coords: np.ndarray,
) -> np.ndarray:
    """Extract the displacement field at the final time/load step.

    Correctly handles FENIAX ``ra`` arrays for both dynamic
    (``(Nt, 3, num_nodes)``) and static (``(tn, 3, num_nodes)`` or
    ``(3, num_nodes)``) solutions.

    Parameters
    ----------
    ra : np.ndarray
        Position array from the FENIAX solution.
    undeformed_fem_coords : np.ndarray
        Undeformed node coordinates with shape ``(num_nodes, 3)``.

    Returns
    -------
    np.ndarray
        Displacement array with shape ``(num_nodes, 3)``.

    Raises
    ------
    ValueError
        When *ra* has an unexpected number of dimensions or when the
        resulting deformed-coordinate shape does not match
        *undeformed_fem_coords*.
    """
    if ra.ndim == 3:
        # (Nt, 3, num_nodes) → take last step → (3, num_nodes) → (num_nodes, 3)
        deformed_coords = np.asarray(ra[-1]).T
    elif ra.ndim == 2:
        # (3, num_nodes) → (num_nodes, 3)
        deformed_coords = np.asarray(ra).T
    else:
        raise ValueError(
            f"Unexpected ra.ndim={ra.ndim} (shape={ra.shape}). "
            "Expected 2 (static single-step) or 3 (multi-step / dynamic)."
        )

    if deformed_coords.shape != undeformed_fem_coords.shape:
        raise ValueError(
            f"Shape mismatch: deformed_coords {deformed_coords.shape} vs "
            f"undeformed_fem_coords {undeformed_fem_coords.shape}."
        )

    displacements = deformed_coords - undeformed_fem_coords

    if np.any(np.isnan(displacements)) or np.any(np.isinf(displacements)):
        print(
            "[FENIAX] WARNING: NaN/Inf values detected in displacements; "
            "replacing with zeros."
        )
        displacements = np.nan_to_num(displacements, nan=0.0, posinf=0.0, neginf=0.0)

    return displacements


def extract_full_displacement_history(
    ra: np.ndarray,
    undeformed_fem_coords: np.ndarray,
) -> np.ndarray:
    """Extract displacement history for all time/load steps.

    Parameters
    ----------
    ra : np.ndarray
        Position array from the FENIAX solution.
    undeformed_fem_coords : np.ndarray
        Undeformed node coordinates with shape ``(num_nodes, 3)``.

    Returns
    -------
    np.ndarray
        Displacement history with shape ``(Nt, num_nodes, 3)``, where
        ``Nt`` is 1 for a single-step (2-D *ra*) solution.

    Raises
    ------
    ValueError
        When *ra* has an unexpected number of dimensions.
    """
    ra_np = np.asarray(ra)

    if ra_np.ndim == 2:
        # (3, num_nodes) → (1, num_nodes, 3)
        deformed_all = ra_np.T[np.newaxis, :, :]  # (1, num_nodes, 3)
    elif ra_np.ndim == 3:
        # (Nt, 3, num_nodes) → (Nt, num_nodes, 3)
        deformed_all = ra_np.transpose(0, 2, 1)  # (Nt, num_nodes, 3)
    else:
        raise ValueError(
            f"Unexpected ra.ndim={ra_np.ndim} (shape={ra_np.shape}). "
            "Expected 2 or 3."
        )

    displacements_history = deformed_all - undeformed_fem_coords[np.newaxis, :, :]

    if np.any(np.isnan(displacements_history)) or np.any(np.isinf(displacements_history)):
        print(
            "[FENIAX] WARNING: NaN/Inf values detected in displacement history; "
            "replacing with zeros."
        )
        displacements_history = np.nan_to_num(
            displacements_history, nan=0.0, posinf=0.0, neginf=0.0
        )

    return displacements_history
