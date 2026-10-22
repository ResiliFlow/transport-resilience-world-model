from __future__ import annotations

"""Deterministic numerical kernel for regional Dynamic Propagation Connectivity.

The formulation follows Xu et al. (2026), arXiv:2606.30242.  The paper
defines an infinite propagation-time average but does not publish a finite
quadrature contract.  This module therefore exposes every numerical choice
used by the platform and returns an approximation diagnostic.  It never
labels the result as quantum-hardware output.
"""

from dataclasses import dataclass
import math
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class PropagationEdge:
    """One directed coupling and the candidate region containing the link."""

    from_node: str
    to_node: str
    strength: float
    region_id: str


@dataclass(frozen=True)
class DPCNumericalContract:
    degradation_alpha: float = 0.5
    scaled_horizon: float = 30.0
    sample_count: int = 33
    dense_node_limit: int = 80
    krylov_dimension: int = 28

    def validate(self) -> None:
        if not 0.0 <= self.degradation_alpha <= 1.0:
            raise ValueError("DPC regional degradation alpha must be in [0, 1].")
        if not math.isfinite(self.scaled_horizon) or self.scaled_horizon <= 0:
            raise ValueError("DPC scaled propagation horizon must be positive.")
        if self.sample_count < 3:
            raise ValueError("DPC quadrature needs at least three propagation-time samples.")
        if self.dense_node_limit < 2 or self.krylov_dimension < 2:
            raise ValueError("DPC numerical dimensions must be at least two.")


def _operator_arrays(
    edges: Sequence[PropagationEdge],
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    nodes = sorted(
        {str(edge.from_node) for edge in edges} | {str(edge.to_node) for edge in edges},
        key=str,
    )
    if len(nodes) < 2:
        raise ValueError("DPC requires at least two topology nodes.")
    node_index = {node: index for index, node in enumerate(nodes)}
    rows: list[int] = []
    columns: list[int] = []
    weights: list[float] = []
    edge_regions: list[str] = []
    for edge in edges:
        strength = float(edge.strength)
        if not math.isfinite(strength) or strength < 0:
            raise ValueError("Every DPC propagation strength must be finite and non-negative.")
        source = node_index[str(edge.from_node)]
        target = node_index[str(edge.to_node)]
        # H_raw[source, target] = strength.  H=(H_raw+H_raw^T)/2.
        rows.extend((source, target))
        columns.extend((target, source))
        weights.extend((0.5 * strength, 0.5 * strength))
        edge_regions.extend((str(edge.region_id), str(edge.region_id)))
    return (
        nodes,
        np.asarray(rows, dtype=np.int64),
        np.asarray(columns, dtype=np.int64),
        np.asarray(weights, dtype=np.float64),
        np.asarray(edge_regions, dtype=object),
    )


def _matvec(
    node_count: int,
    rows: np.ndarray,
    columns: np.ndarray,
    weights: np.ndarray,
    vector: np.ndarray,
) -> np.ndarray:
    output = np.zeros(node_count, dtype=np.result_type(vector.dtype, np.float64))
    np.add.at(output, rows, weights * vector[columns])
    return output


def _dense_matrix(
    node_count: int,
    rows: np.ndarray,
    columns: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    matrix = np.zeros((node_count, node_count), dtype=np.float64)
    np.add.at(matrix, (rows, columns), weights)
    # The construction is Hermitian analytically.  This guard removes only
    # floating-point accumulation noise.
    return 0.5 * (matrix + matrix.T)


def _participation_average(
    eigenvalues: np.ndarray,
    basis: np.ndarray,
    coefficients: np.ndarray,
    *,
    scaled_horizon: float,
    sample_count: int,
) -> tuple[float, float]:
    spectral_scale = float(max(np.ptp(eigenvalues), np.max(np.abs(eigenvalues)), 1e-12))
    scaled_times = np.linspace(0.0, scaled_horizon, sample_count, dtype=np.float64)
    propagation_times = scaled_times / spectral_scale
    phases = np.exp(-1j * eigenvalues[:, None] * propagation_times[None, :])
    states = basis @ (coefficients[:, None] * phases)
    probabilities = np.abs(states) ** 2
    totals = probabilities.sum(axis=0)
    probabilities /= np.maximum(totals, 1e-15)[None, :]
    participation = 1.0 / np.maximum((probabilities**2).sum(axis=0), 1e-15)
    average = float(np.trapezoid(participation, scaled_times) / scaled_horizon)
    return average, spectral_scale


def _dense_dpc(
    node_count: int,
    rows: np.ndarray,
    columns: np.ndarray,
    weights: np.ndarray,
    contract: DPCNumericalContract,
) -> tuple[float, dict[str, float | int | str]]:
    matrix = _dense_matrix(node_count, rows, columns, weights)
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    initial_state = np.full(node_count, 1.0 / math.sqrt(node_count), dtype=np.float64)
    coefficients = eigenvectors.T @ initial_state
    value, spectral_scale = _participation_average(
        eigenvalues,
        eigenvectors,
        coefficients,
        scaled_horizon=contract.scaled_horizon,
        sample_count=contract.sample_count,
    )
    return value, {
        "method": "dense Hermitian eigendecomposition with deterministic trapezoidal quadrature",
        "basis_dimension": node_count,
        "spectral_scale": spectral_scale,
        "maximum_krylov_residual": 0.0,
    }


def _lanczos_basis(
    node_count: int,
    rows: np.ndarray,
    columns: np.ndarray,
    weights: np.ndarray,
    maximum_dimension: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    target_dimension = min(node_count, maximum_dimension)
    initial_state = np.full(node_count, 1.0 / math.sqrt(node_count), dtype=np.float64)
    basis_vectors: list[np.ndarray] = []
    diagonal: list[float] = []
    off_diagonal: list[float] = []
    previous = np.zeros(node_count, dtype=np.float64)
    current = initial_state
    previous_beta = 0.0
    final_beta = 0.0

    for index in range(target_dimension):
        basis_vectors.append(current.copy())
        image = _matvec(node_count, rows, columns, weights, current)
        alpha = float(np.dot(current, image))
        diagonal.append(alpha)
        residual = image - alpha * current - previous_beta * previous
        # Full deterministic re-orthogonalisation keeps the compact basis
        # stable for networks with clustered or repeated eigenvalues.
        for vector in basis_vectors:
            residual -= float(np.dot(vector, residual)) * vector
        beta = float(np.linalg.norm(residual))
        final_beta = beta
        if beta <= 1e-12 or index + 1 >= target_dimension:
            break
        off_diagonal.append(beta)
        previous, current = current, residual / beta
        previous_beta = beta

    basis = np.column_stack(basis_vectors)
    dimension = basis.shape[1]
    tridiagonal = np.diag(np.asarray(diagonal[:dimension], dtype=np.float64))
    if dimension > 1:
        beta_values = np.asarray(off_diagonal[: dimension - 1], dtype=np.float64)
        tridiagonal += np.diag(beta_values, 1) + np.diag(beta_values, -1)
    return basis, tridiagonal, final_beta


def _lanczos_dpc(
    node_count: int,
    rows: np.ndarray,
    columns: np.ndarray,
    weights: np.ndarray,
    contract: DPCNumericalContract,
) -> tuple[float, dict[str, float | int | str]]:
    basis, reduced, final_beta = _lanczos_basis(
        node_count,
        rows,
        columns,
        weights,
        contract.krylov_dimension,
    )
    eigenvalues, eigenvectors = np.linalg.eigh(reduced)
    # The first Lanczos vector is the normalised initial state, so its reduced
    # coordinate is e_1.
    coefficients = eigenvectors[0, :]
    value, spectral_scale = _participation_average(
        eigenvalues,
        basis @ eigenvectors,
        coefficients,
        scaled_horizon=contract.scaled_horizon,
        sample_count=contract.sample_count,
    )
    residual_ratio = float(final_beta / max(spectral_scale, 1e-12))
    return value, {
        "method": "Lanczos projection with deterministic trapezoidal quadrature",
        "basis_dimension": int(basis.shape[1]),
        "spectral_scale": spectral_scale,
        "maximum_krylov_residual": residual_ratio,
    }


def propagation_connectivity(
    edges: Sequence[PropagationEdge],
    *,
    contract: DPCNumericalContract | None = None,
) -> tuple[float, dict[str, float | int | str]]:
    """Approximate the paper's propagation-time averaged participation index."""

    selected_contract = contract or DPCNumericalContract()
    selected_contract.validate()
    nodes, rows, columns, weights, _regions = _operator_arrays(edges)
    if not np.any(weights > 0):
        raise ValueError("DPC requires at least one positive propagation coupling.")
    if len(nodes) <= selected_contract.dense_node_limit:
        return _dense_dpc(len(nodes), rows, columns, weights, selected_contract)
    return _lanczos_dpc(len(nodes), rows, columns, weights, selected_contract)


def regional_dpc_impacts(
    edges: Sequence[PropagationEdge],
    candidate_regions: Iterable[str],
    *,
    contract: DPCNumericalContract | None = None,
) -> dict[str, object]:
    """Evaluate Eq. (20) for each candidate region under one observed state."""

    selected_contract = contract or DPCNumericalContract()
    selected_contract.validate()
    nodes, rows, columns, base_weights, edge_regions = _operator_arrays(edges)
    if not np.any(base_weights > 0):
        raise ValueError("DPC requires at least one positive propagation coupling.")

    def evaluate(weights: np.ndarray) -> tuple[float, dict[str, float | int | str]]:
        if len(nodes) <= selected_contract.dense_node_limit:
            return _dense_dpc(len(nodes), rows, columns, weights, selected_contract)
        return _lanczos_dpc(len(nodes), rows, columns, weights, selected_contract)

    baseline, baseline_diagnostics = evaluate(base_weights)
    if baseline <= 0 or not math.isfinite(baseline):
        raise ValueError("The observed propagation operator did not produce a positive finite DPC baseline.")

    impacts: list[dict[str, object]] = []
    maximum_residual = float(baseline_diagnostics["maximum_krylov_residual"])
    for region_id in sorted({str(value) for value in candidate_regions}, key=str):
        degraded_weights = base_weights.copy()
        degraded_weights[edge_regions == region_id] *= 1.0 - selected_contract.degradation_alpha
        degraded, diagnostics = evaluate(degraded_weights)
        signed_impact = (baseline - degraded) / baseline
        score = max(0.0, float(signed_impact))
        maximum_residual = max(
            maximum_residual,
            float(diagnostics["maximum_krylov_residual"]),
        )
        impacts.append(
            {
                "region_id": region_id,
                "baseline_dpc": float(baseline),
                "degraded_dpc": float(degraded),
                "signed_dpc_impact": float(signed_impact),
                "regional_dpc_score": score,
                "numerical_method": diagnostics["method"],
                "basis_dimension": diagnostics["basis_dimension"],
            }
        )

    impacts.sort(
        key=lambda item: (
            -float(item["regional_dpc_score"]),
            -float(item["signed_dpc_impact"]),
            str(item["region_id"]),
        )
    )
    return {
        "baseline_dpc": float(baseline),
        "impacts": impacts,
        "diagnostics": {
            **baseline_diagnostics,
            "node_count": len(nodes),
            "directed_link_count": len(edges),
            "candidate_region_count": len(impacts),
            "degradation_alpha": selected_contract.degradation_alpha,
            "scaled_horizon": selected_contract.scaled_horizon,
            "sample_count": selected_contract.sample_count,
            "requested_krylov_dimension": selected_contract.krylov_dimension,
            "maximum_krylov_residual": maximum_residual,
            "residual_interpretation": (
                "Lanczos basis-boundary coupling divided by the reduced spectral scale; "
                "it is visible as a projection diagnostic and is not a certified DPC error bound."
            ),
            "paper_long_time_limit_exactly_evaluated": False,
            "finite_time_approximation": (
                "Deterministic trapezoidal average over the disclosed scaled horizon and samples; "
                "the paper does not publish a finite quadrature contract for Equation (9)."
            ),
            "hardware": "classical CPU numerical linear algebra",
        },
    }
