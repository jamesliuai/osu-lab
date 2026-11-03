"""Position-only Section 21 priors fitted to the canonical B0 tanh-Normal.

Quadrature is deterministic float64 CPU work. No image, target, game metadata,
or training observation enters this calibration. This mathematical prior is not
the required rendered random-opportunity/occupancy probe.
"""
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
import math

import numpy as np

GRID = (16, 12)
LOG_STD_BOUNDS = (-7.0, 0.5)


def validate_grid(grid: tuple[int, int]) -> tuple[int, int]:
    if type(grid) is not tuple or len(grid) != 2 or any(type(n) is not int or not 1 <= n <= 64 for n in grid):
        raise ValueError("cell grid requires two integer dimensions in [1,64]")
    return grid


@dataclass(frozen=True)
class CellPrior:
    grid: tuple[int, int]
    log_masses: tuple[float, ...]
    means: tuple[tuple[float, float], ...]
    log_stds: tuple[tuple[float, float], ...]
    version: str
    quadrature_points: int
    residual_integration_limit: float
    maximum_mass_error: float
    maximum_moment_refinement_error: float

    def metadata(self) -> dict:
        return {"scheme": "canonical_b0_conditional_local_normal_v1", "version": self.version,
                "grid_wh": list(self.grid), "canonical_mean": 0.0, "canonical_log_std": -0.7,
                "quadrature": "Gauss-Legendre in local pre-tanh residual",
                "quadrature_points": self.quadrature_points,
                "residual_integration_limit": self.residual_integration_limit,
                "maximum_relative_mass_error": self.maximum_mass_error,
                "maximum_moment_refinement_error": self.maximum_moment_refinement_error,
                "rendered_exploration_calibrated": False}


def _axis(cells: int, points: int, limit: float):
    nodes, weights = np.polynomial.legendre.leggauss(points)
    residual, weights = nodes * limit, weights * limit
    centres = -1 + (2 * np.arange(cells, dtype=np.float64) + 1) / cells
    cursor = centres[:, None] + np.tanh(residual)[None, :] / cells
    if not (np.abs(cursor) < 1).all():
        raise ValueError("quadrature reached a nonfinite inverse-tanh endpoint")
    sigma = math.exp(-0.7)
    z = np.arctanh(cursor)
    log_pdf = -0.5 * (z / sigma) ** 2 - math.log(sigma * math.sqrt(2 * math.pi)) - np.log1p(-cursor * cursor)
    log_jacobian = 2 * (math.log(2) - residual - np.logaddexp(0, -2 * residual)) - math.log(cells)
    weighted = np.exp(log_pdf + log_jacobian) * weights
    measured_mass = weighted.sum(-1)
    boundaries = np.linspace(-1, 1, cells + 1)
    cdf = np.array([0.0 if x == -1 else 1.0 if x == 1 else
                    0.5 * math.erfc(-math.atanh(float(x)) / (sigma * math.sqrt(2))) for x in boundaries])
    mass = np.diff(cdf)
    if not (mass > 0).all():
        raise ValueError("canonical cell mass underflowed")
    mean = (weighted * residual).sum(-1) / measured_mass
    variance = (weighted * (residual - mean[:, None]) ** 2).sum(-1) / measured_mass
    if not (variance > 0).all() or not np.isfinite(variance).all():
        raise ValueError("invalid fitted residual variance")
    return mass, mean, variance, float(np.max(np.abs(measured_mass / mass - 1)))


@lru_cache(maxsize=8)
def calibrated_prior(grid: tuple[int, int] = GRID, *, quadrature_points: int = 256) -> CellPrior:
    """Fit conditional residual moments; verify by doubling quadrature order.

    Integration truncation is explicitly recorded at |u|=16. Endpoint limits
    for analytic canonical cell masses are evaluated at infinity, not clipped.
    The refined moments are used; the coarse pass checks numerical convergence.
    """
    gx, gy = validate_grid(grid)
    if type(quadrature_points) is not int or not 128 <= quadrature_points <= 1024:
        raise ValueError("quadrature order must be an integer in [128,1024]")
    limit, fits, errors, refinements = 16.0, [], [], []
    for cells in (gx, gy):
        coarse = _axis(cells, quadrature_points, limit)
        fitted = _axis(cells, quadrature_points * 2, limit)
        refinements.append(max(float(np.max(np.abs(coarse[i] - fitted[i]))) for i in (1, 2)))
        fits.append(fitted)
        errors.append(fitted[3])
    if max(errors) > 1e-8 or max(refinements) > 1e-6:
        raise ValueError("conditional-prior quadrature did not converge; increase the declared order")
    mx, ux, vx, _ = fits[0]
    my, uy, vy, _ = fits[1]
    masses = np.outer(my, mx).reshape(-1)
    means = tuple((float(ux[x]), float(uy[y])) for y in range(gy) for x in range(gx))
    scales = tuple((float(np.log(vx[x]) / 2), float(np.log(vy[y]) / 2)) for y in range(gy) for x in range(gx))
    if any(not LOG_STD_BOUNDS[0] <= value <= LOG_STD_BOUNDS[1] for pair in scales for value in pair):
        raise ValueError("calibrated residual scale exceeds the declared shared bounds")
    log_masses = tuple(float(x) for x in np.log(masses))
    payload = {"scheme": "canonical_b0_conditional_local_normal_v1", "grid": grid,
               "log_masses": log_masses, "means": means, "log_stds": scales,
               "quadrature_points": quadrature_points * 2, "limit": limit}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    return CellPrior(grid, log_masses, means, scales, digest, quadrature_points * 2, limit,
                     max(errors), max(refinements))
