import numpy as np


def _lsq_initial_estimate(distances, antenna_positions):
    """
    Linear Least Squares (slides 16-19).
    Uses B1 as reference point and forms the linear system Ax ≈ b.
    """
    n = len(distances)
    r = distances
    B = antenna_positions

    # A: (n-1) x 3, b: (n-1) x 1
    A = np.zeros((n - 1, 3))
    b = np.zeros(n - 1)

    for i in range(1, n):
        A[i - 1] = B[i] - B[0]
        d_i1 = np.linalg.norm(B[i] - B[0])
        b[i - 1] = 0.5 * (r[0]**2 - r[i]**2 + d_i1**2)

    # Solve Ax = b via least squares (using QR internally)
    x_rel, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

    # x_rel = (x - x1, y - y1, z - z1), so add back B1
    return x_rel + B[0]


def _gauss_newton_step(theta, distances, antenna_positions):
    """
    One iteration of the Gauss-Newton method (slide 25).
    Returns the updated theta.
    """
    n = len(distances)
    r = distances
    B = antenna_positions

    # f(theta): n x 1, residuals d_i(theta) - r_i
    d = np.linalg.norm(B - theta, axis=1)  # true distances from current estimate
    f = d - r

    # J(theta): n x 3, Jacobian of d_i w.r.t. (x, y, z)
    # partial d_i / partial x_j = (x_j - xi_j) / d_i
    J = (theta - B) / d[:, np.newaxis]

    # theta_{k+1} = theta_k - (J^T J)^{-1} J^T f
    JtJ = J.T @ J
    Jtf = J.T @ f
    delta, _, _, _ = np.linalg.lstsq(JtJ, Jtf, rcond=None)

    return theta - delta


def trilaterate(distances, antenna_positions, max_iter=20, tol=1e-8):
    """
    Estimate target position via trilateration (Hereman, CSM).

    1) LSQ (exact linearization) for initial guess
    2) NLSQ Gauss-Newton iteration to refine

    Parameters
    ----------
    distances : array-like, shape (n,)
        Measured distances from the target to each antenna.
    antenna_positions : array-like, shape (n, 3)
        Known (x, y, z) coordinates of each antenna.
    max_iter : int
        Maximum number of Gauss-Newton iterations.
    tol : float
        Convergence tolerance on ||theta_{k+1} - theta_k||.

    Returns
    -------
    theta : np.ndarray, shape (3,)
        Estimated (x, y, z) position of the target.
    """
    distances = np.asarray(distances, dtype=float)
    antenna_positions = np.asarray(antenna_positions, dtype=float)

    # Step 1: LSQ initial estimate
    theta = _lsq_initial_estimate(distances, antenna_positions)

    # Step 1b: If antennas are (near-)coplanar, LSQ cannot determine the
    # out-of-plane coordinate.  Estimate it from distance residuals:
    #   r_i^2 = d_xy_i^2 + (z - z_i)^2
    # where d_xy_i is the in-plane distance from the LSQ solution to antenna i.
    # With z_i ~ z_mean for all i:  z_offset^2 ≈ median(r_i^2 - d_xy_i^2)
    z_spread = np.ptp(antenna_positions[:, 2])  # max - min
    if z_spread < 0.1:  # near-coplanar threshold (meters)
        z_plane = antenna_positions[:, 2].mean()
        d_xy_sq = np.sum((theta[:2] - antenna_positions[:, :2])**2, axis=1)
        z_offset_sq = np.median(distances**2 - d_xy_sq)
        if z_offset_sq > 0:
            z_offset = np.sqrt(z_offset_sq)
            # Try both sides; pick the one closer to any prior z info,
            # defaulting to below the antenna plane (target_z < antenna_z).
            z_below = z_plane - z_offset
            z_above = z_plane + z_offset
            if abs(theta[2] - z_plane) > 0.01:
                # LSQ gave some z hint (slightly non-coplanar) — trust its side
                theta[2] = z_below if theta[2] < z_plane else z_above
            else:
                # No z info at all — use prior: target is below antennas
                theta[2] = z_below

    # Step 2: Gauss-Newton refinement
    for _ in range(max_iter):
        theta_new = _gauss_newton_step(theta, distances, antenna_positions)
        if np.linalg.norm(theta_new - theta) < tol:
            break
        theta = theta_new

    return theta