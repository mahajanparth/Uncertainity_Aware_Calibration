from typing import List, Optional, Tuple
import numpy as np
from scipy.optimize import linprog, minimize_scalar
import gtsam
import scipy
from enum import Enum
import matplotlib.pyplot as plt
from functools import partial
from gtsam.utils import plot
# from Experiments import exp_utils
from scipy.optimize import minimize, Bounds, LinearConstraint
from numpy import linalg as la
from scipy.sparse import csr_matrix, identity
from scipy.sparse.linalg import eigsh, spsolve, inv
from joblib import Parallel, delayed
import scipy.sparse as sp
from functools import partial
import time
try:
    from gurobipy import Model, GRB, quicksum
except ImportError:  # pragma: no cover - optional dependency
    Model = None
    GRB = None
    quicksum = None

from . import utilities
from . import FIM as infmat

L = gtsam.symbol_shorthand.L
X = gtsam.symbol_shorthand.X

class Metric(Enum):
    LOGDET = 1
    MIN_EIG = 2
    MSE = 3
#Test with dense matrices
def compute_schur_complement_d(x, inf_mats, H0, num_poses):
    """
    Computes the Schur complement and auxiliary matrices for use in the objective and gradient computations.

    Args:
        x (np.ndarray): Continuous selection vector.
        inf_mats (List[scipy.sparse.csr_matrix]): List of sparse information matrices.
        H0 (scipy.sparse.csr_matrix): Prior information matrix.
        num_poses (int): Number of poses.

    Returns:
        Tuple[scipy.sparse.csc_matrix, np.ndarray, scipy.sparse.csc_matrix, scipy.sparse.csc_matrix]:
        - H_schur: The Schur complement matrix.
        - min_eig_vec: Eigenvector corresponding to the smallest eigenvalue of H_schur.
        - T1: Dense intermediate matrix Hxl * inv(Hll)
        - T2: Dense intermediate matrix from solving Hll * T2 = Hlx.
    """
    # Combine the Fisher Information Matrices
    # s= time.time()
    #print("in schur complement")
    combined_fim = H0.copy()
    for xi, Hi in zip(x, inf_mats):
        combined_fim += xi * Hi

    # Extract submatrices
    pose_dim = 6
    num_pose_elements = num_poses * pose_dim
    measurement_dim = combined_fim.shape[0] - num_pose_elements
    Hll = combined_fim[:measurement_dim, :measurement_dim]
    Hlx = combined_fim[:measurement_dim, measurement_dim:]
    Hxx = combined_fim[measurement_dim:, measurement_dim:]

    # s1 = time.time()
    try:
        # # Convert Hll to dense if it is sparse
        if scipy.sparse.issparse(Hll):
            Hll = Hll.toarray()
        # Regularize and compute the pseudoinverse of Hll
        reg_term = 1e-8 * np.eye(Hll.shape[0])

        Hll_inv = np.linalg.pinv(Hll + reg_term)
    except Exception as e:
        raise ValueError(f"Failed to compute pseudoinverse for candidate: {e}")

    # Compute the Schur complement
    H_schur = Hxx - Hlx.T @ Hll_inv @ Hlx

    # Ensure H_schur is symmetric
    H_schur = (H_schur + H_schur.T) / 2
    T2 = Hll_inv @ Hlx
    T1 = Hlx.transpose().dot(Hll_inv)

    # e1 = time.time()
    # execution_time = e1 - s1
    # print(f"execution time dense schur: {execution_time:.4f} seconds")

    # # Compute smallest eigenvalue and eigenvector
    # try:
    #     v , w = scipy.linalg.eigh(H_schur)
    #     min_eig_vec = w[:,0:1]
    # except Exception as e:
    #     print("Eigenvalue solver failed:", e)
    #     min_eig_vec = np.zeros(H_schur.shape[1])

    return H_schur, T1, T2, measurement_dim

def compute_schur_complement(x, inf_mats, H0, num_poses):
    """
    Computes the Schur complement and auxiliary matrices for use in the objective and gradient computations.

    Args:
        x (np.ndarray): Continuous selection vector.
        inf_mats (List[scipy.sparse.csr_matrix]): List of sparse information matrices.
        H0 (scipy.sparse.csr_matrix): Prior information matrix.
        num_poses (int): Number of poses.

    Returns:
        Tuple[scipy.sparse.csc_matrix, np.ndarray, scipy.sparse.csc_matrix, scipy.sparse.csc_matrix]:
        - H_schur: The Schur complement matrix.
        - min_eig_vec: Eigenvector corresponding to the smallest eigenvalue of H_schur.
        - T1: Dense intermediate matrix Hxl * inv(Hll)
        - T2: Dense intermediate matrix from solving Hll * T2 = Hlx.
    """
    # Combine the Fisher Information Matrices
    combined_fim = H0.copy()
    for xi, Hi in zip(x, inf_mats):
        combined_fim += xi * Hi

    # Extract submatrices
    pose_dim = 6
    num_pose_elements = num_poses * pose_dim
    measurement_dim = combined_fim.shape[0] - num_pose_elements
    Hll = combined_fim[:measurement_dim, :measurement_dim].tocsc()
    Hlx = combined_fim[:measurement_dim, measurement_dim:].tocsc()
    Hxx = combined_fim[measurement_dim:, measurement_dim:].tocsc()

    try:
        Hll_inv = scipy.sparse.linalg.inv(Hll.tocsc())
        T2 = Hll_inv.dot(Hlx)
        T1 = Hlx.transpose().dot(Hll_inv)
    except Exception as e:
        print("Linear solver failed:", e)
        T2 = Hll.inverse().dot(Hlx).toarray()

    H_schur = Hxx - Hlx.transpose().dot(T2)

    try:
        min_eig_vals, min_eig_vec = eigsh(H_schur, k=1, which='SA')
        min_eig_val = float(min_eig_vals[0])
    except Exception as e:
        print("Eigenvalue solver failed:", e)
        min_eig_val = 0.0
        min_eig_vec = np.zeros((H_schur.shape[1], 1))

    return H_schur, min_eig_val, min_eig_vec, T1, T2, measurement_dim

def compute_combined_fim(x, inf_mats, H0):
    combined_fim = H0.copy()
    for xi, Hi in zip(x, inf_mats):
        combined_fim += xi * Hi
    return combined_fim    

def compute_calibration_schur_sparse(x, inf_mats, H0, num_poses):
    """Calibration Schur complement: marginalizes the block-diagonal pose block,
    leaving a small (intrinsics_dim × intrinsics_dim) H_cal matrix.

    H_cal = H_tt - H_tn @ inv(H_nn) @ H_nt

    H_nn is block diagonal (one 6×6 block per pose), so its inverse is computed
    by inverting each 6×6 block independently — no large matrix inversion needed.
    """
    pose_dim = 6
    combined_fim = H0.copy()
    for xi, Hi in zip(x, inf_mats):
        combined_fim += xi * Hi
    combined_fim = combined_fim.tocsc()

    intr_dim = combined_fim.shape[0] - num_poses * pose_dim

    H_tt = np.asarray(combined_fim[:intr_dim, :intr_dim].todense())        # (intr_dim × intr_dim)
    H_tn = np.asarray(combined_fim[:intr_dim, intr_dim:].todense())        # (intr_dim × num_poses*6)
    H_nn = combined_fim[intr_dim:, intr_dim:]                              # (num_poses*6 × num_poses*6)

    # Invert H_nn block by block: each pose contributes an independent 6×6 block
    reg = 1e-9 * np.eye(pose_dim)
    T2_cal = np.zeros((num_poses * pose_dim, intr_dim), dtype=float)       # H_nn_inv @ H_nt
    for i in range(num_poses):
        s, e = i * pose_dim, (i + 1) * pose_dim
        block = np.asarray(H_nn[s:e, s:e].todense())
        block_inv = np.linalg.pinv(block + reg)
        T2_cal[s:e, :] = block_inv @ H_tn[:, s:e].T                       # (6×9)

    T1_cal = T2_cal.T                                                       # (intr_dim × num_poses*6)
    H_cal = H_tt - H_tn @ T2_cal                                           # (intr_dim × intr_dim)
    H_cal = 0.5 * (H_cal + H_cal.T)

    eigvals, eigvecs = np.linalg.eigh(H_cal)
    min_eig_val = float(eigvals[0])
    min_eig_vec = eigvecs[:, 0:1]

    return H_cal, min_eig_val, min_eig_vec, T1_cal, T2_cal, intr_dim


def compute_grad_calibration_parallel(idx, Hi, T1_cal, T2_cal, min_eig_vec, intr_dim, pose_dim):
    """Gradient of lambda_min(H_cal) w.r.t. selection weight x_i.

    d H_cal / dx_i = h_tt_i - h_tn_i @ T2_cal - T1_cal @ h_nt_i + T1_cal @ h_nn_i @ T2_cal

    Only the slices of T1_cal/T2_cal for pose i's columns/rows are needed,
    keeping all operations small.
    """
    Hi_csc = Hi.tocsc()
    h_tt_i = np.asarray(Hi_csc[:intr_dim, :intr_dim].todense())            # (intr_dim × intr_dim)
    s, e = intr_dim + idx * pose_dim, intr_dim + (idx + 1) * pose_dim
    h_tn_i = np.asarray(Hi_csc[:intr_dim, s:e].todense())                  # (intr_dim × 6)
    h_nn_i = np.asarray(Hi_csc[s:e, s:e].todense())                        # (6 × 6)

    T1_slice = T1_cal[:, idx * pose_dim:(idx + 1) * pose_dim]              # (intr_dim × 6)
    T2_slice = T2_cal[idx * pose_dim:(idx + 1) * pose_dim, :]              # (6 × intr_dim)

    grad_cal = h_tt_i - h_tn_i @ T2_slice - T1_slice @ h_tn_i.T + T1_slice @ h_nn_i @ T2_slice
    v = min_eig_vec.flatten()
    return idx, -float(v @ grad_cal @ v)


def precompute_fim_blocks(inf_mats, H0, num_poses, intr_dim):
    """Extract small dense blocks from all sparse FIMs once before the FW loop.

    Returns pre-stacked numpy arrays so each FW iteration only touches small dense matrices.
    """
    pose_dim = 6
    H0_csc = H0.tocsc()

    H0_tt = np.asarray(H0_csc[:intr_dim, :intr_dim].todense())  # (intr_dim, intr_dim)
    # H0_tn as (N, intr_dim, pose_dim) — one (9,6) block per pose
    H0_tn_raw = np.asarray(H0_csc[:intr_dim, intr_dim:].todense())  # (intr_dim, N*6)
    H0_tn_blocks = H0_tn_raw.reshape(intr_dim, num_poses, pose_dim).transpose(1, 0, 2)  # (N, intr_dim, 6)
    H0_nn_blocks = np.stack([
        np.asarray(H0_csc[
            intr_dim + i * pose_dim: intr_dim + (i + 1) * pose_dim,
            intr_dim + i * pose_dim: intr_dim + (i + 1) * pose_dim,
        ].todense())
        for i in range(num_poses)
    ])  # (N, 6, 6)

    all_h_tt = np.zeros((num_poses, intr_dim, intr_dim), dtype=float)
    all_h_tn = np.zeros((num_poses, intr_dim, pose_dim), dtype=float)
    all_h_nn = np.zeros((num_poses, pose_dim, pose_dim), dtype=float)
    for i, Hi in enumerate(inf_mats):
        Hi_csc = Hi.tocsc()
        all_h_tt[i] = np.asarray(Hi_csc[:intr_dim, :intr_dim].todense())
        s, e = intr_dim + i * pose_dim, intr_dim + (i + 1) * pose_dim
        all_h_tn[i] = np.asarray(Hi_csc[:intr_dim, s:e].todense())
        all_h_nn[i] = np.asarray(Hi_csc[s:e, s:e].todense())

    return H0_tt, H0_tn_blocks, H0_nn_blocks, all_h_tt, all_h_tn, all_h_nn


def compute_calibration_schur_vectorized(x, H0_tt, H0_tn_blocks, H0_nn_blocks, all_h_tt, all_h_tn, all_h_nn):
    """Vectorized calibration Schur complement — no sparse ops, no Python loops.

    H_cal = H_tt - sum_i H_tn_i @ H_nn_i^{-1} @ H_nt_i
    All blocks are pre-extracted dense arrays; this runs entirely in numpy.
    """
    pose_dim = 6

    # H_tt: (intr_dim, intr_dim)
    H_tt = H0_tt + np.einsum('n,njk->jk', x, all_h_tt)

    # H_tn_blocks: (N, intr_dim, 6)
    H_tn_blocks = H0_tn_blocks + all_h_tn * x[:, None, None]

    # H_nn_blocks: (N, 6, 6)
    H_nn_blocks = H0_nn_blocks + all_h_nn * x[:, None, None]

    # Invert each 6×6 block: (N, 6, 6)
    reg = 1e-9 * np.eye(pose_dim)[None, :, :]
    H_nn_inv = np.linalg.pinv(H_nn_blocks + reg)  # batch pinv

    # T2_blocks[i] = H_nn_inv[i] @ H_tn_blocks[i].T : (N, 6, intr_dim)
    T2_blocks = np.einsum('nij,nkj->nik', H_nn_inv, H_tn_blocks)

    # Schur correction: sum_i H_tn_i @ T2_i → (intr_dim, intr_dim)
    schur_contrib = np.einsum('nij,njk->ik', H_tn_blocks, T2_blocks)

    H_cal = H_tt - schur_contrib
    H_cal = 0.5 * (H_cal + H_cal.T)

    eigvals, eigvecs = np.linalg.eigh(H_cal)
    return H_cal, float(eigvals[0]), eigvecs[:, 0:1], T2_blocks, H_tn_blocks


def compute_grad_vectorized(v, all_h_tt, all_h_tn, all_h_nn, T2_blocks, H_tn_blocks):
    """Vectorized gradient of lambda_min(H_cal) w.r.t. selection x.

    grad[i] = -v.T @ (d H_cal/dx_i) @ v
            = -(t1 - 2*t2 + t4)
    No Python loops, no joblib — pure numpy einsum.
    """
    v = v.flatten()

    # t1 = v.T @ h_tt_i @ v  for each i
    t1 = np.einsum('j,njk,k->n', v, all_h_tt, v)

    # t2 = v.T @ h_tn_i @ T2_blocks[i] @ v  (symmetric with t3)
    vh_tn = np.einsum('j,njk->nk', v, all_h_tn)       # (N, 6)
    T2v = np.einsum('njk,k->nj', T2_blocks, v)         # (N, 6)
    t2 = np.einsum('nk,nk->n', vh_tn, T2v)             # (N,)

    # t4 = T2v[i].T @ h_nn_i @ T2v[i]
    t4 = np.einsum('nj,njk,nk->n', T2v, all_h_nn, T2v)  # (N,)

    return -(t1 - 2.0 * t2 + t4)


def compute_grad_a_optimal_vectorized(H_cal, all_h_tt, all_h_tn, all_h_nn, T2_blocks):
    """Vectorized gradient of -trace(H_cal^{-1}) w.r.t. selection x.

    For A-optimality, the objective is f(u) = -trace(H_cal^{-1}(u)).
    Gradient: grad_i = -trace(H_cal^{-2} @ G_i)  where G_i = d H_cal / d u_i.

    Same structure as compute_grad_vectorized but replaces v⊗v with H_cal^{-2}.
    grad[i] = -(t1 - 2*t2 + t4)  with P2 = H_cal^{-1} @ H_cal^{-1} replacing v*v^T.
    """
    reg = 1e-12 * np.eye(H_cal.shape[0])
    P = np.linalg.pinv(H_cal + reg)   # H_cal^{-1},  (d, d)
    P2 = P @ P                         # H_cal^{-2},  (d, d)

    # t1[i] = trace(P2 @ h_tt_i) = einsum('jk,njk->n', P2, all_h_tt)
    t1 = np.einsum('jk,njk->n', P2, all_h_tt)

    # t2[i] = trace(P2 @ h_tn_i @ T2_i)
    #       = trace(T2_i @ P2 @ h_tn_i)  [cyclic]
    #       = einsum('nkj,jl,nlk->n', T2_blocks, P2, all_h_tn)
    T2P2 = np.einsum('nkj,jl->nkl', T2_blocks, P2)           # (N, 6, d)
    t2 = np.einsum('nkl,nlk->n', T2P2, all_h_tn)              # (N,)

    # t4[i] = trace(P2 @ T2_i.T @ h_nn_i @ T2_i)
    #       = trace((T2_i @ P2 @ T2_i.T) @ h_nn_i)  [cyclic]
    T2P2T2T = np.einsum('nkl,njl->nkj', T2P2, T2_blocks)     # (N, 6, 6) = T2@P2@T2^T
    t4 = np.einsum('nkj,nkj->n', T2P2T2T, all_h_nn)          # (N,)

    return -(t1 - 2.0 * t2 + t4)


def compute_grad_d_optimal_vectorized(H_cal, all_h_tt, all_h_tn, all_h_nn, T2_blocks):
    """Vectorized gradient of log det(H_cal) w.r.t. selection x.

    For D-optimality, the objective is f(u) = log det(H_cal(u)).
    Gradient: grad_i = trace(H_cal^{-1} @ G_i)  where G_i = d H_cal / d u_i.

    Same structure as compute_grad_a_optimal_vectorized but using P = H_cal^{-1}
    instead of P^2 = H_cal^{-2}.  Returns the *negative* gradient (convention: LMO
    picks most-negative entries to maximise the objective).
    """
    reg = 1e-12 * np.eye(H_cal.shape[0])
    P = np.linalg.pinv(H_cal + reg)   # H_cal^{-1},  (d, d)

    # t1[i] = trace(P @ h_tt_i)
    t1 = np.einsum('jk,njk->n', P, all_h_tt)

    # t2[i] = trace(P @ h_tn_i @ T2_i) = trace(T2_i @ P @ h_tn_i)  [cyclic]
    T2P = np.einsum('nkj,jl->nkl', T2_blocks, P)             # (N, 6, d)
    t2 = np.einsum('nkl,nlk->n', T2P, all_h_tn)               # (N,)

    # t4[i] = trace(P @ T2_i.T @ h_nn_i @ T2_i)
    #       = trace((T2_i @ P @ T2_i.T) @ h_nn_i)  [cyclic]
    T2PT2T = np.einsum('nkl,njl->nkj', T2P, T2_blocks)       # (N, 6, 6) = T2@P@T2^T
    t4 = np.einsum('nkj,nkj->n', T2PT2T, all_h_nn)            # (N,)

    return -(t1 - 2.0 * t2 + t4)


def frank_wolfe_optimization_d_optimal(
    inf_mats: List[csr_matrix],
    H0: csr_matrix,
    selection_init: np.ndarray,
    num_poses: int,
    A: np.ndarray,
    b: np.ndarray,
) -> Tuple[np.ndarray, float, int]:
    """Frank-Wolfe continuous relaxation of the D-optimal calibration design problem.

    Maximises f(u) = log det(H_cal(u))  subject to sum(u) <= K, u in [0,1]^N.
    Uses the same vectorised Schur machinery as FW-E / FW-A but with the D-optimal gradient.
    """
    H0 = H0.tocsc()
    inf_mats = [Hi.tocsc() for Hi in inf_mats]
    A = sp.csc_matrix(A)
    pose_dim = 6
    intr_dim = H0.shape[0] - num_poses * pose_dim

    print(f"[FW-D] Pre-extracting dense blocks (intr_dim={intr_dim}, num_poses={num_poses})...", flush=True)
    H0_tt, H0_tn_blocks, H0_nn_blocks, all_h_tt, all_h_tn, all_h_nn = \
        precompute_fim_blocks(inf_mats, H0, num_poses, intr_dim)
    print(f"[FW-D] Block extraction done. Starting iterations...", flush=True)

    selection_cur = selection_init.copy().astype(float)
    prev_obj = -np.inf
    d_obj = prev_obj
    fw_gap_final = float("inf")

    for iteration in range(300):
        try:
            H_cal, _, _, T2_blocks, _ = compute_calibration_schur_vectorized(
                selection_cur, H0_tt, H0_tn_blocks, H0_nn_blocks, all_h_tt, all_h_tn, all_h_nn
            )
        except Exception as e:
            print(f"[FW-D] Schur error at iter {iteration}: {e}")
            break

        reg = 1e-12 * np.eye(H_cal.shape[0])
        sign, logabsdet = np.linalg.slogdet(H_cal + reg)
        d_obj = float(logabsdet) if sign > 0 else -np.inf

        rel_change = abs(d_obj - prev_obj) / max(abs(prev_obj), 1.0)
        prev_obj = d_obj

        print(f"[FW-D] iter={iteration:3d}  logdet={d_obj:.6e}", flush=True)

        grad = compute_grad_d_optimal_vectorized(H_cal, all_h_tt, all_h_tn, all_h_nn, T2_blocks)

        s = solve_lmo(grad, A, b)
        if s is None:
            break

        fw_gap_final = float(grad @ (selection_cur - s))

        if rel_change < 1e-4 and fw_gap_final < 1e-4:
            print(f"[FW-D] Converged at iter {iteration}  logdet={d_obj:.6e}  gap={fw_gap_final:.2e}")
            break

        alpha = 2.0 / (iteration + 2)
        selection_cur = selection_cur + alpha * (s - selection_cur)
        selection_cur = np.clip(selection_cur, 0.0, 1.0)

    return selection_cur, d_obj, iteration + 1, fw_gap_final


def frank_wolfe_optimization_a_optimal(
    inf_mats: List[csr_matrix],
    H0: csr_matrix,
    selection_init: np.ndarray,
    num_poses: int,
    A: np.ndarray,
    b: np.ndarray,
) -> Tuple[np.ndarray, float, int]:
    """Frank-Wolfe continuous relaxation of the A-optimal calibration design problem.

    Maximises f(u) = -trace(H_cal^{-1}(u))  subject to sum(u) <= K, u in [0,1]^N.
    Uses the same vectorised Schur machinery as FW-E but with the A-optimal gradient.
    """
    H0 = H0.tocsc()
    inf_mats = [Hi.tocsc() for Hi in inf_mats]
    A = sp.csc_matrix(A)
    pose_dim = 6
    intr_dim = H0.shape[0] - num_poses * pose_dim

    print(f"[FW-A] Pre-extracting dense blocks (intr_dim={intr_dim}, num_poses={num_poses})...", flush=True)
    H0_tt, H0_tn_blocks, H0_nn_blocks, all_h_tt, all_h_tn, all_h_nn = \
        precompute_fim_blocks(inf_mats, H0, num_poses, intr_dim)
    print(f"[FW-A] Block extraction done. Starting iterations...", flush=True)

    selection_cur = selection_init.copy().astype(float)
    prev_obj = np.inf
    a_obj = prev_obj
    fw_gap_final = float("inf")

    for iteration in range(300):
        try:
            H_cal, _, _, T2_blocks, _ = compute_calibration_schur_vectorized(
                selection_cur, H0_tt, H0_tn_blocks, H0_nn_blocks, all_h_tt, all_h_tn, all_h_nn
            )
        except Exception as e:
            print(f"[FW-A] Schur error at iter {iteration}: {e}")
            break

        reg = 1e-12 * np.eye(H_cal.shape[0])
        P = np.linalg.pinv(H_cal + reg)
        a_obj = -float(np.trace(P))   # f(u) = -trace(H_cal^{-1}), negative = worse

        rel_change = abs(a_obj - prev_obj) / max(abs(prev_obj), 1.0)
        prev_obj = a_obj

        print(f"[FW-A] iter={iteration:3d}  -trace(H^{{-1}})={a_obj:.6e}", flush=True)

        grad = compute_grad_a_optimal_vectorized(H_cal, all_h_tt, all_h_tn, all_h_nn, T2_blocks)

        s = solve_lmo(grad, A, b)
        if s is None:
            break

        fw_gap_final = float(grad @ (selection_cur - s))

        if rel_change < 1e-4 and fw_gap_final < 1e-4:
            print(f"[FW-A] Converged at iter {iteration}  -trace(H^{{-1}})={a_obj:.6e}  gap={fw_gap_final:.2e}")
            break

        alpha = 2.0 / (iteration + 2)
        selection_cur = selection_cur + alpha * (s - selection_cur)
        selection_cur = np.clip(selection_cur, 0.0, 1.0)

    return selection_cur, a_obj, iteration + 1, fw_gap_final


def greedy_selection(
    inf_mats: List[np.ndarray],
    prior: np.ndarray,
    Nc: int,
    metric: Metric = Metric.MIN_EIG,
    num_runs: int = 1,
    num_poses: int = None
) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    Greedy selection algorithm to maximize information gain using the Schur complement.

    Args:
        inf_mats (List[np.ndarray or sparse matrix]): List of information matrices for each sensor.
        prior (np.ndarray): Prior information matrix.
        Nc (int): Number of sensors to select.
        metric (Metric): Metric to use for selection.
        num_runs (int): Number of runs (default is 1).
        num_poses (int): Number of poses in the problem (must be provided).

    Returns:
        Tuple[np.ndarray, float, np.ndarray]: Selection vector, best score, and availability vector.
    """
    if num_poses is None:
        raise ValueError("num_poses must be provided and cannot be None.")

    best_selection_indices = []
    best_score = float('-inf')
    avail_cand = np.ones(len(inf_mats), dtype=int)

    # Initialize the combined information matrix with the prior
    combined_inf_mat = prior.copy()

    for run in range(num_runs):
        for i in range(Nc):
            max_inf = float('-inf')
            selected_cand = None
            start_time = time.time()

            for j in range(len(inf_mats)):
                if avail_cand[j] == 1:
                    # Tentatively add the candidate sensor's information matrix
                    temp_inf_mat = combined_inf_mat + inf_mats[j]

                    # Compute the Schur complement
                    total_size = temp_inf_mat.shape[0]
                    pose_dim = 6
                    num_pose_elements = num_poses * pose_dim
                    measurement_dim = total_size - num_pose_elements

                    # Ensure the dimensions of submatrices are correct
                    if measurement_dim <= 0:
                        raise ValueError(f"Invalid measurement dimension: {measurement_dim}")

                    Hll = temp_inf_mat[:measurement_dim, :measurement_dim]
                    Hlx = temp_inf_mat[:measurement_dim, measurement_dim:]
                    Hxx = temp_inf_mat[measurement_dim:, measurement_dim:]

                    # Convert Hll to dense if it is sparse
                    if scipy.sparse.issparse(Hll):
                        Hll = Hll.toarray()

                    # Regularize and compute the pseudoinverse of Hll
                    reg_term = 1e-8 * np.eye(Hll.shape[0])
                    try:
                        Hll_inv = np.linalg.pinv(Hll + reg_term)
                    except np.linalg.LinAlgError as e:
                        raise ValueError(f"Failed to compute pseudoinverse for candidate {j}: {e}")

                    # Compute the Schur complement
                    H_schur = Hxx - Hlx.T @ Hll_inv @ Hlx

                    # Ensure H_schur is symmetric
                    H_schur = (H_schur + H_schur.T) / 2

                    # Compute the minimum eigenvalue of the Schur complement
                    eigvals = np.linalg.eigvalsh(H_schur)
                    min_eig_val = eigvals[0]
                    score = min_eig_val

                    if score > max_inf:
                        max_inf = score
                        selected_cand = j
            
            elapsed_time = time.time() - start_time
            print(f"Iteration {i} compute time: {elapsed_time:.4f} seconds - Min eigen: {max_inf:.4f}")

            if selected_cand is not None:
                best_score = max_inf
                best_selection_indices.append(selected_cand)
                avail_cand[selected_cand] = 0

                # Update the combined information matrix
                combined_inf_mat += inf_mats[selected_cand]

    print("Selected candidates are:", best_selection_indices)

    selection_vector = np.zeros(len(inf_mats))
    selection_vector[best_selection_indices] = 1
    return selection_vector, best_score, avail_cand


def solve_lmo(grad, A, b):
    """Closed-form LMO for the box + budget constraint: sum(x) <= K, x in [0,1]^N.

    The optimal solution is the K-sparse indicator of the K indices with the
    most negative gradient.  This is O(N log K) vs O(N^3) for linprog and is
    numerically stable regardless of gradient magnitude.
    """
    K = int(round(float(b[0])))
    s = np.zeros(len(grad), dtype=float)
    top_k = np.argpartition(grad, K)[:K]
    s[top_k] = 1.0
    return s


def frank_wolfe_optimization(
    inf_mats: List[csr_matrix],
    H0: csr_matrix,
    selection_init: np.ndarray,
    num_poses: int,
    A: np.ndarray,
    b: np.ndarray
) -> Tuple[np.ndarray, float, int]:
    """
    Performs Frank-Wolfe optimization to select sensors that maximize the smallest eigenvalue of the Schur complement.

    Args:
        inf_mats (List[csr_matrix]): List of sparse information matrices.
        prior (csr_matrix): Prior information matrix.
        n_iters (int): Number of iterations.
        selection_init (np.ndarray): Initial selection vector.
        k (int): Number of sensors to select.
        num_poses (int): Number of poses.
        A (np.ndarray): Inequality constraint matrix.
        b (np.ndarray): Inequality constraint bounds.

    Returns:
        Tuple[np.ndarray, float, int]: Final selection vector, best score (smallest eigenvalue), and number of iterations.
    """

    H0 = H0.tocsc()
    inf_mats = [Hi.tocsc() for Hi in inf_mats]
    A = sp.csc_matrix(A)
    pose_dim = 6
    intr_dim = H0.shape[0] - num_poses * pose_dim

    print(f"[FW] Pre-extracting dense blocks (intr_dim={intr_dim}, num_poses={num_poses})...", flush=True)
    H0_tt, H0_tn_blocks, H0_nn_blocks, all_h_tt, all_h_tn, all_h_nn = \
        precompute_fim_blocks(inf_mats, H0, num_poses, intr_dim)
    print(f"[FW] Block extraction done. Starting iterations...", flush=True)

    selection_cur = selection_init.copy().astype(float)
    prev_min_eig = -np.inf
    min_eig_val = prev_min_eig
    fw_gap_final = float("inf")

    for iteration in range(300):
        try:
            _, min_eig_val, min_eig_vec, T2_blocks, H_tn_blocks = \
                compute_calibration_schur_vectorized(
                    selection_cur, H0_tt, H0_tn_blocks, H0_nn_blocks, all_h_tt, all_h_tn, all_h_nn
                )
        except Exception as e:
            print(f"Error during Schur complement computation at iteration {iteration}: {e}")
            break

        rel_change = abs(min_eig_val - prev_min_eig) / max(abs(prev_min_eig), 1.0)
        prev_min_eig = min_eig_val

        print(f"[FW] iter={iteration:3d}  min_eig={min_eig_val:.6e}", flush=True)

        grad = compute_grad_vectorized(min_eig_vec, all_h_tt, all_h_tn, all_h_nn, T2_blocks, H_tn_blocks)

        s = solve_lmo(grad, A, b)
        if s is None:
            print(f"LMO failed to find a feasible solution at iteration {iteration}.")
            break

        fw_gap_final = float(grad @ (selection_cur - s))

        if rel_change < 1e-4 and fw_gap_final < 1e-4:
            print(f"[FW] Converged at iteration {iteration}  min_eig={min_eig_val:.6e}  gap={fw_gap_final:.2e}")
            break

        alpha = 2 / (iteration + 2)
        selection_cur = selection_cur + alpha * (s - selection_cur)
        selection_cur = np.clip(selection_cur, 0, 1)

    return selection_cur, min_eig_val, iteration + 1, fw_gap_final


def roundsolution(selection, k):
    """
    Selects the top `k` elements in the `selection` vector and sets them to 1 in `rounded_sol`.
    This method does not handle ties specifically, so it may arbitrarily choose elements if
    there are multiple candidates with the same value around the `k`-th element.

    Args:
        selection (np.ndarray): Array of selection scores for each candidate.
        k (int): Number of elements to select.

    Returns:
        np.ndarray: Binary vector where the top `k` elements in `selection` are marked as 1, others as 0.
    """
    idx = np.argpartition(selection, -k)[-k:]
    rounded_sol = np.zeros(len(selection))
    if k > 0:
        rounded_sol[idx] = 1.0
    return rounded_sol

def roundsolution_breakties(selection, k, all_mats, H0):
    """
    Selects the top `k` elements in the `selection` vector, breaking ties by using the smallest eigenvalue
    of a matrix formed by adding each candidate matrix in `all_mats` to the prior `H0`. This method ensures
    more robust selection by accounting for eigenvalue differences.

    Args:
        selection (np.ndarray): Array of selection scores for each candidate.
        k (int): Number of elements to select.
        all_mats (List[np.ndarray]): List of candidate matrices to compute eigenvalues for tie-breaking.
        H0 (np.ndarray): Prior information matrix added to each candidate matrix.

    Returns:
        np.ndarray: Binary vector where the top `k` elements in `selection`, based on both selection values and
                    eigenvalues, are marked as 1, others as 0.
    """
    s_rnd = np.round(selection, decimals=5)
    all_eigs = []
    for i, m in enumerate(all_mats):
        # Compute the smallest eigenvalue of the matrix H0 + m for each candidate
        m_p = H0 + m

        # Ensure symmetry
        m_p = (m_p + m_p.T) / 2  # Enforce numerical symmetry

        # Add a small regularization term for stability
        reg_term = 1e-8 * np.eye(m_p.shape[0])
        m_p += reg_term

        try:
            # Attempt sparse computation of the smallest eigenvalue
            eigval, _ = eigsh(m_p, k=1, which='SA', maxiter=2000)
        except Exception as e:
            print(f"ARPACK failed for matrix {i}, falling back to dense computation: {e}")
            if m_p.shape[0] <= 1000:  # Use dense computation for smaller matrices
                eigvals = np.linalg.eigh(m_p)[0]
                eigval = eigvals[:1]  # Take the smallest eigenvalue
            else:
                raise RuntimeError(f"Eigenvalue computation failed for large matrix {i}")

        all_eigs.append(eigval[0])  # Store the smallest eigenvalue
    all_eigs = np.array(all_eigs)

    # Combine selection scores and eigenvalues for tie-breaking
    zipped_vals = np.array([(s_rnd[i], all_eigs[i]) for i in range(len(s_rnd))],
                           dtype=[('w', 'float'), ('weight', 'float')])
    idx = np.argpartition(zipped_vals, -k, order=['w', 'weight'])[-k:]

    rounded_sol = np.zeros(len(s_rnd))
    if k > 0:
        rounded_sol[idx] = 1.0
    return rounded_sol


# def roundsolution_madow(selection, k):
#     """
#     Implements a weighted probabilistic rounding to select `k` elements
#     while preserving the relative importance of the `selection` scores.

#     Args:
#         selection (np.ndarray): Array of selection scores for each candidate (non-negative values).
#         k (int): Number of elements to select.

#     Returns:
#         np.ndarray: Binary vector with exactly `k` elements selected probabilistically based on their weights.
#     """
#     num = len(selection)
#     if k > num:
#         raise ValueError("k cannot be greater than the number of candidates.")
    
#     # Normalize the selection scores to form probabilities
#     normalized_selection = selection / np.sum(selection)

#     # Perform weighted random sampling without replacement
#     selected_indices = np.random.choice(
#         np.arange(num), size=k, replace=False, p=normalized_selection
#     )

#     # Construct the binary solution vector
#     rounded_sol = np.zeros(num, dtype=int)
#     rounded_sol[selected_indices] = 1

#     return rounded_sol

def roundsolution_madow(selection, k):
    """
    Uses a probabilistic approach to select `k` candidates based on the cumulative sum of the selection values.
    This method introduces randomness to the rounding process, which is useful if a non-deterministic selection is preferred.

    Args:
        selection (np.ndarray): Array of selection scores for each candidate.
        k (int): Number of elements to select.

    Returns:
        np.ndarray: Binary vector with exactly `k` elements selected probabilistically based on their cumulative weights.
    """
    num = len(selection)
    phi = np.zeros(num + 1)  
    rounded_sol = np.zeros(num)
    phi[1:] = np.cumsum(selection)  # Cumulative sum of selection scores
    u = np.random.rand()  # Random number for probabilistic selection

    for i in range(k):
        for j in range(num):
            # Check if the random value falls within the cumulative range
            if (phi[j] <= u + i) and (u + i < phi[j + 1]):
                if rounded_sol[j] == 1:  # Ensure the same element isn't selected twice
                    continue
                rounded_sol[j] = 1
                break
    # print("Number of candidates selected after rounding:", np.sum(rounded_sol))
    return rounded_sol

def evaluate_solution(inf_mats, H0, solution, num_poses):
    """
    Evaluates a binary solution by computing the smallest eigenvalue of the 
    information matrix constructed from selected sensors.
    
    Args:
        inf_mats (List[np.ndarray]): Information matrices for each sensor.
        H0 (np.ndarray): Prior information matrix.
        solution (np.ndarray): Binary selection vector for sensors.
        num_poses (int): Number of poses.
    
    Returns:
        float: The smallest eigenvalue of the resulting information matrix.
    """
    # Build the combined information matrix based on the selected sensors
    combined_fim = H0.copy()
    for i, selected in enumerate(solution):
        if selected:
            combined_fim += inf_mats[i]
    
    # Compute the smallest eigenvalue of the combined information matrix
    eigvals = np.linalg.eigvalsh(combined_fim)
    min_eig_val = eigvals[0]
    
    return min_eig_val

# Define a function to parallelize the computation for a single Hi
def compute_grad_parallel(idx, Hi, T1, T2, min_eig_vec, measurement_dim):
    """
    Compute gradient contribution for a single Hi matrix.

    Args:
        idx (int): Index of the matrix.
        Hi (scipy.sparse.csr_matrix): Information matrix.
        T1 (np.ndarray): Dense intermediate matrix from Schur computation - Hxl * inv(Hll)
        T2 (np.ndarray): intermediate matrix from Schur computation -  inv(Hll) * Hlx
        min_eig_vec (np.ndarray): Smallest eigenvector of Schur complement.
        measurement_dim (int): Dimensionality of the measurement space.

    Returns:
        Tuple[int, float]: Index and gradient contribution.
    """
    # Extract submatrices from Hi (keep sparse)
    Hll_i = Hi[:measurement_dim, :measurement_dim].tocsc()
    Hlx_i = Hi[:measurement_dim, measurement_dim:].tocsc()
    Hxx_i = Hi[measurement_dim:, measurement_dim:].tocsc()

    # Compute grad_schur
    # try:
    #     Y = spsolve(Hll, Hlx_i.tocsc())
    # except Exception as e:
    #     print(f"Linear solver failed for Hi index {idx}:", e)
    #     Y = inv(Hll).dot(Hlx_i)  # Sparse fallback
    #     Y = Y.toarray()  # Convert to dense for consistency

    grad_schur = Hxx_i - Hlx_i.transpose().dot(T2) + T1.dot(Hll_i.dot(T2)) - T1.dot(Hlx_i)

    # Flatten vectors to ensure proper alignment
    grad_value = -min_eig_vec.flatten().dot(grad_schur.dot(min_eig_vec).flatten())
    return idx, grad_value

'''
################################################################
Scipy optimization methods
'''
def min_eig_obj(x, inf_mats, H0, num_poses):
    """
    Computes the objective function value (negative smallest eigenvalue of Schur complement).

    Args:
        x (np.ndarray): Continuous selection vector.
        inf_mats (List[scipy.sparse.csr_matrix]): List of sparse information matrices.
        H0 (scipy.sparse.csr_matrix): Prior information matrix.
        num_poses (int): Number of poses.

    Returns:
        float: Objective function value.
    """
    _, min_eig_val, _, _, _, _ = compute_schur_complement(x, inf_mats, H0, num_poses)
    return -min_eig_val

def min_eig_grad(x, inf_mats, H0, num_poses):
    """
    Computes the gradient of the objective function (Jacobian).

    Args:
        x (np.ndarray): Continuous selection vector.
        inf_mats (List[scipy.sparse.csr_matrix]): List of sparse information matrices.
        H0 (scipy.sparse.csr_matrix): Prior information matrix.
        num_poses (int): Number of poses.

    Returns:
        np.ndarray: Gradient vector.
    """
    start_time = time.time()
    _, _, min_eig_vec, T1, T2, measurement_dim = compute_schur_complement(x, inf_mats, H0, num_poses)

    # Parallelized gradient computation
    results = Parallel(n_jobs=-1)(
        delayed(compute_grad_parallel)(idx, Hi, T1, T2, min_eig_vec, measurement_dim)
        for idx, Hi in enumerate(inf_mats)
    )

    grad = np.zeros_like(x)
    for idx, grad_value in results:
        grad[idx] = grad_value
    
    run_time = time.time() - start_time
    #print(f"jacobian call compute time: {run_time:.4f}")

    return grad

def scipy_minimize(inf_mats, H0, selection_init, num_poses, A, b):
    """
    Uses `scipy.optimize.minimize` with inequality constraints to solve a sensor selection problem.
    The objective is to maximize the smallest eigenvalue of the FIM.

    Args:
        inf_mats (List[np.ndarray]): List of information matrices for each candidate sensor configuration.
        H0 (np.ndarray): Prior information matrix.
        selection_init (np.ndarray): Initial continuous selection vector.
        num_poses (int): Number of poses in the problem.
        A (np.ndarray): Matrix defining inequality constraints.
        b (np.ndarray): Vector defining inequality constraints.

    Returns:
        Tuple[np.ndarray, float]: Continuous solution vector, maximum minimum eigenvalue.
    """
    # Optimization function that maximizes the smallest eigenvalue
    # Define bounds for all variables between 0 and 1
    bounds = Bounds([0] * len(selection_init), [1] * len(selection_init))

    # Define linear constraint Ax <= b
    linear_constraint = LinearConstraint(A, -np.inf, b)

    # Partial functions for objective and gradient
    obj_fun = partial(min_eig_obj, inf_mats=inf_mats, H0=H0, num_poses=num_poses)
    grad_fun = partial(min_eig_grad, inf_mats=inf_mats, H0=H0, num_poses=num_poses)

    # Optimization function that maximizes the smallest eigenvalue
    res = minimize(
        fun=obj_fun,
        x0=selection_init,  # Initial guess for the optimization variables
        method='SLSQP',  # Use Sequential Least Squares Quadratic Programming
        jac=grad_fun,
        constraints=[linear_constraint],
        bounds=bounds,  # Provide bounds
        options={'disp': True, 'maxiter': 10000, 'ftol': 1e-4} )

    # Derive intrinsics_dim the same way compute_schur_complement_d does:
    # the FIM is laid out as [intrinsics | poses], so intrinsics_dim = total - num_poses * 6.
    pose_dim = 6
    intrinsics_dim = H0.shape[0] - num_poses * pose_dim

    # Get the minimum eigenvalue of the continuous solution
    min_eig_val_unr, _, _ = infmat.find_min_eig_pair(inf_mats, res.x, H0, intrinsics_dim)

    return res.x, min_eig_val_unr

'''
################################################################
Scipy optimization methods with Log sum exponential on dense
'''
def min_eig_obj_lse_d(x, inf_mats, H0, num_poses, beta=5.0):
    """
    Computes the objective function value using the Log-Sum-Exp (LSE) approximation.
    """
    # Use the helper function to compute the Schur complement and auxiliary matrices
    H_schur, T1, T2,measurement_dim = compute_schur_complement_d(x, inf_mats, H0, num_poses)

    # Compute all eigenvalues of H_schur
    try:
        if H_schur.shape[0] <= 500:  # Threshold to determine when to switch to dense
            # Convert sparse matrix to dense for full eigenvalue computation
            if isinstance(H_schur, np.matrix):
                H_schur = np.asarray(H_schur)
            eigvals, eigvecs = np.linalg.eigh(H_schur)
        else:
            # For very large matrices, use eigsh for performance and check fallback
            eigvals, _ = eigsh(H_schur, k=H_schur.shape[0] - 1, which="SA")
    except Exception as e:
        print(f"Eigenvalue computation failed: {e}")
        return np.inf , np.zeros_like(x)

    # Stabilize Log-Sum-Exp computation
    eigvals_shifted = eigvals - eigvals.min()  # Shift to stabilize
    scaled_exp_eigvals = np.exp(-beta * eigvals_shifted)
    weight_sum = np.sum(scaled_exp_eigvals) + 1e-12

    # Objective value using Log-Sum-Exp
    f = (-1 / beta) * np.log(weight_sum) + eigvals.min()
    return -f

def min_eig_grad_lse_d(x, inf_mats, H0, num_poses, beta=5.0):
    """
    Computes the gradient of the objective function using the Log-Sum-Exp (LSE) approximation.
    """
    # Use the helper function to compute the Schur complement and auxiliary matrices
    s=time.time()
    H_schur, T1, T2, measurement_dim = compute_schur_complement_d(x, inf_mats, H0, num_poses)

    # Compute eigenvalues and eigenvectors
    try:
        if isinstance(H_schur, np.matrix):
            H_schur = np.asarray(H_schur)
        eigvals, eigvecs = np.linalg.eigh(H_schur)

    except Exception as e:
        print(f"Eigenvalue computation failed: {e}")
        return np.zeros_like(x)  # Return zero gradient if computation fails

    eigvals_shifted = eigvals - eigvals.min()  # Stabilize
    scaled_exp_eigvals = np.exp(-beta * eigvals_shifted)
    weight_sum = np.sum(scaled_exp_eigvals) + 1e-12
    softmax_weights = scaled_exp_eigvals / weight_sum

    # Parallelized gradient computation
    def compute_grad_lse_d(idx, H_j, eigvecs, softmax_weights, measurement_dim):

        s1 = time.time()
        Hll_j_d = H_j[:measurement_dim, :measurement_dim]
        Hlx_j_d = H_j[:measurement_dim, measurement_dim:]
        Hxx_j_d = H_j[measurement_dim:, measurement_dim:]
        grad_schur_d = Hxx_j_d - Hlx_j_d.T @ T2 + T1 @ Hll_j_d @ T2 - T1 @ Hlx_j_d

        lambda_derivatives_d = np.array([
            eigvecs[:, i].T @ grad_schur_d @ eigvecs[:, i]
            for i in range(len(eigvals))
        ])

        tmp_d = np.sum(softmax_weights * lambda_derivatives_d)
        # e1 = time.time()
        # execution_time = e1 - s1
        # print(f"execution time grad schur computation dense : {execution_time:.4f} seconds")

        return idx, tmp_d
    s1 = time.time()
    results = Parallel(n_jobs=-1)(
        delayed(compute_grad_lse_d)(idx, H_j.toarray(), eigvecs, softmax_weights, measurement_dim)
        for idx, H_j in enumerate(inf_mats)
    )
    # Collect results into the gradient vector
    grad = np.zeros_like(x)
    for idx, grad_value in results:
        grad[idx] = grad_value
    # e1 = time.time()
    # execution_time = e1 - s1
    # print(f"execution time grad schur computation : {execution_time:.4f} seconds")
    return -grad

def scipy_minimize_lse_d(inf_mats, H0, selection_init, num_poses, A, b):
    """
    Uses `scipy.optimize.minimize` with inequality constraints to solve a sensor selection problem
    using the Log-Sum-Exp (LSE) approximation, separating the objective and gradient calculations.
    """
    # Set bounds for each variable in x (between 0 and 1)
    bounds = [(0, 1) for _ in range(selection_init.shape[0])]

    # Define constraints
    cons = [{'type': 'ineq', 'fun': lambda x: b - A @ x}]

    # Define objective and gradient as separate functions
    def objective(x):
        return min_eig_obj_lse_d(x, inf_mats, H0, num_poses)

    def gradient(x):
        return min_eig_grad_lse_d(x, inf_mats, H0, num_poses)

    # Optimization function
    res = minimize(
        fun=objective,
        x0=selection_init,
        method='SLSQP',
        jac=gradient,
        constraints=cons,
        bounds=bounds,
        options={'disp': True, 'maxiter': 1000, 'ftol': 1e-4}
    )

    # Get the approximated minimum eigenvalue
    f_opt = min_eig_obj_lse_d(res.x, inf_mats, H0, num_poses)
    approx_min_eig_val = -f_opt

    return res.x, approx_min_eig_val

'''
################################################################
Scipy optimization methods with Log sum exponential
'''

def min_eig_obj_lse(x, inf_mats, H0, num_poses, beta=5.0):
    """
    Computes the objective function value using the Log-Sum-Exp (LSE) approximation.
    """
    # Use the helper function to compute the Schur complement and auxiliary matrices
    H_schur, min_eig_vec, T1, T2,measurement_dim = compute_schur_complement(x, inf_mats, H0, num_poses)

    # Compute all eigenvalues of H_schur
    try:
        if H_schur.shape[0] <= 500:  # Threshold to determine when to switch to dense
            # Convert sparse matrix to dense for full eigenvalue computation
            if scipy.sparse.issparse(H_schur):
                H_schur = H_schur.toarray()
            if isinstance(H_schur, np.matrix):
                H_schur = np.asarray(H_schur)
            eigvals, _ = np.linalg.eigh(H_schur)
        else:
            # For very large matrices, use eigsh for performance and check fallback
            eigvals, _ = eigsh(H_schur, k=H_schur.shape[0] - 1, which="SA")
    except Exception as e:
        print(f"Eigenvalue computation failed: {e}")
        return np.inf , np.zeros_like(x)

    # Stabilize Log-Sum-Exp computation
    eigvals_shifted = eigvals - eigvals.min()  # Shift to stabilize
    scaled_exp_eigvals = np.exp(-beta * eigvals_shifted)
    weight_sum = np.sum(scaled_exp_eigvals) + 1e-12

    # Objective value using Log-Sum-Exp
    f = (-1 / beta) * np.log(weight_sum) + eigvals.min()

    return -f

def min_eig_grad_lse(x, inf_mats, H0, num_poses, beta=5.0):
    """
    Computes the gradient of the objective function using the Log-Sum-Exp (LSE) approximation.
    """
    # Use the helper function to compute the Schur complement and auxiliary matrices
    H_schur, min_eig_vec, T1, T2, measurement_dim = compute_schur_complement(x, inf_mats, H0, num_poses)

    # Compute eigenvalues and eigenvectors
    try:
        if H_schur.shape[0] <= 500:
            if scipy.sparse.issparse(H_schur):
                H_schur = H_schur.toarray()
            if isinstance(H_schur, np.matrix):
                H_schur = np.asarray(H_schur)
            eigvals, eigvecs = np.linalg.eigh(H_schur)
        else:
            eigvals, eigvecs = eigsh(H_schur, k=H_schur.shape[0] - 1, which="SA")
    except Exception as e:
        print(f"Eigenvalue computation failed: {e}")
        return np.zeros_like(x)  # Return zero gradient if computation fails

    eigvals_shifted = eigvals - eigvals.min()  # Stabilize
    scaled_exp_eigvals = np.exp(-beta * eigvals_shifted)
    weight_sum = np.sum(scaled_exp_eigvals) + 1e-12
    softmax_weights = scaled_exp_eigvals / weight_sum

    # Parallelized gradient computation
    def compute_grad_lse(idx, H_j, eigvecs, softmax_weights, measurement_dim):
        # Hll_j = H_j[:measurement_dim, :measurement_dim].tocsc()
        # Hlx_j = H_j[:measurement_dim, measurement_dim:].tocsc()
        # Hxx_j = H_j[measurement_dim:,measurement_dim:].tocsc()

        H_j_d = H_j.toarray()
        Hll_j_d = H_j_d[:measurement_dim, :measurement_dim]
        Hlx_j_d = H_j_d[:measurement_dim, measurement_dim:]
        Hxx_j_d = H_j_d[measurement_dim:, measurement_dim:]
        grad_schur_d = Hxx_j_d - Hlx_j_d.T @ T2 + T1 @ Hll_j_d @ T2 - T1 @ Hlx_j_d
        lambda_derivatives_d = np.array([
            eigvecs[:, i].T @ grad_schur_d @ eigvecs[:, i]
            for i in range(len(eigvals))
        ])
        tmp_d = np.sum(softmax_weights * lambda_derivatives_d)
        return idx, tmp_d

    results = Parallel(n_jobs=-1)(
        delayed(compute_grad_lse)(idx, H_j, eigvecs, softmax_weights, measurement_dim)
        for idx, H_j in enumerate(inf_mats)
    )
    # Collect results into the gradient vector
    grad = np.zeros_like(x)
    for idx, grad_value in results:
        grad[idx] = grad_value

    return -grad

def scipy_minimize_lse(inf_mats, H0, selection_init, num_poses, A, b):
    """
    Uses `scipy.optimize.minimize` with inequality constraints to solve a sensor selection problem
    using the Log-Sum-Exp (LSE) approximation, separating the objective and gradient calculations.
    """
    # Set bounds for each variable in x (between 0 and 1)
    bounds = [(0, 1) for _ in range(selection_init.shape[0])]

    # Define constraints
    cons = [{'type': 'ineq', 'fun': lambda x: b - A @ x}]

    # Define objective and gradient as separate functions
    def objective(x):
        return min_eig_obj_lse(x, inf_mats, H0, num_poses)

    def gradient(x):
        return min_eig_grad_lse(x, inf_mats, H0, num_poses)

    # Optimization function
    res = minimize(
        fun=objective,
        x0=selection_init,
        method='SLSQP',
        jac=gradient,
        constraints=cons,
        bounds=bounds,
        options={'disp': True, 'maxiter': 1000, 'ftol': 1e-4}
    )

    # Get the approximated minimum eigenvalue
    f_opt = min_eig_obj_lse(res.x, inf_mats, H0, num_poses)
    approx_min_eig_val = -f_opt

    return res.x, approx_min_eig_val

'''
################################################################
GUROBI Implementation
'''

# Define the callback for Branch and Cut
def branch_and_cut_callback(model, where):
    if where == GRB.Callback.MIPSOL:
        # Retrieve the solution values as a list
        sol = model.cbGetSolution([v for v in model._x.values()])
        
        # Identify selected indices
        selected_indices = [i for i, val in enumerate(sol) if val > 0.5]
        if not selected_indices:
            return

        # Combine the FIMs based on selected indices
        combined_fim = model._H0.copy()
        for idx in selected_indices:
            combined_fim += model._inf_mats[idx]

        # Compute Schur complement and minimum eigenvalue
        try:
            pose_dim = 6
            num_pose_elements = model._num_poses * pose_dim
            measurement_dim = combined_fim.shape[0] - num_pose_elements

            Hll = combined_fim[:measurement_dim, :measurement_dim].tocsc()
            Hlx = combined_fim[:measurement_dim, measurement_dim:].tocsc()
            Hxx = combined_fim[measurement_dim:, measurement_dim:].tocsc()

            X = spsolve(Hll, Hlx)
            H_schur = Hxx - Hlx.T @ X

            # Ensure H_schur is symmetric
            H_schur = (H_schur + H_schur.T) / 2

            # Compute the minimum eigenvalue
            min_eig_val, _ = eigsh(H_schur, k=1, which='SA')
        except Exception as e:
            return

        # Add the lazy constraint
        model.cbLazy(model._min_eig_var <= min_eig_val[0])

def gurobi_branch_and_cut(inf_mats, H0, num_sensors, k, num_poses, A, b, upper_bound=1e6):
    """
    Solves the sensor selection problem using Gurobi's MIP solver with Branch and Cut.

    Args:
        inf_mats (List[scipy.sparse.csr_matrix]): List of sparse information matrices.
        H0 (scipy.sparse.csr_matrix): Prior information matrix.
        num_sensors (int): Number of candidate sensors.
        k (int): Number of sensors to select.
        num_poses (int): Number of poses.
        A (scipy.sparse.csr_matrix): Inequality constraint matrix.
        b (np.ndarray): Inequality constraint bounds.
        upper_bound (float): Upper bound for min_eig_var to prevent unboundedness.

    Returns:
        Tuple[List[int], float]: Selected sensors and the best score (maximum minimum eigenvalue).
    """
    # Ensure H0 is in CSC format for efficient arithmetic operations
    H0 = H0.tocsc()
    
    # Ensure all inf_mats are in CSC format
    inf_mats = [Hi.tocsc() for Hi in inf_mats]

    # Optionally, precompute an upper bound for min_eig_var
    try:
        combined_fim_max = H0.copy()
        for Hi in inf_mats:
            combined_fim_max += Hi
        max_eig_val, _ = eigsh(combined_fim_max, k=1, which='LA')
        upper_bound = max_eig_val[0] * 2  # Set upper bound slightly above the maximum eigenvalue
    except Exception as e:
        return
        
    # Initialize Gurobi model
    model = Model("SensorSelection")
    model.Params.LazyConstraints = 1

    # Add decision variables for sensor selection
    x = model.addVars(num_sensors, vtype=GRB.BINARY, name="x")

    # Add a continuous variable for minimum eigenvalue with an upper bound
    min_eig_var = model.addVar(vtype=GRB.CONTINUOUS, name="min_eig_var", ub=upper_bound)

    # Set the objective to maximize the minimum eigenvalue
    model.setObjective(min_eig_var, GRB.MAXIMIZE)
    
    # Add constraint to select exactly k sensors
    model.addConstr(x.sum() == k+1, name="sensor_selection")
    

    # Attach additional data to the model
    model._x = x
    model._min_eig_var = min_eig_var
    model._inf_mats = inf_mats
    model._H0 = H0
    model._num_poses = num_poses

    # Set the callback
    model.optimize(branch_and_cut_callback)

    # Check if an optimal solution was found
    if model.status == GRB.OPTIMAL or model.status == GRB.SUBOPTIMAL:
        # Extract the solution
        selected_sensors = [j for j in range(num_sensors) if x[j].X > 0.5]
        best_score = min_eig_var.X
        return selected_sensors, best_score
    else:
        print(f"Optimization was not successful. Status code: {model.status}")
        return None, None
