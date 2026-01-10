import math
from typing import List

import numpy as np
import torch
import torch.nn as nn
from loguru import logger


class HessianCalculator:
    """
    Helper class to compute Hessian characteristics using Hessian-Vector Products (HVP).
    """

    def __init__(self, model: nn.Module, loss_fn, data_loader, device=None, max_batches=None):
        self.model = model
        self.loss_fn = loss_fn
        self.data_loader = data_loader
        self.device = device or next(model.parameters()).device
        self.max_batches = max_batches

    def _compute_hvp(self, v: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Compute Hessian-Vector Product (Hv) for a given vector v.
        """
        # HVP needs to be computed over the dataset (or a subset)
        # H = E[nabla^2 L]
        # Hv = E[nabla^2 L v] = E[grad(grad L * v)]

        total_hvp = [torch.zeros_like(p) for p in v]
        num_batches = 0

        self.model.eval()
        # We need gradients, so zero out existing grads
        self.model.zero_grad()

        for batch in self.data_loader:
            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                inputs, targets = batch[:2]
            else:
                inputs = batch
                targets = None

            inputs = inputs.to(self.device)
            if targets is not None:
                targets = targets.to(self.device)

            # 1. Compute Loss
            outputs = self.model(inputs)
            if targets is not None:
                loss = self.loss_fn(outputs, targets)
            else:
                loss = self.loss_fn(outputs)

            # 2. Compute Gradients (grad L)
            params = [p for p in self.model.parameters() if p.requires_grad]
            grads = torch.autograd.grad(loss, params, create_graph=True, retain_graph=True)

            # 3. Compute dot product (grad L * v)
            # v is a list of tensors matching params
            dot_prod = 0
            for g, vec in zip(grads, v):
                dot_prod += torch.sum(g * vec)

            # 4. Compute gradient of dot product w.r.t params (Hv)
            hvp = torch.autograd.grad(dot_prod, params, retain_graph=False)

            # Accumulate
            for i, h in enumerate(hvp):
                total_hvp[i] += h.detach()

            num_batches += 1
            if self.max_batches and num_batches >= self.max_batches:
                break

        # Average
        if num_batches > 0:
            total_hvp = [h / num_batches for h in total_hvp]

        return total_hvp

    def compute_top_eigenvalues(
        self, k: int = 1, max_iter: int = 20, tol: float = 1e-3
    ) -> List[float]:
        """
        Compute top k eigenvalues using deflation and power iteration.
        Used for small k (e.g., 1 or 2). For larger k, consider Lanczos.
        """
        params = [p for p in self.model.parameters() if p.requires_grad]
        eigenvalues = []
        eigenvectors = []  # List[List[Tensor]]

        logger.info(f"Computing top {k} eigenvalues (Power Iteration)...")

        for i in range(k):
            # Random initialization
            v = [torch.randn_like(p) for p in params]
            # Normalize
            norm = math.sqrt(sum(torch.sum(t**2).item() for t in v))
            v = [t / norm for t in v]

            curr_eig = 0.0

            for step in range(max_iter):
                # Orthogonalize v against found eigenvectors (Deflation)
                for u in eigenvectors:
                    dot = sum(torch.sum(vec * u_vec).item() for vec, u_vec in zip(v, u))
                    v = [vec - dot * u_vec for vec, u_vec in zip(v, u)]

                # Normalize again
                norm = math.sqrt(sum(torch.sum(t**2).item() for t in v))
                if norm < 1e-6:
                    break
                v = [t / norm for t in v]

                # Compute Hv
                hv = self._compute_hvp(v)

                # Rayleigh quotient
                new_eig = sum(torch.sum(vec * h_vec).item() for vec, h_vec in zip(v, hv))

                if abs(new_eig - curr_eig) < tol:
                    curr_eig = new_eig
                    v = hv  # Approx next direction
                    break

                curr_eig = new_eig
                v = hv

                # Normalize
                norm = math.sqrt(sum(torch.sum(t**2).item() for t in v))
                if norm > 0:
                    v = [t / norm for t in v]

            eigenvalues.append(curr_eig)
            eigenvectors.append(v)

        return eigenvalues

    def compute_spectrum_lanczos(self, k: int = 20, max_iter: int = 40) -> List[float]:
        """
        Compute top/bottom eigenvalues using Lanczos Algorithm.
        This is more efficient for larger k and gives a better approximation of the spectrum density.

        Args:
            k: Number of eigenvalues to return (will return top k magnitude)
            max_iter: Number of Lanczos iterations (dimension of Krylov subspace)
                     Typically max_iter should be >= k, preferably 2*k.
        """
        logger.info(f"Computing spectrum (Lanczos, m={max_iter})...")

        params = [p for p in self.model.parameters() if p.requires_grad]

        # Initial random vector v1
        v = [torch.randn_like(p, device=self.device) for p in params]
        # Normalize
        norm = math.sqrt(sum(torch.sum(t**2).item() for t in v))
        v_curr = [t / norm for t in v]  # v_1

        v_prev = [torch.zeros_like(p) for p in params]  # v_0 (zero vector)
        beta_prev = 0.0  # beta_0 = 0

        alphas = []
        betas = []

        # Lanczos Iteration
        # T matrix (tridiagonal):
        # [ alpha_1, beta_1,  0, ... ]
        # [ beta_1,  alpha_2, beta_2 ]
        # ...

        for j in range(max_iter):
            # w = H * v_j
            w = self._compute_hvp(v_curr)

            # alpha_j = w^T v_j
            alpha = sum(torch.sum(w_p * v_p).item() for w_p, v_p in zip(w, v_curr))
            alphas.append(alpha)

            # w = w - alpha_j * v_j - beta_{j-1} * v_{j-1}
            w = [
                w_p - alpha * v_c_p - beta_prev * v_p_p
                for w_p, v_c_p, v_p_p in zip(w, v_curr, v_prev)
            ]

            # beta_j = ||w||
            beta = math.sqrt(sum(torch.sum(t**2).item() for t in w))

            if beta < 1e-6:
                # Invariant subspace found
                break

            betas.append(beta)

            # Update vectors
            v_prev = v_curr
            beta_prev = beta
            v_curr = [t / beta for t in w]  # v_{j+1}

        # Construct Tridiagonal Matrix T
        # Dimension is actually len(alphas)
        m = len(alphas)
        if m == 0:
            return []

        T = np.zeros((m, m))
        np.fill_diagonal(T, alphas)
        # Fill off-diagonals
        for i, beta in enumerate(betas):
            if i < m - 1:
                T[i, i + 1] = beta
                T[i + 1, i] = beta

        # Compute eigenvalues of T
        eigvals = np.linalg.eigvalsh(T)

        # Sort by magnitude (largest absolute value first)
        # But for spectrum density, we usually want algebraic order (min to max)
        # For top-k, we sort by magnitude.
        # Let's return sorted algebraic eigenvalues (all of them), so we can plot density.
        # But user might want just the top-k significant ones.

        # Filter for top-k magnitude if requested, but better to return all m estimates
        # to get a better density plot.

        # Return sorted eigenvalues (algebraic, smallest to largest)
        # This approximates the distribution.
        eigvals.sort()

        # If we specifically want "Top K Magnitude", we can return just those.
        # But for "Spectrum" plot, we want the whole distribution.
        # We'll return ALL computed Ritz values.
        # However, to be consistent with the API requesting "k", maybe we ensure we return at least k?
        # Actually, let's just return all Ritz values. The frontend can decide how to bin them.

        return eigvals.tolist()

    def compute_trace(self, max_iter: int = 10) -> float:
        """
        Compute Hessian trace using Hutchinson's method.
        """
        params = [p for p in self.model.parameters() if p.requires_grad]
        trace_accum = 0.0

        logger.info(f"Computing Hessian trace (Hutchinson, {max_iter} iters)...")

        for i in range(max_iter):
            # Rademacher distribution (Bernoulli +-1)
            v = [torch.randint_like(p, high=2) * 2 - 1 for p in params]
            v = [t.float() for t in v]

            hv = self._compute_hvp(v)

            # v^T H v
            curr_val = sum(torch.sum(vec * h_vec).item() for vec, h_vec in zip(v, hv))
            trace_accum += curr_val

        return trace_accum / max_iter
