import numpy as np
import torch
import ot
from typing import Optional
from scipy.optimize import linear_sum_assignment


class BarycenterOptimizer:
    """
    Free-support Wasserstein barycenter optimizer for latent morphing
    """

    def __init__(
        self,
        max_iter: int = 50,
    ):
        self.max_iter = max_iter

    @staticmethod
    def _to_numpy(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return x

    def compute_barycenter(
        self,
        lat_src: np.ndarray,
        lat_tgt: np.ndarray,
        alpha: float,
        init_support: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Wukong-style free-support barycenter between source and target
        """

        # --- numpy ---
        if isinstance(lat_src, torch.Tensor):
            lat_src = lat_src.cpu().numpy()
        if isinstance(lat_tgt, torch.Tensor):
            lat_tgt = lat_tgt.cpu().numpy()

        Ns, d = lat_src.shape
        Nt, _ = lat_tgt.shape
        assert Ns == Nt, "Source and target must have the same number of tokens"

        K = Ns

        # --- init support ---
        if init_support is not None:
            Z = init_support.copy()
        else:
            Z = lat_src.copy()

        for _ in range(self.max_iter):

            Z_prev = Z.copy()

            # cost matrices
            C_ZX = ot.dist(Z, lat_src, metric="sqeuclidean")
            C_ZY = ot.dist(Z, lat_tgt, metric="sqeuclidean")

            # exact OT (permutations)
            _, pi_X = linear_sum_assignment(C_ZX)
            _, pi_Y = linear_sum_assignment(C_ZY)

            # free-support barycenter update (ARTICLE FORMULA)
            Z = (1 - alpha) * lat_src[pi_X] + alpha * lat_tgt[pi_Y]

            # convergence
            if np.linalg.norm(Z - Z_prev) / (np.linalg.norm(Z_prev) + 1e-9) < 1e-6:
                break

        return Z


    def compute_morphing_sequence(
        self,
        lat_src: np.ndarray,
        lat_tgt: np.ndarray,
        num_steps: int = 10,
        n_support: Optional[int] = None,
    ):
        """
        Compute morphing sequence via Wasserstein barycenters
        """
        lat_src = self._to_numpy(lat_src)
        lat_tgt = self._to_numpy(lat_tgt)

        sequence = []
        Z_prev = None

        alphas = np.linspace(0.0, 1.0, num_steps + 1)

        for alpha in alphas:
            Z = self.compute_barycenter(
                lat_src,
                lat_tgt,
                alpha,
                init_support=Z_prev,
            )
            sequence.append(Z)
            Z_prev = Z

        return sequence
