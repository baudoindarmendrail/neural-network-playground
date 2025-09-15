from numpy import ndarray, empty, tanh, mean, log, clip
from math import tanh as math_tanh

try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

# --- Version numba ---
if NUMBA_AVAILABLE:
    @njit(fastmath=True, cache=True)
    def sigmoid_numba(z: ndarray) -> ndarray:
        out = empty(z.shape, dtype=z.dtype)
        flat_z = z.ravel()
        flat_out = out.ravel()
        for i in range(flat_z.size):
            x = flat_z[i]
            flat_out[i] = 0.5 * (1.0 + math_tanh(0.5 * x))
        return out.reshape(z.shape)

    @njit(fastmath=True, cache=True)
    def log_loss_numba(data: ndarray, model: ndarray) -> float:
        eps = 1e-15
        data_clipped = clip(data, eps, 1 - eps)
        loss = -(model * log(data_clipped) + (1 - model) * log(1 - data_clipped))
        return mean(loss)

# --- Version numpy pure (fallback) ---
else:
    def sigmoid_numba(z: ndarray) -> ndarray:
        # fallback numpy
        return 0.5 * (1.0 + tanh(0.5 * z))

    def log_loss_numba(data: ndarray, model: ndarray) -> float:
        """
        Log-loss (cross entropy binaire)
        """
        eps = 1e-15  # borne standard
        data = clip(data, eps, 1 - eps)
        loss = -(model * log(data) + (1 - model) * log(1 - data))
        return float(mean(loss))