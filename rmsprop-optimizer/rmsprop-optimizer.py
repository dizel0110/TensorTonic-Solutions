import numpy as np


def rmsprop_step(w, g, s, lr=0.001, beta=0.9, eps=1e-8):
    """
    Perform one RMSProp update step.
    """
    # Конвертация входов в NumPy array для векторизации
    w = np.asarray(w, dtype=float)
    g = np.asarray(g, dtype=float)
    s = np.asarray(s, dtype=float)

    # Step 1: Update running average of squared gradients
    # s_t = β * s_{t-1} + (1 - β) * g_t²
    new_s = beta * s + (1 - beta) * (g ** 2)

    # Step 2: Parameter update
    # w_t = w_{t-1} - lr * g_t / (√s_t + ε)
    new_w = w - lr * g / (np.sqrt(new_s) + eps)

    # Гарантируем возврат np.ndarray (для скаляров это 0-d array)
    return (np.asarray(new_w), np.asarray(new_s))
