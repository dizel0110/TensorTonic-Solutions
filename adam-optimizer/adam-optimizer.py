import numpy as np


def adam_step(param, grad, m, v, t, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
    """
    One Adam optimizer update step.
    Return (param_new, m_new, v_new).
    """
    # Конвертация входов в NumPy array для векторизации
    param = np.asarray(param, dtype=float)
    grad = np.asarray(grad, dtype=float)
    m = np.asarray(m, dtype=float)
    v = np.asarray(v, dtype=float)

    # 1. Обновление первого момента (momentum)
    m_new = beta1 * m + (1 - beta1) * grad

    # 2. Обновление второго момента (squared gradient)
    v_new = beta2 * v + (1 - beta2) * (grad ** 2)

    # 3. Bias correction для первого момента
    m_hat = m_new / (1 - beta1 ** t)

    # 4. Bias correction для второго момента
    v_hat = v_new / (1 - beta2 ** t)

    # 5. Обновление параметра
    param_new = param - lr * m_hat / (np.sqrt(v_hat) + eps)

    # Гарантируем возврат np.ndarray (для скаляров это 0-d array)
    return (np.asarray(param_new), np.asarray(m_new), np.asarray(v_new))
