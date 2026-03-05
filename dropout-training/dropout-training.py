import numpy as np


def dropout(x, p=0.5, rng=None):
    """
    Apply dropout to input x with probability p.
    Return (output, dropout_pattern).
    """
    # Конвертация входов в NumPy array
    x = np.asarray(x, dtype=float)

    # Генерация случайных чисел для dropout mask
    if rng is not None:
        random_vals = rng.random(x.shape)
    else:
        random_vals = np.random.random(x.shape)

    # Маска: сохраняем если random < (1-p), дропаем если random >= (1-p)
    # Вероятность сохранения = (1 - p)
    mask = (random_vals < (1.0 - p)).astype(float)

    # Скалирование: 1 / (1 - p) для сохранения ожидаемого значения
    scale = 1.0 / (1.0 - p) if p < 1.0 else 1.0

    # Dropout pattern: 0 для дропнутых, scale для сохранённых
    dropout_pattern = mask * scale

    # Применяем dropout к входу
    output = x * dropout_pattern

    # Гарантируем возврат np.ndarray (для скаляров это 0-d array)
    return (np.asarray(output), np.asarray(dropout_pattern))
