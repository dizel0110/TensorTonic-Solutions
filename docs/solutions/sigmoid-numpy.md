# Implement Sigmoid in NumPy

**Problem #1** | [TensorTonic](https://www.tensortonic.com/problems/sigmoid-numpy)

---

## 🌐 Language / Язык

- **[🇬🇧 English](#english)**
- **[🇷🇺 Русский](#русский)**

---

<a name="english"></a>
## 🇬🇧 English Version

### 📋 Requirements

Implement the sigmoid function:

```
σ(x) = 1 / (1 + e^(-x))
```

**Constraints:**
- Vectorized implementation only (no Python loops)
- NumPy only
- Works with scalars, lists, and arrays
- Returns `np.ndarray` of floats
- Time limit: 200 ms, Memory: 64 MB

### 🧪 Test Examples

| Input | Output |
|-------|--------|
| `x = 0` | `0.5` |
| `x = [0, 2, -2]` | `[0.5, 0.88079708, 0.11920292]` |
| `x = [[-1, 0], [1, 2]]` | `[[0.26894142, 0.5], [0.73105858, 0.88079708]]` |

### 🔧 Solution

```python
import numpy as np


def sigmoid(x) -> np.ndarray:
    """
    Compute the sigmoid function for input.
    
    Sigmoid formula: σ(x) = 1 / (1 + e^(-x))
    
    Args:
        x: Scalar, list, or NumPy array
        
    Returns:
        NumPy array of floats with sigmoid values
    """
    x = np.asarray(x, dtype=float)
    return np.asarray(1 / (1 + np.exp(-x)))
```

### ✅ Tests

All 7 tests pass.

---

<a name="русский"></a>
## 🇷🇺 Русская версия

### 📋 Требования

Реализовать сигмоидную функцию:

```
σ(x) = 1 / (1 + e^(-x))
```

**Ограничения:**
- Только векторизованная реализация (без Python циклов)
- Только NumPy
- Работает со скалярами, списками и массивами
- Возвращает `np.ndarray` float
- Time limit: 200 ms, Memory: 64 MB

### 🧪 Тестовые примеры

| Вход | Выход |
|------|-------|
| `x = 0` | `0.5` |
| `x = [0, 2, -2]` | `[0.5, 0.88079708, 0.11920292]` |
| `x = [[-1, 0], [1, 2]]` | `[[0.26894142, 0.5], [0.73105858, 0.88079708]]` |

### 🔧 Решение

Код идентичен английской версии выше.

### ✅ Тесты

Все 7 тестов проходят.

---

## 📊 Sigmoid Plot / График сигмоиды

```
     1.0 |                    ╱────
         |                 ╱
         |              ╱
σ(x)   0.5 |───────────●──────────
         |        ╱
         |     ╱
     0.0 |────╱
         ─────┼────┼────┼────┼────
            -2   -1    0    1    2
                  x
```

**Properties / Свойства:**
- σ(0) = 0.5 (center / центр)
- σ(x) → 1 as x → +∞
- σ(x) → 0 as x → -∞
- Range / Диапазон: (0, 1)
- Monotonically increasing / Монотонно возрастает

---

## 🔑 Key Points / Ключевые моменты

| Aspect / Аспект | Solution / Решение |
|-----------------|-------------------|
| **Input conversion** | `np.asarray(x, dtype=float)` |
| **Formula** | `1 / (1 + np.exp(-x))` |
| **Vectorization** | All NumPy operations are vectorized |
| **Output type** | `np.asarray()` guarantees ndarray |
| **Complexity** | O(n) — single pass |

---

## 📖 Resources / Ресурсы

- [NumPy: np.asarray](https://numpy.org/doc/stable/reference/generated/numpy.asarray.html)
- [NumPy: Broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html)
- [Sigmoid Function — Wikipedia](https://en.wikipedia.org/wiki/Sigmoid_function)
