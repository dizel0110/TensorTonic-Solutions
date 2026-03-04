# Implement Sigmoid in NumPy

**Problem #1** | [TensorTonic](https://www.tensortonic.com/problems/sigmoid-numpy)

> 🇷🇺 [Русская версия ниже](#russian-summary)

---

## 📋 Requirements

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

---

## 🧪 Test Examples

| Input | Output |
|-------|--------|
| `x = 0` | `0.5` |
| `x = [0, 2, -2]` | `[0.5, 0.88079708, 0.11920292]` |
| `x = [[-1, 0], [1, 2]]` | `[[0.26894142, 0.5], [0.73105858, 0.88079708]]` |

---

## 🔧 Solution

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

### Key Implementation Details

| Step | Code | Purpose |
|------|------|---------|
| 1. Convert input | `x = np.asarray(x, dtype=float)` | Handles scalars, lists, arrays |
| 2. Apply formula | `1 / (1 + np.exp(-x))` | Vectorized sigmoid computation |
| 3. Ensure ndarray | `np.asarray(...)` | Guarantees return type for scalars |

---

## 📊 Sigmoid Plot

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

**Properties:**
- σ(0) = 0.5 (center)
- σ(x) → 1 as x → +∞
- σ(x) → 0 as x → -∞
- Range: (0, 1)
- Monotonically increasing

---

## ✅ Tests

All 7 tests pass:

| Test | Checks |
|------|--------|
| `test_scalar_zero` | Scalar input → array, value 0.5 |
| `test_list_positive_negative` | List input, exact values |
| `test_2d_array` | 2D structure preserved |
| `test_numpy_array` | NumPy array input |
| `test_large_positive` | σ(10) ≈ 1.0 |
| `test_large_negative` | σ(-10) ≈ 0.0 |
| `test_vectorized_no_loops` | Works on 1000 elements |

---

<a name="russian-summary"></a>
## 🇷🇺 Краткое резюме (Russian Summary)

### Требования
Реализовать функцию сигмоиды **векторизованно** (без циклов), только NumPy. Работает со скалярами, списками, массивами. Возвращает `np.ndarray` float.

### Решение
```python
x = np.asarray(x, dtype=float)      # Конвертация входа
return 1 / (1 + np.exp(-x))         # Формула
```

### Ключевые моменты
- `np.asarray()` — универсальная конвертация (скаляр → 0-d массив)
- `np.exp(-x)` — векторизованная экспонента
- Финальный `np.asarray()` гарантирует возврат ndarray даже для скаляра

### Тесты
7 тестов проверяют скаляры, списки, 2D массивы и граничные значения (±10).

---

## 📖 Resources

- [NumPy: np.asarray](https://numpy.org/doc/stable/reference/generated/numpy.asarray.html)
- [Sigmoid Function — Wikipedia](https://en.wikipedia.org/wiki/Sigmoid_function)
