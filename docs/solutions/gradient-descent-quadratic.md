# Implement Gradient Descent for a 1D Quadratic

**Problem #6** | [TensorTonic](https://www.tensortonic.com/problems/gradient-descent-quadratic)

> 🇷🇺 [Русская версия ниже](#russian-summary)

---

## 📋 Requirements

Implement vanilla gradient descent to minimize a 1-D quadratic function:

```
f(x) = ax² + bx + c
```

**Gradient:**
```
f'(x) = 2ax + b
```

**Update Rule:**
```
x = x - lr * f'(x)
  = x - lr * (2ax + b)
```

**Constraints:**
- Use the update rule repeated `steps` times
- Do NOT use the closed-form minimizer during updates
- Return a Python `float` (not list/array)
- Assume `a > 0`, `lr > 0`, `steps >= 1`
- Time limit: 200 ms, Memory: 64 MB
- Pure Python / NumPy (no ML libs)

---

## 🧪 Test Examples

### Example 1

**Input:**
```python
a = 1, b = -4, c = 3, x0 = 0, lr = 0.1, steps = 50
```

**Output:** `≈ 2.0`

**Explanation:** Minimum of `x² - 4x + 3` is at `x = -b/(2a) = 4/2 = 2.0`

### Example 2

**Input:**
```python
a = 0.5, b = -1, c = 0, x0 = -5, lr = 0.2, steps = 100
```

**Output:** `≈ 1.0`

**Explanation:** Minimum of `0.5x² - x` is at `x = -b/(2a) = 1/1 = 1.0`

---

## 💡 Hint

Use a loop to update `x` repeatedly.

---

## 🔧 Solution

```python
def gradient_descent_quadratic(a, b, c, x0, lr, steps):
    """
    Minimize f(x) = ax² + bx + c using gradient descent.
    
    Gradient: f'(x) = 2ax + b
    
    Update rule: x = x - lr * f'(x)
    
    Args:
        a: Coefficient of x² (must be > 0)
        b: Coefficient of x
        c: Constant term
        x0: Initial x value
        lr: Learning rate (must be > 0)
        steps: Number of iterations (must be >= 1)
        
    Returns:
        Final x value as Python float
    """
    x = float(x0)
    
    for _ in range(steps):
        # Compute gradient: f'(x) = 2ax + b
        gradient = 2 * a * x + b
        
        # Update x: x = x - lr * gradient
        x = x - lr * gradient
    
    return float(x)
```

### Key Implementation Details

| Step | Code | Purpose |
|------|------|---------|
| 1. Initialize | `x = float(x0)` | Convert to Python float |
| 2. Loop | `for _ in range(steps)` | Repeat `steps` times |
| 3. Gradient | `gradient = 2 * a * x + b` | Derivative of quadratic |
| 4. Update | `x = x - lr * gradient` | Move opposite to gradient |
| 5. Return | `return float(x)` | Python float result |

---

## 📊 Visualization

### Gradient Descent Trajectory

```
f(x) = x² - 4x + 3  (minimum at x = 2)

f(x)
  |
15|● x0 = 0
  | \
  |  \
  |   \
  |    \
  |     ● x1 = 0.8
  |      \
  |       \
  |        \
  |         ● x2 = 1.44
  |          \
  |           \
  |            ●●●●● x ≈ 2.0 (converged)
  |___________________________ x
  0    1    2    3    4
```

### Update Steps (first 5 iterations)

```
Step 0:  x = 0.00,  f'(x) = -4.00,  x_new = 0.00 - 0.1*(-4.00) = 0.40
Step 1:  x = 0.40,  f'(x) = -3.20,  x_new = 0.40 - 0.1*(-3.20) = 0.72
Step 2:  x = 0.72,  f'(x) = -2.56,  x_new = 0.72 - 0.1*(-2.56) = 0.98
Step 3:  x = 0.98,  f'(x) = -2.04,  x_new = 0.98 - 0.1*(-2.04) = 1.18
Step 4:  x = 1.18,  f'(x) = -1.64,  x_new = 1.18 - 0.1*(-1.64) = 1.34
...
Step 50: x ≈ 2.00 (converged to minimum)
```

---

## ✅ Tests

All 12 tests pass:

| Test | Checks |
|------|--------|
| `test_example_1` | Example 1 (minimum at x=2) |
| `test_example_2` | Example 2 (minimum at x=1) |
| `test_return_type` | Returns Python float |
| `test_zero_gradient` | Start at minimum → no change |
| `test_single_step` | Manual verification of 1 step |
| `test_negative_start` | Negative x0 converges |
| `test_large_learning_rate` | lr=0.5 converges faster |
| `test_small_learning_rate` | lr=0.01 needs more steps |
| `test_many_steps` | High precision (1000 steps) |
| `test_different_coefficients` | Various a, b values |
| `test_convergence_to_minimum` | Matches analytical solution |
| `test_c_does_not_affect_minimum` | c doesn't change minimum |

---

<a name="russian-summary"></a>
## 🇷🇺 Краткое резюме (Russian Summary)

### Требования
Реализовать градиентный спуск для минимизации квадратичной функции:

```
f(x) = ax² + bx + c
```

**Градиент:**
```
f'(x) = 2ax + b
```

**Правило обновления:**
```
x = x - lr * f'(x)
```

**Ограничения:**
- Использовать update rule `steps` раз
- НЕ использовать готовую формулу минимума
- Вернуть Python `float`
- `a > 0`, `lr > 0`, `steps >= 1`

### Решение

```python
def gradient_descent_quadratic(a, b, c, x0, lr, steps):
    x = float(x0)
    
    for _ in range(steps):
        gradient = 2 * a * x + b  # Градиент
        x = x - lr * gradient     # Обновление
    
    return float(x)
```

### Как это работает

1. **Инициализация:** Начинаем с `x0`
2. **Вычисляем градиент:** `f'(x) = 2ax + b`
3. **Обновляем x:** Двигаемся в направлении, противоположном градиенту
4. **Повторяем:** `steps` раз
5. **Возвращаем:** Финальное значение `x`

### Аналитический минимум

Для проверки: минимум квадратичной функции находится в точке:

```
x_min = -b / (2a)
```

**Примеры:**
- `x² - 4x + 3`: минимум при `x = 4/2 = 2`
- `0.5x² - x`: минимум при `x = 1/1 = 1`
- `2x² + 4x + 1`: минимум при `x = -4/4 = -1`

### Влияние параметров

| Параметр | Влияние |
|----------|---------|
| `lr` большой | Быстрая сходимость, может осциллировать |
| `lr` маленький | Медленная сходимость, стабильно |
| `steps` больше | Выше точность |
| `a` больше | Круче парабола, быстрее сходимость |

### Тесты
12 тестов проверяют примеры из задачи, сходимость к аналитическому минимуму, разные коэффициенты, влияние learning rate и что константа `c` не влияет на положение минимума.

---

## 📖 Resources

- [Gradient Descent — Wikipedia](https://en.wikipedia.org/wiki/Gradient_descent)
- [Gradient Descent — Coursera ML](https://www.coursera.org/learn/machine-learning/lecture/97YRl/gradient-descent)
- [Quadratic Function — Minimum](https://en.wikipedia.org/wiki/Quadratic_function)
