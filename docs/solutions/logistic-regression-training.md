# Logistic Regression Training Loop

**Задача #2** | [TensorTonic](https://www.tensortonic.com/problems/logistic-regression-training-loop)

---

## 📋 Требования

Реализовать обучение логистической регрессии через градиентный спуск.

**Модель:**
```
p = σ(Xw + b) = 1 / (1 + e^(-(Xw + b)))
```

**Функция потерь (Binary Cross-Entropy):**
```
L = -1/N * Σ[y_i*log(p_i) + (1-y_i)*log(1-p_i)]
```

**Ограничения:**
- Инициализация: w = zeros(D), b = 0.0
- Только градиентный спуск
- Возврат: `(w, b)` где w shape (D,), b — float
- Только NumPy (без sklearn)
- N ≤ 200, D ≤ 10
- Time limit: 1000 ms, Memory: 128 MB

---

## 🧪 Тестовые примеры

| Input | Output |
|-------|--------|
| `X=[[0],[1],[2],[3]]`<br>`y=[0,0,1,1]`<br>`lr=0.1, steps=500` | **Accuracy ≥ 95%** |

---

## 💡 Подсказки

### Градиенты

```
∇w = X^T(p - y) / N
∇b = mean(p - y)
```

### Обновление параметров

```
w ← w - lr * ∇w
b ← b - lr * ∇b
```

---

## 🔧 Решение

```python
import numpy as np


def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 
                    1/(1+np.exp(-z)), 
                    np.exp(z)/(1+np.exp(z)))


def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    
    Args:
        X: Input features, shape (N, D)
        y: Labels, shape (N,) with values 0 or 1
        lr: Learning rate (default 0.1)
        steps: Number of gradient descent steps (default 1000)
        
    Returns:
        Tuple (w, b) where:
            w: Weights, shape (D,)
            b: Bias, scalar float
    """
    # Convert inputs to numpy arrays
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    
    N, D = X.shape
    
    # Initialize weights and bias to zeros
    w = np.zeros(D)
    b = 0.0
    
    for _ in range(steps):
        # Forward pass: compute predictions
        z = X @ w + b           # Linear combination, shape (N,)
        p = _sigmoid(z)         # Predictions (probabilities), shape (N,)
        
        # Compute gradients
        error = p - y                   # shape (N,)
        dw = (X.T @ error) / N          # Gradient for weights, shape (D,)
        db = np.mean(error)             # Gradient for bias, scalar
        
        # Update parameters via gradient descent
        w = w - lr * dw
        b = b - lr * db
    
    return (w, b)
```

---

## 💡 Почему это работает

### 1. Численно стабильный Sigmoid

**Проблема:** При больших отрицательных `z`:
```python
np.exp(-1000)  # → Overflow!
```

**Решение:** Используем эквивалентную формулу:
```python
def _sigmoid(z):
    return np.where(z >= 0, 
                    1/(1+np.exp(-z)),      # Для z >= 0
                    np.exp(z)/(1+np.exp(z)))  # Для z < 0
```

| z | Формула | Результат |
|---|---------|-----------|
| `z = 10` | `1/(1+exp(-10))` | `0.99995` |
| `z = -10` | `exp(-10)/(1+exp(-10))` | `0.000045` |
| `z = -1000` | `exp(-1000)/(1+exp(-1000))` | `~0` (без overflow!) |

---

### 2. Вывод градиентов

**Функция потерь:**
```
L = -1/N * Σ[y_i*log(p_i) + (1-y_i)*log(1-p_i)]
```

**Производная sigmoid:**
```
σ'(z) = σ(z) * (1 - σ(z)) = p * (1 - p)
```

**Градиент по w:**
```
∂L/∂w = ∂L/∂p * ∂p/∂z * ∂z/∂w
      = (p - y) * X
      = X^T(p - y) / N
```

**Градиент по b:**
```
∂L/∂b = mean(p - y)
```

---

### 3. Векторизованная реализация

| Операция | Код | Shape |
|----------|-----|-------|
| Linear combination | `z = X @ w + b` | `(N,)` |
| Predictions | `p = _sigmoid(z)` | `(N,)` |
| Error | `error = p - y` | `(N,)` |
| Gradient w | `dw = (X.T @ error) / N` | `(D,)` |
| Gradient b | `db = np.mean(error)` | `scalar` |
| Update w | `w = w - lr * dw` | `(D,)` |
| Update b | `b = b - lr * db` | `scalar` |

**Преимущества:**
- ✅ Нет Python циклов по N (быстро)
- ✅ NumPy использует оптимизированные BLAS операции
- ✅ Читаемый код

---

## ✅ Тесты

### Проверка на примере из задачи

```python
X = [[0], [1], [2], [3]]
y = [0, 0, 1, 1]

w, b = train_logistic_regression(X, y, lr=0.1, steps=500)

# Проверка accuracy
X_arr = np.asarray(X)
z = X_arr @ w + b
predictions = (z >= 0).astype(int)
accuracy = np.mean(predictions == y)

assert accuracy >= 0.95  # ✅ 100%
```

### Проверка типов возврата

```python
assert isinstance(w, np.ndarray)
assert isinstance(b, float)
assert w.shape == (D,)
```

### Проверка многомерных данных

```python
X = [[0, 0], [1, 0], [0, 1], [2, 2], [3, 2], [2, 3]]
y = [0, 0, 0, 1, 1, 1]

w, b = train_logistic_regression(X, y, lr=0.1, steps=1000)

assert w.shape == (2,)  # ✅ 2 признака
```

### Численная стабильность

```python
X = [[-100], [100], [0]]
y = [0, 1, 0.5]

w, b = train_logistic_regression(X, y, lr=0.1, steps=100)

assert not np.any(np.isnan(w))  # ✅ Нет NaN
assert not np.any(np.isinf(w))  # ✅ Нет Inf
```

---

## 📊 Визуализация обучения

### Decision Boundary

```
Класс 1:           ●
                  ●
Класс 0:    ○
           ○
           
Decision boundary: X @ w + b = 0
```

### Loss Curve

```
Loss
0.72 |●
     | \
     |  \
     |   \
     |    ●─────
0.00 |──────────────
     0   500   1000  Epoch
```

---

## 🔑 Ключевые моменты

| Аспект | Решение |
|--------|---------|
| **Инициализация** | `w = np.zeros(D)`, `b = 0.0` |
| **Sigmoid** | Численно стабильный через `np.where` |
| **Градиент w** | `(X.T @ (p - y)) / N` |
| **Градиент b** | `mean(p - y)` |
| **Обновление** | `w -= lr * dw`, `b -= lr * db` |
| **Сходимость** | 500-1000 шагов для N ≤ 200 |
| **Сложность** | O(steps × N × D) |

---

## 📖 Дополнительные материалы

- [Logistic Regression — Wikipedia](https://en.wikipedia.org/wiki/Logistic_regression)
- [Gradient Descent — Coursera](https://www.coursera.org/learn/machine-learning/lecture/54GQ6/gradient-descent-for-logistic-regression)
- [Binary Cross-Entropy Loss](https://en.wikipedia.org/wiki/Cross_entropy#Logistic_regression)
- [NumPy: np.where](https://numpy.org/doc/stable/reference/generated/numpy.where.html)
