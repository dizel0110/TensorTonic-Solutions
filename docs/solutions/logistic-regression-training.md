# Logistic Regression Training Loop

**Problem #2** | [TensorTonic](https://www.tensortonic.com/problems/logistic-regression-training-loop)

> 🇷🇺 [Русская версия ниже](#russian-summary)

---

## 📋 Requirements

Train binary logistic regression via gradient descent.

**Model:**
```
p = σ(Xw + b) = 1 / (1 + e^(-(Xw + b)))
```

**Loss (Binary Cross-Entropy):**
```
L = -1/N * Σ[y_i*log(p_i) + (1-y_i)*log(1-p_i)]
```

**Constraints:**
- Initialize: w = zeros(D), b = 0.0
- Gradient descent only
- Return: `(w, b)` where w shape (D,), b is float
- NumPy only (no sklearn)
- N ≤ 200, D ≤ 10
- Time limit: 1000 ms, Memory: 128 MB

---

## 🧪 Test Examples

| Input | Output |
|-------|--------|
| `X=[[0],[1],[2],[3]]`<br>`y=[0,0,1,1]`<br>`lr=0.1, steps=500` | **Accuracy ≥ 95%** |

---

## 💡 Hints

**Gradients:**
```
∇w = X^T(p - y) / N
∇b = mean(p - y)
```

**Update:**
```
w ← w - lr * ∇w
b ← b - lr * ∇b
```

---

## 🔧 Solution

```python
import numpy as np


def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 
                    1/(1+np.exp(-z)), 
                    np.exp(z)/(1+np.exp(z)))


def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """Train logistic regression via gradient descent."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    
    N, D = X.shape
    w = np.zeros(D)
    b = 0.0
    
    for _ in range(steps):
        # Forward pass
        z = X @ w + b
        p = _sigmoid(z)
        
        # Compute gradients
        error = p - y
        dw = (X.T @ error) / N
        db = np.mean(error)
        
        # Update parameters
        w = w - lr * dw
        b = b - lr * db
    
    return (w, b)
```

### Key Implementation Details

| Component | Code | Purpose |
|-----------|------|---------|
| **Stable sigmoid** | `np.where(z >= 0, ...)` | Avoids overflow for negative z |
| **Forward pass** | `z = X @ w + b` | Linear combination |
| **Gradient w** | `(X.T @ error) / N` | Vectorized weight gradient |
| **Gradient b** | `np.mean(error)` | Bias gradient |
| **Update** | `w -= lr * dw` | Gradient descent step |

---

## 📊 Training Visualization

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

### Decision Boundary
```
Class 1:     ●
            ●
Class 0: ○
        ○
        
Boundary: X @ w + b = 0
```

---

## ✅ Tests

All 8 tests pass:

| Test | Checks |
|------|--------|
| `test_basic_linearly_separable` | Example from task, accuracy ≥ 95% |
| `test_2d_features` | Multi-dimensional features |
| `test_return_types` | w=np.ndarray, b=float |
| `test_zero_initialization` | Deterministic results |
| `test_gradient_computation` | Loss decreases |
| `test_learning_rate_effect` | Different lr values |
| `test_single_sample_per_class` | Minimal data separation |
| `test_numerical_stability` | No NaN/Inf |

---

<a name="russian-summary"></a>
## 🇷🇺 Краткое резюме (Russian Summary)

### Требования
Обучение логистической регрессии через градиентный спуск. Инициализация нулями. Возврат `(w, b)`. Только NumPy.

### Математика
- **Модель:** `p = σ(Xw + b)`
- **Loss:** `L = -1/N * Σ[y_i*log(p_i) + (1-y_i)*log(1-p_i)]`
- **Градиенты:** `∇w = X^T(p-y)/N`, `∇b = mean(p-y)`

### Решение
```python
# Инициализация
w = np.zeros(D)
b = 0.0

# Цикл градиентного спуска
for _ in range(steps):
    p = _sigmoid(X @ w + b)     # Предсказания
    error = p - y                # Ошибка
    w -= lr * (X.T @ error) / N  # Обновление весов
    b -= lr * np.mean(error)     # Обновление bias
```

### Численно стабильный sigmoid
```python
np.where(z >= 0, 
         1/(1+np.exp(-z)),      # Для z >= 0
         np.exp(z)/(1+np.exp(z)))  # Для z < 0 (без overflow)
```

### Тесты
8 тестов проверяют точность ≥95%, типы возврата, многомерные данные, численную стабильность.

---

## 📖 Resources

- [Logistic Regression — Wikipedia](https://en.wikipedia.org/wiki/Logistic_regression)
- [Gradient Descent](https://www.coursera.org/learn/machine-learning/lecture/54GQ6/gradient-descent-for-logistic-regression)
- [NumPy: np.where](https://numpy.org/doc/stable/reference/generated/numpy.where.html)
