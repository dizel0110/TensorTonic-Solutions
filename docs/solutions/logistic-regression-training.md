# Logistic Regression Training Loop

**Problem #2** | [TensorTonic](https://www.tensortonic.com/problems/logistic-regression-training-loop)

---

## 🌐 Language / Язык

- **[🇬🇧 English](#english)**
- **[🇷🇺 Русский](#русский)**

---

<a name="english"></a>
## 🇬🇧 English Version

### 📋 Requirements

Implement binary logistic regression training using gradient descent.

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

### 🧪 Test Examples

| Input | Output |
|-------|--------|
| `X=[[0],[1],[2],[3]]`<br>`y=[0,0,1,1]`<br>`lr=0.1, steps=500` | **Accuracy ≥ 95%** |

### 💡 Hints

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

### 🔧 Solution

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
        z = X @ w + b
        p = _sigmoid(z)
        
        error = p - y
        dw = (X.T @ error) / N
        db = np.mean(error)
        
        w = w - lr * dw
        b = b - lr * db
    
    return (w, b)
```

### ✅ Tests

All 8 tests pass with 100% accuracy on linearly separable data.

---

<a name="русский"></a>
## 🇷🇺 Русская версия

### 📋 Требования

Реализовать обучение логистической регрессии через градиентный спуск.

**Модель:**
```
p = σ(Xw + b) = 1 / (1 + e^(-(Xw + b)))
```

**Функция потерь:**
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

### 🧪 Тестовые примеры

| Вход | Выход |
|------|-------|
| `X=[[0],[1],[2],[3]]`<br>`y=[0,0,1,1]`<br>`lr=0.1, steps=500` | **Accuracy ≥ 95%** |

### 💡 Подсказки

**Градиенты:**
```
∇w = X^T(p - y) / N
∇b = mean(p - y)
```

**Обновление:**
```
w ← w - lr * ∇w
b ← b - lr * ∇b
```

### 🔧 Решение

Код идентичен английской версии выше.

### ✅ Тесты

Все 8 тестов проходят с точностью 100% на линейно разделимых данных.

---

## 📊 Visual / Визуализация

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

### Decision Boundary / Граница решений

```
Class 1:     ●
            ●
Class 0: ○
        ○
        
Boundary: X @ w + b = 0
```

---

## 🔑 Key Points / Ключевые моменты

| Aspect / Аспект | Solution / Решение |
|-----------------|-------------------|
| **Initialization** | `w = np.zeros(D)`, `b = 0.0` |
| **Sigmoid** | Numerically stable via `np.where` |
| **Gradient w** | `(X.T @ (p - y)) / N` |
| **Gradient b** | `mean(p - y)` |
| **Update** | `w -= lr * dw`, `b -= lr * db` |

---

## 📖 Resources / Ресурсы

- [Logistic Regression — Wikipedia](https://en.wikipedia.org/wiki/Logistic_regression)
- [Gradient Descent — Coursera](https://www.coursera.org/learn/machine-learning/lecture/54GQ6/gradient-descent-for-logistic-regression)
- [NumPy: np.where](https://numpy.org/doc/stable/reference/generated/numpy.where.html)
