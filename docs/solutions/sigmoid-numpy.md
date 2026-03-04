# Implement Sigmoid in NumPy

**Задача #1** | [TensorTonic](https://www.tensortonic.com/problems/sigmoid-numpy)

---

## 📋 Требования

Реализовать сигмоидную функцию:

```
σ(x) = 1 / (1 + e^(-x))
```

**Ограничения:**
- Только векторизованная реализация (без Python циклов)
- Только библиотека NumPy
- Работает со скалярами, списками и массивами
- Возвращает `np.ndarray` float
- Time limit: 200 ms, Memory: 64 MB

---

## 🧪 Тестовые примеры

| Input | Output |
|-------|--------|
| `x = 0` | `0.5` |
| `x = [0, 2, -2]` | `[0.5, 0.88079708, 0.11920292]` |
| `x = [[-1, 0], [1, 2]]` | `[[0.26894142, 0.5], [0.73105858, 0.88079708]]` |

---

## 💡 Решение

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

### Почему это работает

**1. Конвертация входа:**
```python
x = np.asarray(x, dtype=float)
```
- `np.asarray()` универсален: работает со скалярами, списками, массивами
- `dtype=float` гарантирует float64 для всех элементов
- Скаляр `0` → `array(0.)` (0-d массив)
- Список `[0, 1]` → `array([0., 1.])` (1-d массив)
- 2D список `[[0], [1]]` → `array([[0.], [1.]])` (2-d массив)

**2. Векторизованная формула:**
```python
result = 1 / (1 + np.exp(-x))
```
- `np.exp(-x)` применяется ко всем элементам сразу (векторизовано)
- `-x` меняет знак каждого элемента
- `1 + ...` добавляет 1 к каждому элементу
- `1 / ...` делит 1 на каждый элемент

**3. Гарантия возврата ndarray:**
```python
return np.asarray(result)
```
Без этого для скаляра `sigmoid(0)` вернётся `numpy.float64`, а не `np.ndarray`.

---

## ✅ Тесты

### Проверка скаляра
```python
assert sigmoid(0) == 0.5
assert isinstance(sigmoid(0), np.ndarray)
```

### Проверка списка
```python
result = sigmoid([0, 2, -2])
expected = [0.5, 0.88079708, 0.11920292]
assert np.allclose(result, expected)
```

### Проверка 2D массива
```python
result = sigmoid([[-1, 0], [1, 2]])
assert result.shape == (2, 2)
assert np.allclose(result, [[0.26894142, 0.5], [0.73105858, 0.88079708]])
```

### Граничные значения
```python
assert np.isclose(sigmoid(10), 1.0, atol=1e-4)   # ~1.0
assert np.isclose(sigmoid(-10), 0.0, atol=1e-4)  # ~0.0
```

### Векторизация
```python
x = np.linspace(-5, 5, 1000)
result = sigmoid(x)
assert result.shape == (1000,)
assert all(0 < val < 1 for val in result)
```

---

## 📊 График сигмоиды

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

**Свойства:**
- σ(0) = 0.5 (центр)
- σ(x) → 1 при x → +∞
- σ(x) → 0 при x → -∞
- Диапазон: (0, 1)
- Монотонно возрастает

---

## 🔑 Ключевые моменты

| Аспект | Решение |
|--------|---------|
| **Входные данные** | `np.asarray(x, dtype=float)` |
| **Формула** | `1 / (1 + np.exp(-x))` |
| **Векторизация** | Все операции NumPy векторизованы |
| **Выходной тип** | `np.asarray()` гарантирует ndarray |
| **Сложность** | O(n) — один проход по данным |

---

## 📖 Дополнительные материалы

- [NumPy: np.asarray](https://numpy.org/doc/stable/reference/generated/numpy.asarray.html)
- [NumPy: Broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html)
- [Sigmoid Function — Wikipedia](https://en.wikipedia.org/wiki/Sigmoid_function)
