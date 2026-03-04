# Implement Positional Encoding (sin/cos)

**Problem #5** | [TensorTonic](https://www.tensortonic.com/problems/positional-encoding)

> 🇷🇺 [Русская версия ниже](#russian-summary)

---

## 📋 Requirements

Implement sinusoidal positional encodings as described in "Attention Is All You Need" to inject sequence order into token embeddings.

**Mathematical Definition:**

For position `pos` and dimension index `i`:

```
PE(pos, 2i)   = sin(pos / base^(2i/d_model))
PE(pos, 2i+1) = cos(pos / base^(2i/d_model))
```

- Even-indexed columns use **sine**
- Odd-indexed columns use **cosine**
- Frequency decreases with dimension index

**Constraints:**
- Fully vectorized (no Python loops over positions or dimensions)
- Support any `seq_len ≥ 1` and `d_model ≥ 1`
- For odd `d_model`, the last column should be **sin**
- Return `dtype=float` for stable downstream use
- NumPy only
- Sequence lengths up to 10,000
- Model dimensions up to 2,048
- Time limit: 300 ms

---

## 🧪 Test Examples

### Example 1: seq_len=3, d_model=4

**Input:**
```python
positional_encoding(3, 4)
```

**Output (approx):**
```
[[0.0000, 1.0000, 0.0000, 1.0000],   # pos=0: sin(0)=0, cos(0)=1
 [0.8415, 0.5403, 0.0100, 0.9999],   # pos=1
 [0.9093, -0.4161, 0.0200, 0.9998]]  # pos=2
```

**Columns:** `[sin(i=0), cos(i=0), sin(i=1), cos(i=1)]`

For `i=1`: divisor = `10000^(2/4) = 100`, so angles are small (~0.01, 0.02)

### Example 2: seq_len=2, d_model=3 (odd)

**Input:**
```python
positional_encoding(2, 3)
```

**Output (approx):**
```
[[0.0000, 1.0000, 0.0000],    # pos=0
 [0.8415, 0.5403, 0.0022]]    # pos=1
```

**Columns:** `[sin(i=0), cos(i=0), sin(i=1)]`

Odd `d_model` → last column is **sin** without paired cos.

---

## 💡 Hints

1. Build a column vector of positions `(T, 1)` and a row vector of frequencies `(1, ⌈d/2⌉)` for broadcasting.

2. Use `pe[:, 0::2] = sin(...)` and `pe[:, 1::2] = cos(...)` to fill alternating columns.

---

## 🔧 Solution

```python
import numpy as np


def positional_encoding(seq_len, d_model, base=10000.0):
    """
    Return PE of shape (seq_len, d_model) using sin/cos formulation.
    
    Mathematical definition (Attention Is All You Need):
    PE(pos, 2i) = sin(pos / base^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / base^(2i/d_model))
    
    Args:
        seq_len: Length of sequence (T)
        d_model: Model dimension (d)
        base: Base for frequency calculation (default 10000.0)
        
    Returns:
        np.ndarray of shape (seq_len, d_model), dtype=float
    """
    # Create position vector: shape (seq_len, 1)
    positions = np.arange(seq_len, dtype=float).reshape(-1, 1)
    
    # Create frequency vector for i values: shape (ceil(d_model/2),)
    num_pairs = (d_model + 1) // 2  # ceil(d_model / 2)
    i = np.arange(num_pairs, dtype=float)
    
    # Compute divisor: base^(2i/d_model)
    # This creates decreasing frequencies for higher dimensions
    divisors = base ** (2 * i / d_model)
    
    # Compute angles: pos / divisor
    # Broadcasting: (seq_len, 1) / (num_pairs,) -> (seq_len, num_pairs)
    angles = positions / divisors
    
    # Compute sin and cos
    sin_vals = np.sin(angles)  # (seq_len, num_pairs)
    cos_vals = np.cos(angles)  # (seq_len, num_pairs)
    
    # Stack sin and cos alternately
    pe = np.empty((seq_len, 2 * num_pairs), dtype=float)
    pe[:, 0::2] = sin_vals  # Even columns: sin
    pe[:, 1::2] = cos_vals  # Odd columns: cos
    
    # If d_model is odd, trim to exact d_model
    # (last column remains sin, as required)
    if pe.shape[1] > d_model:
        pe = pe[:, :d_model]
    
    return pe
```

### Key Implementation Details

| Step | Code | Purpose |
|------|------|---------|
| 1. Position vector | `np.arange(seq_len).reshape(-1, 1)` | Shape `(T, 1)` for broadcasting |
| 2. Frequency indices | `np.arange(ceil(d/2))` | Column pair indices `i` |
| 3. Divisors | `base ** (2*i / d_model)` | Decreasing frequencies |
| 4. Angles | `positions / divisors` | Broadcasting to `(T, d/2)` |
| 5. Sin/Cos | `np.sin(angles)`, `np.cos(angles)` | Vectorized computation |
| 6. Interleave | `pe[:, 0::2] = sin`, `pe[:, 1::2] = cos` | Alternating columns |
| 7. Trim odd | `pe[:, :d_model]` | Handle odd `d_model` |

---

## 📊 Visualization

### Positional Encoding Heatmap (seq_len=50, d_model=32)

```
Dimension →
  0    8    16   24   32
┌─────────────────────────┐ 0
│ █░░ █░░ ░░░ ░░░ ░░░     │
│ ░█░ ░█░ ░░░ ░░░ ░░░     │ 10
│ ░░█ ░░█ ░█░ ░░░ ░░░     │ 20
│ ░░░ ░░░ ░█░ ░█░ ░░░     │ 30
│ ░░░ ░░░ ░░█ ░░█ ░█░     │ 40
│ ░░░ ░░░ ░░░ ░░░ ░░█     │ 49
└─────────────────────────┘

█ = High magnitude (sin/cos peak)
░ = Low magnitude
```

**Frequency Decay:**
- **Dim 0** (high freq): Rapid oscillations
- **Dim 31** (low freq): Nearly constant

### Formula Visualization

```
Position 0: [sin(0), cos(0), sin(0), cos(0), ...]
           = [0,     1,      0,     1,      ...]

Position 1: [sin(θ₀), cos(θ₀), sin(θ₁), cos(θ₁), ...]
           where θᵢ = 1 / 10000^(2i/d)

Position 2: [sin(2θ₀), cos(2θ₀), sin(2θ₁), cos(2θ₁), ...]
```

---

## ✅ Tests

All 12 tests pass:

| Test | Checks |
|------|--------|
| `test_example_1` | Example 1 (3×4 matrix) |
| `test_example_2_odd_d_model` | Odd d_model (last=sin) |
| `test_position_zero` | PE(0) = [0,1,0,1,...] |
| `test_even_odd_columns` | sin²+cos²=1 verification |
| `test_frequency_decay` | Lower dims vary more |
| `test_single_position` | seq_len=1 |
| `test_single_dimension` | d_model=1 |
| `test_large_sequence` | 1000 positions |
| `test_large_d_model` | 512 dimensions |
| `test_custom_base` | Different base values |
| `test_return_type` | np.ndarray, float |
| `test_vectorized_no_loops` | Performance (<500ms) |

---

<a name="russian-summary"></a>
## 🇷🇺 Краткое резюме (Russian Summary)

### Требования
Реализовать синусоидальные positional encoding из статьи "Attention Is All You Need" для кодирования порядка токенов в последовательности.

**Формула:**
```
PE(pos, 2i)   = sin(pos / base^(2i/d_model))
PE(pos, 2i+1) = cos(pos / base^(2i/d_model))
```

**Ограничения:**
- Полностью векторизованная реализация (без циклов)
- Чётные столбцы — sin, нечётные — cos
- Для нечётного `d_model` последний столбец — sin
- Time limit: 300 ms, NumPy only

### Решение

```python
def positional_encoding(seq_len, d_model, base=10000.0):
    # 1. Вектор позиций (T, 1)
    positions = np.arange(seq_len).reshape(-1, 1)
    
    # 2. Индексы пар столбцов
    num_pairs = (d_model + 1) // 2
    i = np.arange(num_pairs)
    
    # 3. Дивизоры для частот: base^(2i/d_model)
    divisors = base ** (2 * i / d_model)
    
    # 4. Углы через broadcasting
    angles = positions / divisors
    
    # 5. Sin и cos
    sin_vals = np.sin(angles)
    cos_vals = np.cos(angles)
    
    # 6. Чередование столбцов
    pe = np.empty((seq_len, 2*num_pairs))
    pe[:, 0::2] = sin_vals  # Чётные: sin
    pe[:, 1::2] = cos_vals  # Нечётные: cos
    
    # 7. Обрезка для нечётного d_model
    if pe.shape[1] > d_model:
        pe = pe[:, :d_model]
    
    return pe
```

### Ключевые моменты

1. **Broadcasting:** `(T, 1) / (num_pairs,)` → `(T, num_pairs)`
2. **Затухание частот:** `base^(2i/d)` — чем больше `i`, тем меньше частота
3. **Чередование:** `pe[:, 0::2]` и `pe[:, 1::2]` — срезы через один
4. **Нечётный d_model:** Обрезаем лишний cos-столбец, остаётся sin

### Визуализация

```
Позиция 0: [0, 1, 0, 1, 0, 1, ...]  (sin(0)=0, cos(0)=1)
Позиция 1: [0.84, 0.54, 0.01, 0.99, ...]
Позиция 2: [0.91, -0.42, 0.02, 0.99, ...]

Столбцы: [sin₀, cos₀, sin₁, cos₁, sin₂, cos₂, ...]
```

### Тесты
12 тестов проверяют примеры из задачи, чётность/нечётность столбцов, затухание частот, одиночные позиции/измерения, большие размеры (10000×2048) и производительность.

---

## 📖 Resources

- ["Attention Is All You Need" (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762)
- [Positional Encoding — Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- [NumPy: Broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html)
