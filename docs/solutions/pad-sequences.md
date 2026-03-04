# Pad Sequences

**Problem #3** | [TensorTonic](https://www.tensortonic.com/problems/pad-sequences)

> 🇷🇺 [Русская версия ниже](#russian-summary)

---

## 📋 Requirements

Pad variable-length sequences to equal length for NLP batch processing.

**Task:**
- Given: List of token ID sequences (lists of ints)
- Pad shorter sequences with `pad_value`
- Truncate longer sequences to `max_len`
- Return: NumPy array of shape (N, L)

**Constraints:**
- If `max_len` is None, use maximum length among sequences
- Pad at the end (right padding)
- Truncate at the end if sequence > max_len
- Empty input → shape (0, 0)
- Output dtype: `int`
- NumPy required
- N ≤ 10,000 sequences, each length ≤ 1,000

---

## 🧪 Test Examples

| Input | Output |
|-------|--------|
| `seqs = [[1,2,3], [4,5], [6]]`<br>`pad_value=0` | `[[1,2,3], [4,5,0], [6,0,0]]` |
| `seqs = [[1,2,3,4], [5,6]]`<br>`pad_value=-1, max_len=3` | `[[1,2,3], [5,6,-1]]` (truncated) |

---

## 💡 Hints

1. If `max_len` is None, compute as `max(len(seq) for seq in seqs)`
2. Use `np.full()` to initialize result, then copy each sequence

---

## 🔧 Solution

```python
import numpy as np


def pad_sequences(seqs, pad_value=0, max_len=None):
    """
    Pad sequences to equal length.
    
    Args:
        seqs: List of sequences (lists of ints)
        pad_value: Value for padding (default 0)
        max_len: Maximum length (default: max length in seqs)
        
    Returns:
        np.ndarray of shape (N, L) where:
            N = len(seqs)
            L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
    # Handle empty input
    if len(seqs) == 0:
        return np.array([], dtype=int).reshape(0, 0)
    
    # Compute max_len if not provided
    if max_len is None:
        max_len = max(len(seq) for seq in seqs) if seqs else 0
    
    # Handle case where max_len is 0
    if max_len == 0:
        return np.array([], dtype=int).reshape(len(seqs), 0)
    
    N = len(seqs)
    
    # Initialize result array with pad_value
    result = np.full((N, max_len), pad_value, dtype=int)
    
    # Copy each sequence (with truncation if needed)
    for i, seq in enumerate(seqs):
        seq_len = min(len(seq), max_len)
        if seq_len > 0:
            result[i, :seq_len] = seq[:seq_len]
    
    return result
```

### Key Implementation Details

| Step | Code | Purpose |
|------|------|---------|
| 1. Empty check | `if len(seqs) == 0` | Return (0, 0) array |
| 2. Compute max_len | `max(len(seq) for seq in seqs)` | Auto-detect length |
| 3. Initialize | `np.full((N, max_len), pad_value, dtype=int)` | Pre-allocate with padding |
| 4. Copy sequences | `result[i, :seq_len] = seq[:seq_len]` | Copy with truncation |

---

## 📊 Visualization

### Input (Variable Length)
```
[1, 5, 3, 8, 2]     len=5
[4, 7, 9]           len=3
[2, 6]              len=2
[3, 1, 4, 1, 5, 9]  len=6
[7, 1, 2]           len=3
```

### Output (Padded to max_len=6)
```
[1, 5, 3, 8, 2, 0]  ← padded with 0
[4, 7, 9, 0, 0, 0]  ← padded with 0
[2, 6, 0, 0, 0, 0]  ← padded with 0
[3, 1, 4, 1, 5, 9]  ← no padding needed
[7, 1, 2, 0, 0, 0]  ← padded with 0
```

**Output shape:** (5, 6)

---

## ✅ Tests

All 11 tests pass:

| Test | Checks |
|------|--------|
| `test_example_1` | Basic padding (example 1) |
| `test_example_2_with_max_len` | Padding + truncation |
| `test_empty_input` | Empty list → (0, 0) |
| `test_all_same_length` | No padding needed |
| `test_custom_pad_value` | Custom pad value (-999) |
| `test_max_len_larger_than_all` | Extra padding |
| `test_truncation_all_sequences` | All truncated |
| `test_single_sequence` | Single sequence |
| `test_one_element_sequences` | Single element each |
| `test_large_batch` | 1000 sequences |
| `test_return_type` | np.ndarray with int dtype |

---

<a name="russian-summary"></a>
## 🇷🇺 Краткое резюме (Russian Summary)

### Требования
Даны последовательности токенов разной длины. Нужно привести их к одной длине:
- Более короткие — дополнить `pad_value` справа
- Более длинные — обрезать справа до `max_len`
- Пустой ввод → массив формы (0, 0)
- Возврат: `np.ndarray` dtype `int`

### Решение
```python
# 1. Обработка пустого ввода
if len(seqs) == 0:
    return np.array([], dtype=int).reshape(0, 0)

# 2. Вычисляем max_len если не задан
if max_len is None:
    max_len = max(len(seq) for seq in seqs)

# 3. Создаём массив с заполнением pad_value
result = np.full((N, max_len), pad_value, dtype=int)

# 4. Копируем последовательности с обрезкой
for i, seq in enumerate(seqs):
    seq_len = min(len(seq), max_len)
    result[i, :seq_len] = seq[:seq_len]
```

### Ключевые моменты
- `np.full()` — создаёт массив сразу с нужным значением padding
- `seq[:seq_len]` — обрезка если последовательность слишком длинная
- `dtype=int` — обязательное требование задачи
- Правый padding (дополнение в конец последовательности)

### Тесты
11 тестов проверяют примеры из задачи, пустой ввод, кастомные значения padding, обрезку и большие батчи (1000 последовательностей).

---

## 📖 Resources

- [NumPy: np.full](https://numpy.org/doc/stable/reference/generated/numpy.full.html)
- [NumPy: Array indexing](https://numpy.org/doc/stable/reference/arrays.indexing.html)
- [NLP Padding — Common Practice](https://neptune.ai/blog/padding-in-nlp)
