# Matrix Transpose

**Problem #4** | [TensorTonic](https://www.tensortonic.com/problems/matrix-transpose)

> 🇷🇺 [Русская версия ниже](#russian-summary)

---

## 📋 Requirements

Implement matrix transpose where element at position (i, j) moves to (j, i).

**Mathematical Definition:**
```
(A^T)[j, i] = A[i, j]
```

An n×m matrix becomes an m×n matrix.

**Constraints:**
- Return new NumPy array of shape (M, N)
- Must NOT modify original matrix
- Must work for non-square (rectangular) matrices
- **Do NOT use `.T` or `np.transpose()`**
- Use manual indexing with loops or array operations
- 1 ≤ N, M ≤ 1000
- Time limit: 200 ms

---

## 🧪 Test Examples

| Input | Output |
|-------|--------|
| `A = [[1, 2, 3], [4, 5, 6]]` | `[[1, 4], [2, 5], [3, 6]]` |
| `A = [[1, 2], [3, 4]]` | `[[1, 3], [2, 4]]` |
| `A = [[1, 2, 3, 4]]` | `[[1], [2], [3], [4]]` |

---

## 💡 Hint

Create a new array with shape (m, n) using `np.zeros()`, then fill it with a nested loop.

---

## 🔧 Solution

```python
import numpy as np


def matrix_transpose(A):
    """
    Return the transpose of matrix A (swap rows and columns).
    
    Mathematical definition: (A^T)[j, i] = A[i, j]
    An n×m matrix becomes an m×n matrix.
    
    Args:
        A: 2D NumPy array, shape (N, M) - input matrix
        
    Returns:
        New NumPy array of shape (M, N) - transposed matrix
    """
    A = np.asarray(A, dtype=float)
    
    N, M = A.shape
    
    # Create result array with swapped shape
    result = np.empty((M, N), dtype=float)
    
    # Manual indexing (requirement: no .T or np.transpose())
    for i in range(N):
        for j in range(M):
            result[j, i] = A[i, j]
    
    return result
```

### Key Implementation Details

| Step | Code | Purpose |
|------|------|---------|
| 1. Convert | `A = np.asarray(A, dtype=float)` | Ensure numpy array |
| 2. Get shape | `N, M = A.shape` | Original dimensions |
| 3. Create result | `np.empty((M, N))` | Swapped dimensions |
| 4. Fill manually | `result[j, i] = A[i, j]` | Transpose mapping |

---

## 📊 Visualization

### Original Matrix A (2×3)
```
[1, 2, 3]
[4, 5, 6]
```

### Transposed Matrix Aᵀ (3×2)
```
[1, 4]    ← Row 0: A[0,0], A[1,0]
[2, 5]    ← Row 1: A[0,1], A[1,1]
[3, 6]    ← Row 2: A[0,2], A[1,2]
```

### Mapping Logic
```
A[0,0]=1  →  Aᵀ[0,0]=1
A[0,1]=2  →  Aᵀ[1,0]=2
A[0,2]=3  →  Aᵀ[2,0]=3
A[1,0]=4  →  Aᵀ[0,1]=4
A[1,1]=5  →  Aᵀ[1,1]=5
A[1,2]=6  →  Aᵀ[2,1]=6
```

---

## ✅ Tests

All 11 tests pass:

| Test | Checks |
|------|--------|
| `test_example_1` | Example 1 (2×3 → 3×2) |
| `test_example_2_square` | Square matrix (2×2) |
| `test_example_3_row_vector` | Row → column vector |
| `test_non_square_matrix` | Rectangular matrix |
| `test_single_element` | 1×1 matrix |
| `test_column_vector` | Column → row vector |
| `test_float_values` | Float elements |
| `test_negative_values` | Negative elements |
| `test_does_not_modify_original` | Original unchanged |
| `test_return_type` | np.ndarray, float |
| `test_large_matrix` | 100×50 matrix |

---

<a name="russian-summary"></a>
## 🇷🇺 Краткое резюме (Russian Summary)

### Требования
Реализовать транспонирование матрицы: элемент на позиции (i, j) перемещается на (j, i). Матрица n×m становится m×n.

**Ограничения:**
- **НЕ использовать `.T` или `np.transpose()`**
- Не модифицировать исходную матрицу
- Работать с прямоугольными матрицами
- Вернуть новый `np.ndarray` shape (M, N)

### Решение
```python
def matrix_transpose(A):
    A = np.asarray(A, dtype=float)
    N, M = A.shape
    
    # Создаём результат с переставленными размерами
    result = np.empty((M, N), dtype=float)
    
    # Ручное заполнение (требование задачи)
    for i in range(N):
        for j in range(M):
            result[j, i] = A[i, j]
    
    return result
```

### Ключевые моменты
- **`np.empty((M, N))`** — создаёт массив нужной формы (быстрее чем `np.zeros()`)
- **`result[j, i] = A[i, j]`** — формула транспонирования
- **Двойной цикл** — требование задачи (ручная индексация)
- **`dtype=float`** — поддержка вещественных чисел

### Визуализация
```
Оригинал (2×3):        Транспонированная (3×2):
[1, 2, 3]              [1, 4]
[4, 5, 6]              [2, 5]
                       [3, 6]
                       
Строки → Столбцы
Столбцы → Строки
```

### Тесты
11 тестов проверяют примеры из задачи, квадратные/прямоугольные матрицы, векторы-строки/столбцы, отрицательные значения, и что оригинал не модифицируется.

---

## 📖 Resources

- [Matrix Transpose — Wikipedia](https://en.wikipedia.org/wiki/Transpose)
- [NumPy: np.empty](https://numpy.org/doc/stable/reference/generated/numpy.empty.html)
- [NumPy: Array indexing](https://numpy.org/doc/stable/reference/arrays.indexing.html)
