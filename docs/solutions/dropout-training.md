# Problem #8: Implement Dropout (Training Mode)

**Ссылка:** https://www.tensortonic.com/problems/dropout-training
**Папка решения:** `sandbox/dropout-training/`
**Файл решения:** `dropout-training/dropout-training.py`
**Файл тестов:** `dropout-training/test_dropout_training.py`
**Статус:** ✅ Готово

---

## 📋 Требования задачи

**Задача:**
- Реализовать dropout для обучения нейронных сетей
- Случайно обнулять элементы с вероятностью `p`
- Масштабировать оставшиеся элементы на `1 / (1 - p)` для сохранения ожидаемого значения

**Формула:**

```
output_i = { 0                    с вероятностью p
           { x_i * 1/(1-p)        с вероятностью (1-p)
```

**Ограничения:**
- Полностью векторизованная реализация (без циклов)
- Входной массив до размера (1000, 1000)
- `0.0 ≤ p < 1.0`
- Использовать `rng.random()` если `rng` предоставлен, иначе `np.random.random()`
- Только NumPy
- Time limit: 200 ms

**Возврат:**
- Кортеж `(output, dropout_pattern)`
- `dropout_pattern` показывает, какие элементы сохранены/удалены

---

## 🧪 Тестовые примеры из задачи

| Input | Output |
|-------|--------|
| `x=[1., 2., 3.], p=0.0` | `([1., 2., 3.], [1., 1., 1.])` |
| `x=[2., 4.], p=0.5` | `([0., 8.], [0., 2.])` или `([4., 0.], [2., 0.])` |

---

## 💡 Подсказки из задачи

1. **Генерация случайных чисел:** Создать случайные значения от 0 до 1. Если `random_value < (1-p)`, сохранить элемент и масштабировать, иначе обнулить.

2. **Dropout pattern:** Создать массив паттернов, где `0` — дропнутые элементы, `1/(1-p)` — сохранённые. Умножить вход на этот паттерн.

---

## 🔧 Решение

**Ключевые моменты реализации:**

```python
def dropout(x, p=0.5, rng=None):
    # 1. Конвертация входов в NumPy array
    x = np.asarray(x, dtype=float)

    # 2. Генерация случайных чисел для dropout mask
    if rng is not None:
        random_vals = rng.random(x.shape)
    else:
        random_vals = np.random.random(x.shape)

    # 3. Маска: 1 если элемент сохраняется (random >= p), 0 если дропается
    mask = (random_vals >= p).astype(float)

    # 4. Скалирование: 1 / (1 - p) для сохранения ожидаемого значения
    scale = 1.0 / (1.0 - p) if p < 1.0 else 1.0

    # 5. Dropout pattern: 0 для дропнутых, scale для сохранённых
    dropout_pattern = mask * scale

    # 6. Применяем dropout к входу
    output = x * dropout_pattern

    # 7. Гарантируем возврат np.ndarray
    return (np.asarray(output), np.asarray(dropout_pattern))
```

**Почему это работает:**

1. **Inverted Dropout:**
   - Традиционный dropout: обнуляем с вероятностью `p`, выход не масштабируем
   - Inverted dropout (используемый здесь): обнуляем с вероятностью `p`, но **масштабируем на `1/(1-p)`**
   - Преимущество: на инференсе dropout не нужен — выход уже имеет правильное ожидаемое значение

2. **Сохранение ожидаемого значения:**
   ```
   E[output] = p * 0 + (1-p) * (x * 1/(1-p)) = x
   ```
   - Математическое ожидание выхода равно входу
   - Суммарная активация слоя сохраняется

3. **Векторизация:**
   - `np.random.random(x.shape)` — генерирует случайные числа для всех элементов сразу
   - `(random_vals >= p).astype(float)` — векторизованное сравнение
   - `x * dropout_pattern` — поэлементное умножение

4. **rng параметр:**
   - Позволяет воспроизводимость тестов с фиксированным seed
   - Если не предоставлен — используется глобальный `np.random`

---

## ✅ Тесты

**Набор тестов в `dropout-training/test_dropout_training.py`:**

| Тест | Описание | Проверяет |
|------|----------|-----------|
| `test_p0_no_dropout` | `p=0.0` | Выход = вход |
| `test_p05_with_fixed_rng` | `p=0.5` с rng | Паттерн 0 или 2.0 |
| `test_scalar_input` | Скалярный вход | 0-d array возврат |
| `test_2d_array` | 2D массив | Форма (n, m) |
| `test_dropout_pattern_values` | Значения паттерна | Только 0 или scale |
| `test_expected_value_preservation` | Сохранение E[x] | Среднее ~ исходному |
| `test_high_dropout_rate` | `p=0.9` | scale = 10.0 |
| `test_low_dropout_rate` | `p=0.1` | scale = 1.111 |
| `test_return_types` | Типы возврата | Tuple из np.ndarray |
| `test_large_array_vectorized` | 1000×1000 | Векторизация |
| `test_negative_input_values` | Отрицательные x | Корректная обработка |
| `test_zero_input_values` | Нулевые x | Выход = 0 |
| `test_deterministic_with_same_seed` | Одинаковый seed | Детерминизм |
| `test_different_seeds_different_results` | Разные seeds | Разные результаты |

**Результат:** Все 14 тестов ✅

---

## 📊 Пример работы

```python
import numpy as np
from dropout_training import dropout

# Пример 1: Без dropout (p=0)
x = np.array([1., 2., 3.])
output, pattern = dropout(x, p=0.0)
print(f"p=0: output={output}, pattern={pattern}")
# output: [1. 2. 3.], pattern: [1. 1. 1.]

# Пример 2: С dropout 50% (с фиксированным seed)
rng = np.random.default_rng(seed=42)
output, pattern = dropout(x, p=0.5, rng=rng)
print(f"p=0.5: output={output}, pattern={pattern}")
# output: [0. 4. 6.], pattern: [0. 2. 2.] (пример)

# Пример 3: 2D массив
x_2d = np.array([[1., 2.], [3., 4.]])
output, pattern = dropout(x_2d, p=0.5, rng=np.random.default_rng(123))
print(f"2D output:\n{output}")
print(f"2D pattern:\n{pattern}")
```

---

## 📝 Примечания

**Зачем нужно масштабирование?**

| Без масштабирования | С масштабированием (inverted) |
|---------------------|-------------------------------|
| Train: `output = x * mask` | Train: `output = x * mask / (1-p)` |
| Infer: `output = x * (1-p)` | Infer: `output = x` (без изменений) |
| Нужно масштабировать на инференсе | Инференс без изменений |

**Почему inverted dropout лучше:**
- ✅ Не нужно менять код на инференсе
- ✅ Выход всегда имеет правильное ожидаемое значение
- ✅ Используется в современных фреймворках (PyTorch, TensorFlow)

**Типичные значения dropout rate:**

| Слой | Рекомендуемый p |
|------|-----------------|
| Input layer | 0.0 - 0.2 |
| Hidden layers | 0.3 - 0.5 |
| Large networks | 0.5 - 0.7 |

**Когда использовать dropout:**
- ✅ Большие нейронные сети (переобучение)
- ✅ Мало данных для обучения
- ✅ Полносвязные слои (fully connected)

**Когда НЕ использовать:**
- ❌ Сверточные сети (лучше BatchNorm)
- ❌ RNN/LSTM (специальные варианты dropout)
- ❌ Маленькие сети (не нужно регуляризировать)

---

## 📚 Ссылки

- [Dropout Paper: "Dropout: A Simple Way to Prevent Neural Networks from Overfitting"](https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf)
- [TensorTonic Problem](https://www.tensortonic.com/problems/dropout-training)
- [PyTorch Dropout Documentation](https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html)
