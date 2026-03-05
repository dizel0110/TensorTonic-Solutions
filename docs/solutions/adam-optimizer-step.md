# Problem #7: Implement Adam Optimizer Step

**Ссылка:** https://www.tensortonic.com/problems/adam-optimizer
**Папка решения:** `adam-optimizer/`
**Файл решения:** `adam-optimizer/adam-optimizer.py`
**Файл тестов:** `adam-optimizer/test_adam_optimizer.py`
**Статус:** ✅ Готово

---

## 📋 Требования задачи

**Задача:**
- Реализовать один шаг оптимизатора Adam (Adaptive Moment Estimation)
- Даны: параметр(ы), градиент(ы), моменты m и v, номер шага t
- Вернуть: обновлённый параметр и новые моменты

**Формулы обновления:**

```
m_t = β1 · m_{t-1} + (1 - β1) · g_t           # 1st moment (momentum)
v_t = β2 · v_{t-1} + (1 - β2) · g_t²          # 2nd moment (squared gradient)
m̂_t = m_t / (1 - β1^t)                        # Bias correction 1
v̂_t = v_t / (1 - β2^t)                        # Bias correction 2
θ_t = θ_{t-1} - lr · m̂_t / (√v̂_t + ε)         # Parameter update
```

**Ограничения:**
- Принимать скаляры и NumPy массивы
- Векторизованные операции (без циклов)
- Bias correction с использованием t (1-based)
- Возврат: `(param_new, m_new, v_new)` с теми же формами
- Без внешних ML библиотек (только NumPy)
- Размер входа до ~10⁵ элементов
- Time limit: 500 ms, Memory: 128 MB

**Гиперпараметры по умолчанию:**
- `lr = 1e-3` (learning rate)
- `beta1 = 0.9` (decay rate for 1st moment)
- `beta2 = 0.999` (decay rate for 2nd moment)
- `eps = 1e-8` (численная стабильность)

---

## 🧪 Тестовые примеры из задачи

| Input | Output |
|-------|--------|
| `grad=0` | `param_new == param` (no change) |
| `t=1` | Significant bias correction `(1 - beta1**1, 1 - beta2**1)` |

---

## 💡 Подсказки из задачи

1. **Update m and v first** — сначала обновить m и v, затем вычислять bias-corrected m̂ и v̂
2. **Use 1-based t** — использовать t в bias correction `(1 - β^t)`, добавить ε в знаменатель для численной стабильности

---

## 🔧 Решение

**Ключевые моменты реализации:**

```python
def adam_step(param, grad, m, v, t, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
    # 1. Конвертация входов в NumPy array для векторизации
    param = np.asarray(param, dtype=float)
    grad = np.asarray(grad, dtype=float)
    m = np.asarray(m, dtype=float)
    v = np.asarray(v, dtype=float)

    # 2. Обновление первого момента (momentum)
    m_new = beta1 * m + (1 - beta1) * grad

    # 3. Обновление второго момента (squared gradient)
    v_new = beta2 * v + (1 - beta2) * (grad ** 2)

    # 4. Bias correction для первого момента
    m_hat = m_new / (1 - beta1 ** t)

    # 5. Bias correction для второго момента
    v_hat = v_new / (1 - beta2 ** t)

    # 6. Обновление параметра
    param_new = param - lr * m_hat / (np.sqrt(v_hat) + eps)

    return (param_new, m_new, v_new)
```

**Почему это работает:**

1. **Первый момент (m):**
   - Экспоненциально затухающее среднее градиентов
   - `beta1 * m` — сохраняем историю
   - `(1 - beta1) * grad` — добавляем текущий градиент
   - Работает как momentum — накапливает направление

2. **Второй момент (v):**
   - Экспоненциально затухающее среднее квадратов градиентов
   - Оценивает дисперсию градиентов
   - Большие градиенты → большое v → меньшее обновление
   - Адаптивный learning rate для каждого параметра

3. **Bias correction:**
   - При t=1: `m_hat = m_new / (1 - 0.9) = m_new / 0.1 = 10 * m_new`
   - При t→∞: `1 - β^t → 1`, correction исчезает
   - Компенсирует инициализацию нулями в начале обучения

4. **Обновление параметра:**
   - `m_hat` — направление обновления
   - `√v_hat + eps` — адаптивный масштаб (нормализация)
   - `eps` защищает от деления на ноль

5. **Векторизация:**
   - `np.asarray()` работает со скалярами, списками, массивами любой размерности
   - Все операции элементwise благодаря NumPy broadcasting

---

## ✅ Тесты

**Набор тестов в `adam-optimizer/test_adam_optimizer.py`:**

| Тест | Описание | Проверяет |
|------|----------|-----------|
| `test_zero_gradient_no_change` | `grad=0` | Param не меняется |
| `test_t1_bias_correction` | `t=1` | Значительная bias correction |
| `test_scalar_input` | Скалярные входы | 0-d array возврат |
| `test_array_input_1d` | 1D массив | Форма (n,) |
| `test_array_input_2d` | 2D массив | Форма (n, m) |
| `test_moment_update` | Обновление моментов | Формулы m_new, v_new |
| `test_bias_correction_t10` | `t=10` | Меньшая коррекция |
| `test_numerical_stability` | `grad=0, v=0` | Нет NaN/Inf |
| `test_large_gradient` | `grad=1000` | Стабильность |
| `test_large_array_vectorized` | 10000 элементов | Векторизация |
| `test_return_types` | Типы возврата | Tuple из np.ndarray |
| `test_learning_rate_effect` | Разные lr | Влияние на обновление |
| `test_beta_parameters` | Разные beta1, beta2 | Влияние на моменты |

**Результат:** Все 13 тестов ✅

---

## 📊 Пример работы

```python
import numpy as np
from adam_optimizer import adam_step

# Инициализация
param = np.array([1.0, 2.0])
grad = np.array([0.1, 0.2])
m = np.zeros(2)
v = np.zeros(2)

# Шаг 1
param, m, v = adam_step(param, grad, m, v, t=1)
print(f"Step 1: param = {param}")  # [0.999, 1.999]

# Шаг 2 (с новыми градиентами)
grad = np.array([0.05, 0.1])
param, m, v = adam_step(param, grad, m, v, t=2)
print(f"Step 2: param = {param}")
```

---

## 📝 Примечания

**Adam vs SGD:**

| Характеристика | SGD | Adam |
|---------------|-----|------|
| Learning rate | Единый для всех | Адаптивный на параметр |
| Momentum | Нет (или отдельный) | Встроенный (m_t) |
| Bias correction | Нет | Да (для t малых) |
| Численная стабильность | Зависит от lr | `eps` в знаменателе |
| Сходимость | Может быть медленной | Обычно быстрее |

**Типичные значения гиперпараметров:**
- `lr = 0.001` (начальное значение, может требовать tuning)
- `beta1 = 0.9` (редко меняется)
- `beta2 = 0.999` (редко меняется)
- `eps = 1e-8` (обычно не меняется)

**Когда использовать Adam:**
- ✅ Глубокие нейронные сети
- ✅ Нестационарные цели
- ✅ Шумные градиенты
- ✅ Требуется быстрая сходимость

**Когда может быть лучше SGD:**
- ✅ Требуется лучшая генерализация
- ✅ Хорошо настроенный learning rate schedule
- ✅ Простые модели

---

## 📚 Ссылки

- [Adam Paper: "Adam: A Method for Stochastic Optimization"](https://arxiv.org/abs/1412.6980)
- [TensorTonic Problem](https://www.tensortonic.com/problems/adam-optimizer)
- [PyTorch Adam Implementation](https://pytorch.org/docs/stable/generated/torch.optim.Adam.html)
