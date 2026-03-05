# Руководство по синхронизации TensorTonic Solutions

## 🏗 Архитектура репозиториев

```
┌─────────────────────────────────────────────────────────────┐
│                    GitHub (один репо)                       │
│            github.com/dizel0110/TensorTonic-Solutions       │
└─────────────────────────────────────────────────────────────┘
                              ▲
                              │
         ┌────────────────────┴────────────────────┐
         │                                         │
         ▼                                         ▼
┌─────────────────┐                       ┌─────────────────┐
│ synced_solutions│                       │   Основной репо │
│ (платформа)     │                       │   (локальный)   │
│ git pull только │                       │   git push      │
└─────────────────┘                       └─────────────────┘
```

## 📁 Структура файлов

| Путь | Назначение | В git? | Кто обновляет |
|------|------------|--------|---------------|
| `synced_solutions/<task>/<task>.py` | Код решений с платформы | ✅ Да | TensorTonic (pull) |
| `synced_solutions/docs/solutions/*.md` | Документация (дубль) | ✅ Да | Мы + платформа |
| `docs/solutions/*.md` | Документация (основная) | ✅ Да | Мы |
| `sandbox/<task>/<task>.py` | Локальная разработка | ❌ Нет | Мы |
| `sandbox/<task>/test_*.py` | Тесты | ❌ Нет | Мы |
| `sandbox/TASKS.md` | Журнал задач | ❌ Нет | Мы |
| `sandbox/SOLUTIONS_JOURNAL.md` | Журнал решений | ❌ Нет | Мы |

## 🔄 Процесс работы над новой задачей

### 1. Получить задачу с платформы

```bash
git -C synced_solutions pull
```

Проверить, что появилось в `synced_solutions/<task>/`

### 2. Скопировать в sandbox для разработки

```bash
# Копируем код задачи из synced_solutions в sandbox
xcopy /E /I synced_solutions\<task> sandbox\<task>

# Или вручную создать папку и файл в sandbox/
```

### 3. Реализовать решение в sandbox

```
sandbox/<task>/
├── <task>.py           # Решение (дорабатываем)
└── test_<task>.py      # Тесты (пишем сами)
```

Запустить тесты:
```bash
cd sandbox/<task>
python test_<task>.py
```

### 4. Создать документацию

```bash
# Создать файл документации
docs/solutions/<task>.md
```

### 5. Обновить документацию

- `docs/solutions/README.md` — добавить задачу в таблицу
- `synced_solutions/docs/solutions/README.md` — добавить задачу в таблицу
- `sandbox/TASKS.md` — отметить задачу как выполненную
- `sandbox/SOLUTIONS_JOURNAL.md` — добавить подробное описание

### 6. Закоммитить и отправить на GitHub

```bash
# Добавить изменения
git add .

# Закоммитить
git commit -m "Add <task> tests and documentation"

# Отправить на GitHub
git push origin main
```

### 7. Синхронизировать synced_solutions

```bash
# Скопировать документацию в synced_solutions
copy docs\solutions\<task>.md synced_solutions\docs\solutions\

# Обновить README в synced_solutions
# (отредактировать synced_solutions\docs\solutions\README.md)

# Закоммитить в synced_solutions
git -C synced_solutions add -A
git -C synced_solutions commit -m "Add <task> documentation"

# Пуш (может потребоваться rebase)
git -C synced_solutions pull --rebase origin main
git -C synced_solutions push origin main
```

## 📋 Чеклист для новой задачи

- [ ] `git -C synced_solutions pull` — получить с платформы
- [ ] Скопировать в `sandbox/<task>/` для разработки
- [ ] Реализовать/доработать решение
- [ ] Написать тесты (`test_<task>.py`)
- [ ] Запустить тесты — все проходят ✅
- [ ] Создать документацию (`docs/solutions/<task>.md`)
- [ ] Обновить `docs/solutions/README.md`
- [ ] Обновить `sandbox/TASKS.md`
- [ ] Обновить `sandbox/SOLUTIONS_JOURNAL.md`
- [ ] `git add . && git commit -m "..." && git push origin main`
- [ ] Скопировать документацию в `synced_solutions/docs/solutions/`
- [ ] Обновить `synced_solutions/docs/solutions/README.md`
- [ ] `git -C synced_solutions add -A && commit && push`

## ⚠️ Важные правила

1. **`sandbox/`** — локальная разработка, **НЕ в git** (игнорируется в `.gitignore`)
2. **`synced_solutions/`** — код с платформы, **только pull**, не редактировать `.py` файлы
3. **`docs/solutions/`** — наша документация, **в git**, пушим на GitHub
4. **Код решений** — хранится в `synced_solutions/` (с платформы) и `sandbox/` (локально)

## 🚨 Разрешение конфликтов

Если `git push` отклонён:

```bash
# Для основного репо
git pull --rebase origin main
git push origin main

# Для synced_solutions
git -C synced_solutions pull --rebase origin main
git -C synced_solutions push origin main
```

## 🔧 Полезные команды

```bash
# Проверить статус обоих репо
git status && git -C synced_solutions status

# Получить последние изменения с GitHub
git pull origin main
git -C synced_solutions pull

# Запустить тесты задачи
cd sandbox/<task>
python test_<task>.py

# Откатить изменения в synced_solutions (если случайно изменил)
git -C synced_solutions checkout <file>
```

## 📝 Пример: задача Adam Optimizer

```bash
# 1. Получить с платформы
git -C synced_solutions pull
# Появилось: synced_solutions/adam-optimizer/adam-optimizer.py

# 2. Скопировать в sandbox
xcopy /E /I synced_solutions\adam-optimizer sandbox\adam-optimizer

# 3. Добавить тесты
# sandbox/adam-optimizer/test_adam_optimizer.py

# 4. Запустить тесты
cd sandbox/adam-optimizer
python test_adam_optimizer.py

# 5. Создать документацию
# docs/solutions/adam-optimizer-step.md

# 6. Обновить README
# docs/solutions/README.md — добавить задачу #7

# 7. Закоммитить и пуш
git add .
git commit -m "Add Adam optimizer tests and documentation"
git push origin main

# 8. Синхронизировать synced_solutions
copy docs\solutions\adam-optimizer-step.md synced_solutions\docs\solutions\
git -C synced_solutions add -A
git -C synced_solutions commit -m "Add Adam optimizer documentation"
git -C synced_solutions pull --rebase origin main
git -C synced_solutions push origin main
```
