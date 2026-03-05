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

| Путь | Назначение | Кто обновляет |
|------|------------|---------------|
| `synced_solutions/*.py` | Код решений с платформы | TensorTonic (pull) |
| `synced_solutions/docs/solutions/*.md` | Документация решений | Мы + платформа |
| `docs/solutions/*.md` | Документация (основная) | Мы |
| `adam-optimizer/test_*.py` | Тесты | Мы |
| `sandbox/` | Локальная разработка | Мы (не в git) |

## 🔄 Процесс синхронизации

### 1. Получение новых задач с платформы

```bash
# Получить новые решения с TensorTonic
git -C synced_solutions pull
```

### 2. Добавление тестов и документации

```bash
# 1. Создать тесты в папке решения (например, adam-optimizer/test_adam_optimizer.py)
# 2. Создать документацию в docs/solutions/ (например, adam-optimizer-step.md)
# 3. Обновить README файлы

# Добавить изменения
git add .

# Закоммитить
git commit -m "Add tests and docs for <task-name>"

# Отправить на GitHub
git push origin main
```

### 3. Синхронизация документации

После добавления новой задачи:

```bash
# 1. Обновить docs/solutions/README.md (добавить новую задачу)
# 2. Скопировать документацию в synced_solutions
copy docs\solutions\<task>.md synced_solutions\docs\solutions\

# 3. Закоммитить в synced_solutions
git -C synced_solutions add -A
git -C synced_solutions commit -m "Add <task> documentation"
git -C synced_solutions pull --rebase origin main  # На случай конфликта
git -C synced_solutions push origin main
```

## ⚠️ Важные правила

1. **`synced_solutions/`** — только pull с платформы, код не редактировать
2. **`docs/solutions/`** — наша документация, ведём самостоятельно
3. **`sandbox/`** — локальная разработка, не в git
4. **Тесты** — в папке решения рядом с кодом (например, `adam-optimizer/test_*.py`)

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

## 📋 Чеклист для новой задачи

- [ ] Получить задачу с платформы (`git -C synced_solutions pull`)
- [ ] Изучить требования в `synced_solutions/<task>/<task>.py`
- [ ] Реализовать решение (если нужно доработать)
- [ ] Написать тесты (`<task>/test_<task>.py`)
- [ ] Создать документацию (`docs/solutions/<task>.md`)
- [ ] Обновить `docs/solutions/README.md`
- [ ] Обновить `synced_solutions/docs/solutions/README.md`
- [ ] Скопировать документацию в `synced_solutions/docs/solutions/`
- [ ] Закоммитить и запушить основное репо
- [ ] Закоммитить и запушить synced_solutions

## 🔧 Полезные команды

```bash
# Проверить статус обоих репо
git status && git -C synced_solutions status

# Получить последние изменения с GitHub
git pull origin main
git -C synced_solutions pull

# Откатить изменения в synced_solutions (если случайно изменил)
git -C synced_solutions checkout <file>
```
