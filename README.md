# TensorTonic Solutions

Welcome to my TensorTonic solutions repository!

Here you'll find my solutions to various machine learning and deep learning problems from [TensorTonic](https://tensortonic.com).

## What is TensorTonic?

TensorTonic is a platform where you can implement core algorithms of Machine Learning from scratch.

This repository contains my personal solutions to these problems, automatically synchronized from the platform.

## 📚 Solutions Documentation

Detailed solutions with explanations and tests are available in the [`docs/`](docs/solutions/README.md) folder:

| # | Problem | Difficulty | Topics |
|---|---------|------------|--------|
| 1 | [Implement Sigmoid in NumPy](docs/solutions/sigmoid-numpy.md) | Easy | Activation Functions |

## 📁 Repository Structure

```
.
├── docs/                    # 📖 Detailed solutions (pushed to GitHub)
│   └── solutions/
│       ├── README.md
│       └── sigmoid-numpy.md
│
├── synced_solutions/        # 🔄 Auto-synced from TensorTonic platform
│   └── sigmoid-numpy/
│       └── sigmoid-numpy.py
│
└── sandbox/                 # 🧪 Local development (not tracked)
    ├── TASKS.md
    ├── SOLUTIONS_JOURNAL.md
    └── sigmoid-numpy/
        ├── sigmoid-numpy.py
        └── test_sigmoid_numpy.py
```

## 🚀 Quick Start

```bash
# Pull latest solutions from TensorTonic
git -C synced_solutions pull

# Run tests for a solution
cd sandbox/sigmoid-numpy
python test_sigmoid_numpy.py
```
