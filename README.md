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
| 2 | [Logistic Regression Training Loop](docs/solutions/logistic-regression-training.md) | Medium | Optimization, Loss Functions |
| 3 | [Pad Sequences](docs/solutions/pad-sequences.md) | Medium | NLP, Data Processing |
| 4 | [Matrix Transpose](docs/solutions/matrix-transpose.md) | Easy | Linear Algebra |
| 5 | [Implement Positional Encoding](docs/solutions/positional-encoding.md) | Medium | Linear Algebra, Transformers |
| 6 | [Gradient Descent for 1D Quadratic](docs/solutions/gradient-descent-quadratic.md) | Easy | Optimization |
| 7 | [Implement Adam Optimizer Step](docs/solutions/adam-optimizer-step.md) | Easy | Optimization |
| 8 | [Implement Dropout (Training Mode)](docs/solutions/dropout-training.md) | Medium | Neural Networks |
| 9 | [RMSProp Optimizer (Single Update Step)](docs/solutions/rmsprop-optimizer.md) | Easy | Optimization |

## 📁 Repository Structure

```
.
├── <task>/                  # 💻 Solution code from TensorTonic (tracked in git)
│   ├── adam-optimizer/
│   │   └── adam-optimizer.py
│   ├── dropout-training/
│   │   └── dropout-training.py
│   ├── rmsprop-optimizer/
│   │   └── rmsprop-optimizer.py
│   ├── sigmoid-numpy/
│   │   └── sigmoid-numpy.py
│   ├── logistic-regression-training/
│   │   └── logistic-regression-training.py
│   ├── pad-sequences/
│   │   └── pad-sequences.py
│   ├── matrix-transpose/
│   │   └── matrix-transpose.py
│   ├── positional-encoding/
│   │   └── positional-encoding.py
│   └── gradient-descent-quadratic/
│       └── gradient-descent-quadratic.py
│
├── docs/                    # 📖 Detailed solutions documentation (pushed to GitHub)
│   └── solutions/
│       ├── README.md
│       ├── sigmoid-numpy.md
│       ├── logistic-regression-training.md
│       ├── pad-sequences.md
│       ├── matrix-transpose.md
│       ├── positional-encoding.md
│       ├── gradient-descent-quadratic.md
│       ├── adam-optimizer-step.md
│       └── dropout-training.md
│
├── synced_solutions/        # 🔄 Auto-synced from TensorTonic platform (git submodule)
│   ├── <task>/              # Code synchronized from platform
│   └── docs/solutions/      # Documentation mirror
│
└── sandbox/                 # 🧪 Local development (NOT tracked in git)
    ├── TASKS.md
    ├── SOLUTIONS_JOURNAL.md
    ├── <task>/
    │   ├── <task>.py
    │   └── test_<task>.py
    └── ...
```

## 🚀 Quick Start

```bash
# Pull latest solutions from TensorTonic
git -C synced_solutions pull

# Run tests for a solution
cd sandbox/sigmoid-numpy
python test_sigmoid_numpy.py
```
