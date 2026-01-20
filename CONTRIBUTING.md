# Contributing to Hagfish-SOTA

Thank you for your interest in contributing to Hagfish-SOTA! This guide will help you get started.

---

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How to Contribute](#how-to-contribute)
- [Development Setup](#development-setup)
- [Pull Request Process](#pull-request-process)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Documentation](#documentation)
- [Issue Guidelines](#issue-guidelines)

---

## Code of Conduct

### Our Pledge

We are committed to providing a welcoming and inclusive experience for everyone, regardless of age, body size, disability, ethnicity, gender identity, experience level, nationality, personal appearance, race, religion, or sexual identity and orientation.

### Our Standards

**Positive behavior includes:**

- Using welcoming and inclusive language
- Being respectful of differing viewpoints
- Gracefully accepting constructive criticism
- Focusing on what's best for the community
- Showing empathy towards others

**Unacceptable behavior includes:**

- Harassment, trolling, or derogatory comments
- Personal or political attacks
- Public or private harassment
- Publishing others' private information
- Other conduct inappropriate in a professional setting

---

## How to Contribute

### Ways to Contribute

We welcome contributions in many forms:

1. **🐛 Bug Reports** - Found a bug? Let us know!
2. **✨ Feature Requests** - Have an idea? Share it!
3. **📖 Documentation** - Improve docs, add examples
4. **🔧 Code** - Fix bugs, add features
5. **🧪 Testing** - Add tests, improve coverage
6. **📊 Benchmarks** - Run on new datasets
7. **💬 Discussions** - Help others, share experiences

### Good First Issues

Look for issues labeled `good first issue` or `help wanted`:

- Documentation improvements
- Adding examples
- Simple bug fixes
- Test additions

---

## Development Setup

### Prerequisites

- Python 3.8+
- Git
- pip or conda

### Fork and Clone

```bash
# Fork on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/hagfish-adaptive-trainer.git
cd hagfish-adaptive-trainer

# Add upstream remote
git remote add upstream https://github.com/original-repo/hagfish-adaptive-trainer.git
```

### Install in Development Mode

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install package in editable mode
pip install -e .

# Install development dependencies
pip install -e ".[dev]"
# This includes: pytest, black, flake8, mypy, etc.

# Install benchmarking dependencies (optional)
pip install -e ".[benchmark]"
```

### Verify Installation

```bash
# Run basic tests
pytest tests/

# Try a quick example
python examples/sklearn_example.py
```

---

## Pull Request Process

### 1. Create a Branch

```bash
# Update your fork
git fetch upstream
git checkout main
git merge upstream/main

# Create feature branch
git checkout -b feature/your-feature-name
# Or for bug fixes:
git checkout -b fix/issue-description
```

**Branch naming:**

- `feature/add-xyz` - New features
- `fix/issue-123` - Bug fixes
- `docs/improve-xyz` - Documentation
- `test/add-xyz-tests` - Tests
- `refactor/cleanup-xyz` - Code refactoring

### 2. Make Changes

```bash
# Make your changes
# ... edit files ...

# Format code
black .
flake8 .

# Run tests
pytest tests/

# Check type hints
mypy adaptive_trainer/
```

### 3. Commit Changes

Use conventional commit messages:

```bash
# Format: <type>: <description>
#
# Types: feat, fix, docs, test, refactor, style, chore
#
# Examples:
git commit -m "feat: add parallel evaluation support"
git commit -m "fix: resolve memory leak in reward history"
git commit -m "docs: improve quick start guide"
git commit -m "test: add convergence tests for Issue #8"
```

### 4. Push and Create PR

```bash
# Push to your fork
git push origin feature/your-feature-name

# Go to GitHub and create Pull Request
```

### 5. PR Description Template

```markdown
## Description

Brief description of changes

## Motivation

Why is this change needed?

## Changes

- Change 1
- Change 2
- Change 3

## Testing

How was this tested?

- [ ] Unit tests added
- [ ] Integration tests added
- [ ] Manual testing performed
- [ ] Benchmarks run

## Checklist

- [ ] Code follows project style
- [ ] Tests pass locally
- [ ] Documentation updated
- [ ] CHANGELOG.md updated (if user-facing)
- [ ] No breaking changes (or documented)

## Related Issues

Fixes #123
Relates to #456
```

### 6. Review Process

- Maintainers will review your PR
- Address feedback in new commits
- Once approved, it will be merged
- Your contribution will be credited!

---

## Coding Standards

### Python Style

We follow **PEP 8** with some modifications:

```python
# Good
def compute_reward(accuracy: float, cost: float, alpha: float = 0.3) -> float:
    """
    Compute reward as accuracy minus cost penalty.

    Args:
        accuracy: Model accuracy (0-1)
        cost: Computational cost
        alpha: Cost penalty weight

    Returns:
        Reward value (higher is better)
    """
    return accuracy - alpha * cost


# Bad
def computeReward(acc,cost,a=0.3):
    return acc-a*cost  # No docstring, unclear names
```

### Code Formatting

Use **Black** for auto-formatting:

```bash
# Format all files
black .

# Format specific file
black adaptive_trainer/planner.py

# Check without modifying
black --check .
```

### Linting

Use **flake8** for linting:

```bash
# Check all files
flake8 .

# Ignore specific errors
flake8 --ignore=E501,W503 .
```

### Type Hints

Use **mypy** for type checking:

```python
from typing import Dict, List, Optional, Any

def plan(self, context: Dict[str, Any]) -> Dict[str, Any]:
    """Type hints for better IDE support and safety"""
    pass
```

Check types:

```bash
mypy adaptive_trainer/
```

### Docstrings

Use **Google-style** docstrings:

```python
def train_model(X: np.ndarray, y: np.ndarray, fidelity: float = 1.0) -> float:
    """
    Train model with given fidelity level.

    Args:
        X: Training features, shape (n_samples, n_features)
        y: Training labels, shape (n_samples,)
        fidelity: Training intensity (0.2-1.0), default 1.0

    Returns:
        Validation accuracy (0-1 scale)

    Raises:
        ValueError: If fidelity is outside valid range

    Example:
        >>> X_train, y_train = load_data()
        >>> accuracy = train_model(X_train, y_train, fidelity=0.75)
        >>> print(f"Accuracy: {accuracy:.4f}")
        Accuracy: 0.8523
    """
    if not 0.2 <= fidelity <= 1.0:
        raise ValueError(f"Fidelity must be in [0.2, 1.0], got {fidelity}")

    # Implementation...
    return accuracy
```

---

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_planner.py

# Run specific test
pytest tests/test_planner.py::test_plan_basic

# Run with coverage
pytest --cov=adaptive_trainer tests/

# Generate coverage report
pytest --cov=adaptive_trainer --cov-report=html tests/
```

### Writing Tests

Use **pytest** framework:

```python
# tests/test_planner.py
import pytest
from adaptive_trainer import AdaptiveTrainer


class TestAdaptiveTrainer:
    """Test suite for AdaptiveTrainer"""

    def test_initialization(self):
        """Test trainer initializes correctly"""
        trainer = AdaptiveTrainer(alpha=0.3)
        assert trainer.alpha == 0.3

    def test_plan_returns_dict(self):
        """Test plan() returns dictionary"""
        trainer = AdaptiveTrainer(alpha=0.3)
        plan = trainer.plan({"dataset_size": 1000})

        assert isinstance(plan, dict)
        assert "fidelity" in plan
        assert 0.2 <= plan["fidelity"] <= 1.0

    def test_observe_updates_memory(self):
        """Test observe() updates internal state"""
        trainer = AdaptiveTrainer(alpha=0.3)
        trainer.plan({"dataset_size": 1000})

        initial_history = len(trainer.memory.reward_history)
        trainer.observe(metric=0.85, cost=1.5)

        assert len(trainer.memory.reward_history) == initial_history + 1

    @pytest.mark.parametrize("alpha", [0.1, 0.3, 0.5, 0.7])
    def test_alpha_values(self, alpha):
        """Test different alpha values"""
        trainer = AdaptiveTrainer(alpha=alpha)
        assert trainer.alpha == alpha

    def test_plan_missing_dataset_size(self):
        """Test plan() raises error without dataset_size"""
        trainer = AdaptiveTrainer(alpha=0.3)

        with pytest.raises(ValueError):
            trainer.plan({})
```

### Test Categories

**Unit Tests** (`tests/test_*.py`)

- Test individual components
- Fast, isolated, deterministic
- Mock external dependencies

**Integration Tests** (`tests/integration/`)

- Test component interactions
- Use real dependencies
- Slower but more comprehensive

**Benchmark Tests** (`tests/benchmark/`)

- Performance regression tests
- Compare to baselines
- Long-running (mark as `@pytest.mark.slow`)

---

## Documentation

### Adding Documentation

**README.md:**

- Update for user-facing changes
- Keep examples up-to-date

**API Reference** (`docs/API_REFERENCE.md`):

- Document new APIs
- Include examples
- Specify parameters, returns, raises

**Quick Start** (`docs/QUICK_START.md`):

- Update for workflow changes
- Add new examples

**CHANGELOG.md:**

- Document changes (user-facing only)
- Follow Keep a Changelog format

### Documentation Style

```markdown
# Use ATX-style headers

## Second level

### Third level

# Use fenced code blocks with language

\`\`\`python
code here
\`\`\`

# Use tables for comparisons

| Method | Accuracy | Cost |
| ------ | -------- | ---- |
| Fixed  | 0.85     | 2.0  |

# Use badges for status

![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)

# Link to other docs

See [Quick Start](QUICK_START.md) for examples.
```

---

## Issue Guidelines

### Reporting Bugs

Use the **Bug Report** template:

```markdown
**Describe the bug**
A clear description of the bug.

**To Reproduce**
Steps to reproduce:

1. Install version X
2. Run command Y
3. See error

**Expected behavior**
What should happen?

**Actual behavior**
What actually happens?

**Environment**

- OS: Windows 10
- Python: 3.9.7
- Hagfish version: 1.0.0
- Dependencies: (output of `pip list`)

**Additional context**
Logs, screenshots, etc.
```

### Feature Requests

Use the **Feature Request** template:

```markdown
**Problem Statement**
What problem does this solve?

**Proposed Solution**
How would this feature work?

**Alternatives Considered**
Other approaches you've considered?

**Additional Context**
Examples, mockups, related issues
```

### Discussion Topics

For open-ended discussions:

- Use GitHub Discussions
- Tag with appropriate labels
- Be respectful and constructive

---

## Development Workflow

### Daily Workflow

```bash
# Start of day: sync with upstream
git fetch upstream
git checkout main
git merge upstream/main

# Create feature branch
git checkout -b feature/my-feature

# Work on changes
# ... edit files ...

# Run tests frequently
pytest tests/

# Commit when ready
git add .
git commit -m "feat: add my feature"

# Push to your fork
git push origin feature/my-feature

# Create PR on GitHub
```

### Code Review Checklist

Before submitting PR:

- [ ] Code follows project style (black, flake8)
- [ ] All tests pass (`pytest`)
- [ ] Type hints added (`mypy`)
- [ ] Docstrings added (Google style)
- [ ] Documentation updated
- [ ] CHANGELOG.md updated (if needed)
- [ ] No debugging code left
- [ ] No TODOs without issue numbers
- [ ] Commit messages follow convention

---

## Areas We Need Help

### High Priority

1. **State Persistence** 🔧
   - Save/load trainer state
   - Resume interrupted runs
   - Cross-session reproducibility

2. **Parallel Evaluation** 🚀
   - Multi-worker support
   - Distributed execution (Ray, Dask)
   - Asynchronous evaluation

3. **More Benchmarks** 📊
   - OpenML tasks
   - Computer vision datasets
   - NLP benchmarks

### Medium Priority

4. **Advanced Pruning** ✂️
   - Learning curve extrapolation
   - Early stopping criteria
   - Adaptive pruning thresholds

5. **Multi-Objective** 🎯
   - Pareto frontier tracking
   - Scalarization methods
   - NSGA-II integration

6. **Visualization** 📈
   - Real-time dashboards
   - Convergence plots
   - Interactive exploration

### Low Priority (But Welcome!)

7. **Integration Examples** 🧪
   - XGBoost/LightGBM
   - JAX/Flax
   - Hugging Face Transformers

8. **Performance Optimization** ⚡
   - Reduce memory footprint
   - Speed up plan() calls
   - Vectorized operations

---

## Recognition

### Contributors

All contributors are recognized in:

- README.md Contributors section
- CHANGELOG.md for their contributions
- GitHub Contributors page

### Types of Contributions

- 💻 Code
- 📖 Documentation
- 🐛 Bug reports
- 💡 Ideas
- 🤔 Answering questions
- 🧪 Testing
- 📊 Benchmarks

---

## Getting Help

### Resources

- **Documentation:** [docs/INDEX.md](docs/INDEX.md)
- **Examples:** `examples/` directory
- **Tests:** `tests/` directory (show patterns)

### Contact

- **GitHub Issues:** For bugs and features
- **GitHub Discussions:** For questions
- **Email:** maintainer@example.com (for private matters)

---

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

**Thank you for contributing to Hagfish-SOTA!** 🐟🚀

Every contribution, no matter how small, makes a difference.
