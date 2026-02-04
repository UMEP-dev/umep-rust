# Contributing

Thank you for your interest in contributing to SOLWEIG!

## Development Setup

### Prerequisites

- Python 3.10+
- Rust toolchain (for building extensions)
- uv (package manager)

### Clone and Install

```bash
git clone https://github.com/UMEP-dev/solweig.git
cd solweig

# Install dependencies
uv sync

# Build Rust extension
maturin develop
```

### Verify Installation

```bash
# Run tests
pytest ./tests

# Full verification (format, lint, typecheck, test)
poe verify_project
```

## Development Workflow

### Making Changes

1. Create a feature branch: `git checkout -b feature/my-feature`
2. Make your changes
3. Run verification: `poe verify_project`
4. Commit with clear messages
5. Open a pull request

### Code Style

We use these tools (configured in `pyproject.toml`):

| Tool | Purpose |
|------|---------|
| **ruff** | Linting and formatting |
| **ty** | Type checking |
| **pytest** | Testing |

Run all checks:

```bash
poe verify_project
```

### Testing

Tests are in `tests/`:

- `tests/spec/` - Physical property tests (shadows, SVF, radiation)
- `tests/golden/` - Reference data validation
- `tests/test_api.py` - Integration tests

Add tests for new functionality:

```bash
# Run specific test file
pytest tests/test_api.py

# Run with coverage
pytest --cov=solweig tests/
```

## Project Structure

```
pysrc/solweig/          # Python source
  api.py                # Public API
  models/               # Dataclasses (SurfaceData, Weather, etc.)
  components/           # Modular calculation functions
  computation.py        # Core orchestration
rust/                   # Rust extensions
specs/                  # Module specifications
tests/                  # Test suite
docs/                   # Documentation (MkDocs)
```

## Types of Contributions

### Bug Reports

Open an issue with:

- Clear description of the bug
- Steps to reproduce
- Expected vs actual behavior
- Version information

### Feature Requests

Open an issue describing:

- The use case
- Proposed solution
- Alternatives considered

### Code Contributions

1. Check existing issues for related work
2. Discuss major changes in an issue first
3. Follow the code style guidelines
4. Add tests for new functionality
5. Update documentation as needed

## License

By contributing, you agree that your contributions will be licensed under the project's license.
