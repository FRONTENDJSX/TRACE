# Contributing to TRACE

Thank you for your interest in contributing to TRACE! We welcome contributions from the community and appreciate your help in making TRACE better.

## 🤝 How to Contribute

### Reporting Issues

Before creating an issue, please:

1. Check if the issue already exists
2. Use the issue templates provided
3. Include as much detail as possible:
   - Operating system and version
   - Python version
   - TRACE version
   - Steps to reproduce
   - Expected vs actual behavior

### Suggesting Features

We welcome feature suggestions! Please:

1. Check if the feature has been requested before
2. Provide a clear description of the feature
3. Explain the use case and benefits
4. Consider implementation complexity

### Code Contributions

#### Getting Started

1. **Fork the repository**
2. **Clone your fork**:
   ```bash
   git clone https://github.com/yourusername/TRACE.git
   cd TRACE
   ```

3. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt  # For development
   ```

5. **Set up pre-commit hooks**:
   ```bash
   pre-commit install
   ```

#### Development Workflow

1. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**:
   - Follow the coding standards
   - Add tests for new functionality
   - Update documentation as needed

3. **Run tests**:
   ```bash
   python -m pytest tests/
   python -m pytest tests/ --cov=TRACE  # With coverage
   ```

4. **Check code quality**:
   ```bash
   flake8 TRACE/
   black TRACE/ --check
   mypy TRACE/
   ```

5. **Commit your changes**:
   ```bash
   git add .
   git commit -m "feat: add new feature"
   ```

6. **Push and create a pull request**:
   ```bash
   git push origin feature/your-feature-name
   ```

## 📋 Coding Standards

### Python Code Style

- Follow PEP 8 guidelines
- Use type hints for function parameters and return values
- Write docstrings for all public functions and classes
- Keep functions small and focused
- Use meaningful variable and function names

### Code Formatting

We use the following tools for code formatting:

- **Black**: Code formatting
- **isort**: Import sorting
- **flake8**: Linting
- **mypy**: Type checking

### Documentation

- Update README.md for user-facing changes
- Add docstrings for new functions/classes
- Update API documentation for new endpoints
- Include examples in docstrings

## 🧪 Testing

### Writing Tests

- Write tests for all new functionality
- Aim for high test coverage
- Use descriptive test names
- Test both success and failure cases
- Mock external dependencies (API calls, file I/O)

### Test Structure

```python
def test_function_name():
    """Test description."""
    # Arrange
    input_data = "test"
    
    # Act
    result = function_under_test(input_data)
    
    # Assert
    assert result == expected_output
```

### Running Tests

```bash
# Run all tests
python -m pytest

# Run specific test file
python -m pytest tests/test_trace.py

# Run with coverage
python -m pytest --cov=TRACE

# Run with verbose output
python -m pytest -v
```

## 📝 Commit Messages

We follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

### Format
```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

### Types
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

### Examples
```
feat: add support for multiple camera inputs
fix: resolve memory leak in image processing
docs: update API documentation
test: add unit tests for face detection
```

## 🔍 Pull Request Process

### Before Submitting

1. **Ensure tests pass**:
   ```bash
   python -m pytest tests/
   ```

2. **Check code quality**:
   ```bash
   flake8 TRACE/
   black TRACE/ --check
   mypy TRACE/
   ```

3. **Update documentation** if needed

4. **Test your changes** thoroughly

### Pull Request Template

When creating a pull request, please include:

- **Description**: What changes were made and why
- **Type**: Bug fix, feature, documentation, etc.
- **Testing**: How the changes were tested
- **Breaking Changes**: Any breaking changes and migration steps
- **Screenshots**: For UI changes
- **Checklist**: Ensure all items are completed

### Review Process

1. **Automated checks** must pass
2. **Code review** by maintainers
3. **Testing** on different environments
4. **Documentation** review
5. **Approval** and merge

## 🏗️ Project Structure

```
TRACE/
├── TRACE.py              # Core TRACE system
├── trace_api.py         # REST API server
├── requirements.txt     # Production dependencies
├── requirements-dev.txt # Development dependencies
├── tests/              # Test files
├── docs/               # Documentation
├── examples/           # Example scripts
└── README.md           # Main documentation
```

## 🐛 Bug Reports

When reporting bugs, please include:

1. **Environment**:
   - Operating system and version
   - Python version
   - TRACE version
   - Dependencies versions

2. **Reproduction steps**:
   - Clear, step-by-step instructions
   - Minimal code example if possible
   - Expected vs actual behavior

3. **Additional context**:
   - Error messages and stack traces
   - Screenshots if applicable
   - Related issues or discussions

## 💡 Feature Requests

When suggesting features:

1. **Clear description** of the feature
2. **Use case** and benefits
3. **Implementation ideas** (optional)
4. **Alternatives considered**
5. **Additional context**

## 📞 Getting Help

- **GitHub Discussions**: For questions and general discussion
- **GitHub Issues**: For bug reports and feature requests
- **Discord**: For real-time chat and support
- **Email**: For security issues (security@trace-ai.com)

## 🏆 Recognition

Contributors will be recognized in:

- CONTRIBUTORS.md file
- Release notes
- Project documentation
- Social media acknowledgments

## 📄 License

By contributing to TRACE, you agree that your contributions will be licensed under the Apache License 2.0.

---

Thank you for contributing to TRACE! 🎉
