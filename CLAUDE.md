# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

zerohertzLib is a comprehensive Python library providing utilities for scientific computing, AI/ML, data visualization, monitoring, and API integrations. The library supports modular installation with optional dependencies through extras like `[api]`, `[mlops]`, `[quant]`, and `[all]`.

## Development Commands

### Environment Setup
```bash
python -m venv venv
source venv/bin/activate
pip install .'[all]' --no-cache-dir
```

### Code Quality
```bash
# Install style requirements
pip install -r requirements/requirements-style.txt

# Format code
black .

# Lint code
flake8 zerohertzLib
pylint -r n zerohertzLib

# Check formatting
black --check .
```

### Testing
```bash
# Install test dependencies
pip install pytest pytest-cov

# Run all tests with coverage
PYTHONPATH=. pytest --cov=zerohertzLib --cov-report=xml --junitxml=junit.xml -o junit_family=legacy

# Run specific test module
PYTHONPATH=. pytest test/test_api.py

# Run specific test function
PYTHONPATH=. pytest test/test_api.py::test_discord_bot_message
```

### Documentation
```bash
# Build Sphinx documentation
cd sphinx
python -m pip install -r requirements/requirements-docs.txt
# Documentation building scripts are in sphinx/ directory
```

### Installation for Testing
```bash
# Required system dependency for vision module
sudo apt install python3-opencv -y

# Install library with all dependencies
pip install .'[all]'
```

## Architecture Overview

### Module Structure
- **api/**: External service integrations (Discord, Slack, OpenAI, GitHub, etc.)
- **algorithm/**: Data structures and algorithms
- **logging/**: Enhanced logging utilities with rich formatting
- **mlops/**: ML operations tools including Triton inference server support
- **monitoring/**: System monitoring and resource tracking
- **plot/**: Data visualization with matplotlib/seaborn
- **quant/**: Quantitative finance tools and data analysis
- **util/**: General utilities and helper functions
- **vision/**: Computer vision utilities with OpenCV

### Optional Dependencies
The library uses conditional imports for modules with heavy dependencies:
- API integrations require `[api]` extra
- MLOps features require `[mlops]` extra  
- Quantitative finance requires `[quant]` extra
- Vision module requires OpenCV system package

### Command Line Tools
Several CLI tools are installed as scripts:
- `trictl`: Triton inference server management
- `vert`: Vision-related operations
- `grid`: Grid operations for vision
- `img2gif`/`vid2gif`: Media conversion utilities

### Testing Environment
Tests require environment variables for API credentials:
- `DISCORD_WEBHOOK_URL`, `DISCORD_BOT_TOKEN`, `DISCORD_CHANNEL_ID`
- `SLACK_WEBHOOK_URL`, `SLACK_BOT_TOKEN`
- `OPENAI_API_KEY`, `GH_TOKEN`

Test data is located in `test/data/` directory.

### Branch Strategy
- `master`: Main production branch
- `dev-*`: Development branches for feature work
- CI/CD runs on pushes to dev branches and PRs to master

### Code Style
- Uses Black formatting
- flake8 for linting
- pylint for additional code quality checks
- Consistent docstring format with type hints