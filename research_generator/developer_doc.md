# Research Data Generator
## Developer Documentation

### Project Overview
A comprehensive tool for generating research conversation datasets with authenticity metrics, edge cases, and sophisticated patterns for machine learning model training.

### Project Structure
```
research_generator/
├── data_generation/
│   ├── __init__.py
│   ├── generator.py        # Main generator class
│   ├── edge_cases.py      # Edge case definitions
│   ├── patterns.py        # Conversation patterns
│   └── metrics.py         # Authenticity metrics
├── models/
│   ├── __init__.py
│   ├── training.py        # Training pipeline
│   └── evaluation.py      # Evaluation metrics
├── utils/
│   ├── __init__.py
│   └── helpers.py         # Utility functions
├── tests/
│   ├── __init__.py
│   ├── test_generator.py
│   └── test_metrics.py
├── examples/
│   ├── basic_usage.py
│   └── advanced_config.py
├── README.md
├── requirements.txt
├── setup.py
└── .gitignore
```

### Prerequisites
- Python 3.8+
- pip
- Virtual environment (recommended)

### Dependencies
Create a `requirements.txt` file with the following contents:
```txt
numpy>=1.21.0
pandas>=1.3.0
torch>=1.9.0
transformers>=4.11.0
scikit-learn>=0.24.2
tqdm>=4.62.0
pydantic>=1.8.2
pytest>=6.2.5
black>=21.9b0
isort>=5.9.3
mypy>=0.910
```

### Local Development Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/research-generator.git
cd research-generator
```

2. Create and activate virtual environment:
```bash
# Using venv
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on Unix or MacOS
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install development dependencies:
```bash
pip install -e ".[dev]"
```

5. Setup pre-commit hooks:
```bash
pre-commit install
```

### Project Setup
Create a `setup.py` file:
```python
from setuptools import setup, find_packages

setup(
    name="research_generator",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "torch>=1.9.0",
        "transformers>=4.11.0",
        "scikit-learn>=0.24.2",
        "tqdm>=4.62.0",
        "pydantic>=1.8.2",
    ],
    extras_require={
        "dev": [
            "pytest>=6.2.5",
            "black>=21.9b0",
            "isort>=5.9.3",
            "mypy>=0.910",
            "pre-commit>=2.15.0",
        ]
    },
)
```

### Running Tests
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_generator.py

# Run with coverage
pytest --cov=research_generator tests/
```

### Code Style
The project uses:
- Black for code formatting
- isort for import sorting
- mypy for type checking

Run formatters:
```bash
# Format code
black research_generator tests

# Sort imports
isort research_generator tests

# Type checking
mypy research_generator
```

### Basic Usage Example
```python
from research_generator import UnifiedResearchGenerator

# Initialize generator
config = {
    'size': 1000,
    'edge_case_ratio': 0.2,
    'domains': ['education', 'psychology']
}

generator = UnifiedResearchGenerator(config)
conversations, metrics = generator.generate_dataset()
generator.save_dataset(conversations, metrics, "research_data")
```

### Advanced Configuration
```python
config = {
    'size': 1000,
    'min_length': 5,
    'max_length': 20,
    'edge_case_ratio': 0.2,
    'domains': ['education', 'psychology', 'stem'],
    'complexity_levels': ['basic', 'medium', 'complex'],
    'metrics': {
        'authenticity': {
            'weights': {
                'methodology_score': 0.3,
                'theoretical_alignment': 0.3,
                'citation_quality': 0.2,
                'consistency': 0.2
            }
        },
        'edge_cases': {
            'min_probability': 0.05,
            'max_probability': 0.15
        }
    }
}
```

### Development Workflow

1. Create a new branch for feature development:
```bash
git checkout -b feature/new-feature
```

2. Make changes and run tests:
```bash
# Run tests
pytest

# Run formatters
black research_generator
isort research_generator
```

3. Commit changes:
```bash
git add .
git commit -m "feat: add new feature"
```

4. Push changes and create pull request:
```bash
git push origin feature/new-feature
```

### Common Development Tasks

1. Adding new domain templates:
```python
# In data_generation/patterns.py
def add_domain_templates(domain: str, templates: List[str]):
    """Add new templates for a domain"""
    pass
```

2. Creating custom metrics:
```python
# In data_generation/metrics.py
def create_custom_metric(name: str, calculator: Callable):
    """Register new metric calculator"""
    pass
```

3. Extending edge cases:
```python
# In data_generation/edge_cases.py
def register_edge_case(edge_case: EdgeCase):
    """Register new edge case type"""
    pass
```

### Troubleshooting

1. Installation Issues:
- Ensure Python 3.8+ is installed
- Check virtual environment activation
- Verify pip is updated: `pip install --upgrade pip`

2. Runtime Errors:
- Check configuration format
- Verify domain/template compatibility
- Ensure sufficient memory for dataset size

3. Performance Issues:
- Use smaller batch sizes
- Enable multiprocessing for large datasets
- Monitor memory usage

### Contributing
1. Fork the repository
2. Create feature branch
3. Make changes and run tests
4. Create pull request

### License
MIT License

### Contact
- Project maintainers
- Issue tracker
- Discussion forum