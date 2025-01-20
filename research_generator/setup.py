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