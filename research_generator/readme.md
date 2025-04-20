# Research Conversation Analysis System
## Student Developer Guide

Welcome to the Research Conversation Analysis System! This guide is designed to help student developers understand and work with this system.

## What Does This System Do?

This system does three main things:
1. **Generates realistic research conversations** - Creates mock conversations between researchers and assistants about academic topics
2. **Analyzes conversations for quality** - Measures how authentic, accurate, and well-structured these conversations are
3. **Detects potential issues** - Identifies biases, citation problems, and methodological weaknesses

## Quick Setup (15 minutes)

### Prerequisites
- Python 3.8 or newer
- Basic knowledge of Python

### Installation Steps
1. Clone the repository:
```bash
git clone https://github.com/yourusername/research-generator.git
cd research-generator
```

2. Set up a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Your First Data Generation (5 minutes)

Run this simple example to generate your first dataset:

```python
# Create a file named my_first_generation.py
from research_generator import create_generator

# Set up a simple configuration
config = {
    'size': 5,  # Start with just 5 conversations for testing
    'domains': ['education'],
    'edge_case_ratio': 0.2
}

# Create the generator
generator = create_generator(config)

# Generate dataset
conversations, metrics = generator.generate_dataset()

# Save to file
generator.save_dataset(conversations, metrics, "my_first_dataset")

# Print a sample
print("\nGenerated a sample conversation:")
for msg in conversations[0]["messages"][:2]:  # Show first 2 messages
    print(f"\n{msg['role'].upper()}: {msg['content'][:100]}...")
```

Then run it:
```bash
python my_first_generation.py
```

## Understanding the System

### Main Components

1. **Data Generation**
   - Uses templates to create realistic research conversations
   - Adds "edge cases" (potential issues like citation errors)
   - Creates in various research domains (education, psychology, etc.)

2. **Analysis Models**
   - Checks if conversations are authentic
   - Detects methodology problems
   - Finds citation errors

### Project Structure
```
research_generator/
├── data_generation/         # Creates conversation datasets
├── models/                  # Analyzes conversations
├── utils/                   # Helper functions
├── examples/                # Example scripts to learn from
└── config/                   # config files
```

## Common Tasks

### Generating More Complex Datasets
```python
from research_generator import create_generator

# More detailed configuration
config = {
    'size': 50,
    'domains': ['education', 'psychology'],
    'complexity_levels': ['basic', 'medium', 'complex'],
    'edge_case_ratio': 0.2
}

generator = create_generator(config)
conversations, metrics = generator.generate_dataset()
```

### Running the Analysis
```python
from research_generator.models import ResearchAnalysisModel

# Load the model
model = ResearchAnalysisModel.load("pretrained_model.pt")

# Analyze a conversation
results = model.analyze(conversations[0])

# Print trust score
print(f"Trust Score: {results['trust_score']:.2f}")
```

## Learning Path

1. **Start with examples**
   - Look at `examples/basic_usage.py` for simple usage
   - Explore `examples/advanced_config.py` when you're ready

2. **Experiment with data generation**
   - Try different domains
   - Change complexity levels
   - Adjust conversation lengths

3. **Try analysis features**
   - Run the model on your generated data
   - Explore what makes conversations more or less trustworthy

4. **Advanced: Add your own templates**
   - Learn how to create templates in `data_generation/patterns.py`
   - Add templates for your specific research domain

## Troubleshooting

### Common Issues

1. **ImportError: No module named 'research_generator'**
   - Make sure you installed the package in development mode:
   ```bash
   pip install -e .
   ```

2. **Memory errors when generating large datasets**
   - Reduce the 'size' in your configuration
   - Generate data in smaller batches

3. **Slow generation or analysis**
   - Start with smaller datasets first
   - Use CPU-only mode if you don't have a good GPU

### Getting Help
- Check the detailed documentation in `docs/`
- Look at the test files in `tests/` for usage examples
- Ask questions in the GitHub Issues section

## Next Steps

After you're comfortable with the basics:
1. Try the advanced configuration options
2. Explore how edge cases affect trust scores
3. Contribute templates for new research domains
4. Experiment with different model parameters

Happy coding!