"""
Test suite for research generator
"""

import pytest
import torch
import numpy as np
from pathlib import Path

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Test data path
TEST_DATA_PATH = Path(__file__).parent / "test_data"
TEST_DATA_PATH.mkdir(exist_ok=True)