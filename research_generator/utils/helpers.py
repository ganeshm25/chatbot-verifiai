"""
Helper functions for research generator
"""

import json
import yaml
import uuid
from typing import Dict, List, Union, Optional, Any
from datetime import datetime, timedelta
import re
from pathlib import Path
import pandas as pd
import numpy as np
from collections import Counter
from dataclasses import asdict
from enum import Enum

def load_config(config_path: Union[str, Path]) -> Dict:
    """
    Load configuration from JSON or YAML file
    
    Args:
        config_path: Path to configuration file
    
    Returns:
        Dict containing configuration
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(path, 'r') as f:
        if path.suffix in ['.yaml', '.yml']:
            return yaml.safe_load(f)
        elif path.suffix == '.json':
            return json.load(f)
        else:
            raise ValueError(f"Unsupported config file format: {path.suffix}")

def save_dataset(
    data: Dict[str, Any],
    output_path: Union[str, Path],
    formats: Optional[List[str]] = None
) -> None:
    """
    Save dataset in multiple formats
    
    Args:
        data: Dictionary containing conversations and metrics
        output_path: Base path for output files
        formats: List of formats to save (csv, json, parquet)
    """
    formats = formats or ['csv', 'json']
    path = Path(output_path)
    
    # Ensure output directory exists
    path.parent.mkdir(parents=True, exist_ok=True)
    
    for fmt in formats:
        if fmt == 'csv':
            # Save conversations
            conv_df = pd.DataFrame(flatten_conversations(data['conversations']))
            conv_df.to_csv(f"{path}_conversations.csv", index=False)
            
            # Save metrics
            metrics_df = pd.DataFrame(data['metrics'])
            metrics_df.to_csv(f"{path}_metrics.csv", index=False)
            
        elif fmt == 'json':
            with open(f"{path}_complete.json", 'w') as f:
                json.dump(data, f, indent=2, default=str)
                
        elif fmt == 'parquet':
            # Save conversations
            conv_df = pd.DataFrame(flatten_conversations(data['conversations']))
            conv_df.to_parquet(f"{path}_conversations.parquet")
            
            # Save metrics
            metrics_df = pd.DataFrame(data['metrics'])
            metrics_df.to_parquet(f"{path}_metrics.parquet")

def validate_conversation(conversation: Dict) -> bool:
    """
    Validate conversation structure and content
    
    Args:
        conversation: Conversation dictionary to validate
    
    Returns:
        bool indicating if conversation is valid
    """
    required_fields = {
        'id': str,
        'timestamp': str,
        'context': dict,
        'messages': list
    }
    
    # Check required fields
    for field, field_type in required_fields.items():
        if field not in conversation:
            raise ValueError(f"Missing required field: {field}")
        if not isinstance(conversation[field], field_type):
            raise TypeError(f"Invalid type for {field}: expected {field_type}")
    
    # Validate messages
    for msg in conversation['messages']:
        validate_message(msg)
    
    # Validate context
    validate_research_context(conversation['context'])
    
    return True

def validate_message(message: Dict) -> bool:
    """
    Validate message structure and content
    
    Args:
        message: Message dictionary to validate
    
    Returns:
        bool indicating if message is valid
    """
    required_fields = {
        'id': str,
        'timestamp': str,
        'role': str,
        'content': str,
        'metadata': dict
    }
    
    for field, field_type in required_fields.items():
        if field not in message:
            raise ValueError(f"Missing required field in message: {field}")
        if not isinstance(message[field], field_type):
            raise TypeError(f"Invalid type for message {field}: expected {field_type}")
    
    if message['role'] not in ['researcher', 'assistant']:
        raise ValueError(f"Invalid role: {message['role']}")
    
    return True

def validate_research_context(context: Dict) -> bool:
    """
    Validate research context
    
    Args:
        context: Research context dictionary to validate
    
    Returns:
        bool indicating if context is valid
    """
    required_fields = {
        'domain': str,
        'topic': str,
        'methodology': str,
        'theoretical_framework': str,
        'research_questions': list,
        'complexity': float,
        'phase': str
    }
    
    for field, field_type in required_fields.items():
        if field not in context:
            raise ValueError(f"Missing required field in context: {field}")
        if not isinstance(context[field], field_type):
            raise TypeError(f"Invalid type for context {field}: expected {field_type}")
    
    if not 0 <= context['complexity'] <= 1:
        raise ValueError("Complexity must be between 0 and 1")
    
    return True

def generate_timestamp(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None
) -> str:
    """
    Generate random timestamp within range
    
    Args:
        start_date: Start date for range (default: 30 days ago)
        end_date: End date for range (default: now)
    
    Returns:
        ISO format timestamp string
    """
    end_date = end_date or datetime.now()
    start_date = start_date or (end_date - timedelta(days=30))
    
    time_range = end_date - start_date
    random_seconds = random.uniform(0, time_range.total_seconds())
    random_date = start_date + timedelta(seconds=random_seconds)
    
    return random_date.isoformat()

def format_citations(citations: List[str], style: str = 'apa') -> List[str]:
    """
    Format citations according to specified style
    
    Args:
        citations: List of citation strings
        style: Citation style ('apa', 'mla', 'chicago')
    
    Returns:
        List of formatted citations
    """
    if style == 'apa':
        return [format_apa_citation(cite) for cite in citations]
    elif style == 'mla':
        return [format_mla_citation(cite) for cite in citations]
    elif style == 'chicago':
        return [format_chicago_citation(cite) for cite in citations]
    else:
        raise ValueError(f"Unsupported citation style: {style}")

def clean_text(text: str) -> str:
    """
    Clean and normalize text
    
    Args:
        text: Input text string
    
    Returns:
        Cleaned text string
    """
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    # Normalize quotes
    text = text.replace('"', '"').replace('"', '"')
    
    # Fix common typos
    text = text.replace('..', '.')
    text = text.replace(',.', '.')
    
    # Ensure proper spacing after punctuation
    text = re.sub(r'([.!?])([A-Za-z])', r'\1 \2', text)
    
    return text.strip()

def calculate_text_metrics(text: str) -> Dict[str, float]:
    """
    Calculate various text metrics
    
    Args:
        text: Input text string
    
    Returns:
        Dictionary of text metrics
    """
    words = text.split()
    sentences = re.split(r'[.!?]+', text)
    
    return {
        'word_count': len(words),
        'sentence_count': len(sentences),
        'avg_word_length': np.mean([len(word) for word in words]),
        'avg_sentence_length': len(words) / max(len(sentences), 1),
        'unique_words_ratio': len(set(words)) / len(words) if words else 0
    }

def load_templates(template_path: Union[str, Path]) -> Dict[str, List[str]]:
    """
    Load conversation templates from file
    
    Args:
        template_path: Path to templates file
    
    Returns:
        Dictionary of templates by category
    """
    path = Path(template_path)
    if not path.exists():
        raise FileNotFoundError(f"Templates file not found: {template_path}")
    
    with open(path, 'r') as f:
        if path.suffix in ['.yaml', '.yml']:
            return yaml.safe_load(f)
        elif path.suffix == '.json':
            return json.load(f)
        else:
            raise ValueError(f"Unsupported template file format: {path.suffix}")

def generate_unique_id(prefix: str = '') -> str:
    """
    Generate unique identifier
    
    Args:
        prefix: Optional prefix for ID
    
    Returns:
        Unique ID string
    """
    unique_id = str(uuid.uuid4())
    return f"{prefix}_{unique_id}" if prefix else unique_id

def flatten_conversations(conversations: List[Dict]) -> List[Dict]:
    """
    Flatten nested conversation structure for DataFrame format
    
    Args:
        conversations: List of conversation dictionaries
    
    Returns:
        List of flattened conversation dictionaries
    """
    flattened = []
    for conv in conversations:
        context = conv['context']
        for msg in conv['messages']:
            flat_msg = {
                'conversation_id': conv['id'],
                'timestamp': msg['timestamp'],
                **msg,
                **{f'context_{k}': v for k, v in context.items()}
            }
            flattened.append(flat_msg)
    return flattened

def format_apa_citation(citation: str) -> str:
    """Format citation in APA style"""
    # Implementation for APA formatting
    pass

def format_mla_citation(citation: str) -> str:
    """Format citation in MLA style"""
    # Implementation for MLA formatting
    pass

def format_chicago_citation(citation: str) -> str:
    """Format citation in Chicago style"""
    # Implementation for Chicago formatting
    pass


def serialize_for_json(obj):
    """Custom serializer that handles Enum types, dataclasses, and datetime objects"""
    if isinstance(obj, Enum):
        return obj.value
    elif hasattr(obj, '__dataclass_fields__'):  # Check if it's a dataclass
        return {k: serialize_for_json(v) for k, v in asdict(obj).items()}
    elif isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, (list, tuple)):
        return [serialize_for_json(x) for x in obj]
    elif isinstance(obj, dict):
        return {k: serialize_for_json(v) for k, v in obj.items()}
    return obj

class EnhancedJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        return serialize_for_json(obj)