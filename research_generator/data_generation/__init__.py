"""
Data generation module for research conversation generation
Provides comprehensive template-based generation capabilities
"""
from enum import Enum
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Union
# Import models
from .models import (
    ConversationPhase,
    ConversationStyle,
    AIInteractionType,
    ResearchContext,
    AIInteraction,
    ContentProvenance
)

# Import core components
from .patterns import PatternManager
from .edge_cases import EdgeCaseManager
from .metrics import MetricsCalculator

from .generator import (
    UnifiedResearchGeneratorA,
    UnifiedResearchGenerator,
    ResearchContext
)
from .patterns import (
    PatternManager,
    ConversationPhase,
    ConversationStyle,
    ResearchTemplate
)
from .edge_cases import (
    EdgeCaseManager,
    EdgeCase,
    EdgeCaseType
)
# Import new components
from .ai_logger import AIInteractionLogger
from .c2pa_manager import C2PAManager

__all__ = [
    'ConversationPhase',
    'ConversationStyle',
    'AIInteractionType',
    'ResearchContext',
    'AIInteraction',
    'ContentProvenance',
    'PatternManager',
    'EdgeCaseManager',
    'MetricsCalculator',
    'AIInteractionLogger',
    'C2PAManager',
    'UnifiedResearchGenerator',
    'classify_content_authenticity',
    'calculate_trust_score', 
    'validate_dataset_diversity',
    'get_verification_status'
]

# Version information
__version__ = "1.0.0"

# Template configuration
TEMPLATE_DEFAULTS = {
    'use_dynamic_templates': True,
    'allow_nested_templates': True,
    'context_sensitivity': 0.8,
    'template_paths': {
        'base': 'templates/base',
        'domain_specific': 'templates/domains',
        'custom': 'templates/custom'
    }
}

# Research domains configuration
RESEARCH_DOMAINS = {
    'education': {
        'patterns': ['pedagogical', 'learning', 'assessment'],
        'complexity_range': (0.4, 1.0),
        'required_components': ['methodology', 'theoretical_framework']
    },
    'psychology': {
        'patterns': ['behavioral', 'cognitive', 'clinical'],
        'complexity_range': (0.5, 1.0),
        'required_components': ['methodology', 'sample_characteristics']
    },
    'stem': {
        'patterns': ['empirical', 'computational', 'experimental'],
        'complexity_range': (0.6, 1.0),
        'required_components': ['methodology', 'data_collection']
    }
}

# Edge case configuration
EDGE_CASE_CONFIG = {
    'probability_ranges': {
        'contradiction': (0.1, 0.2),
        'methodology_mismatch': (0.08, 0.15),
        'citation_error': (0.05, 0.1),
        'theoretical_inconsistency': (0.07, 0.12)
    },
    'severity_levels': {
        'low': 0.3,
        'medium': 0.6,
        'high': 0.8
    }
}

# Pattern generation configuration
PATTERN_CONFIG = {
    'conversation_flow': {
        'min_exchanges': 3,
        'max_exchanges': 15,
        'transition_probability': 0.3
    },
    'template_complexity': {
        'basic': 0.3,
        'intermediate': 0.6,
        'advanced': 0.9
    }
}

# Example usage:
def create_research_context() -> ResearchContext:
    return ResearchContext(
        domain="education",
        topic="Cognitive Load in Online Learning",
        methodology="Mixed Methods Research",
        theoretical_framework="Cognitive Load Theory",
        complexity=0.8,
        phase=ConversationPhase.LITERATURE_REVIEW,
        style=ConversationStyle.ANALYTICAL,
        research_questions=[
            "How does cognitive load affect student performance?",
            "What strategies reduce cognitive load?"
        ],
        citations=[{
            "author": "Smith et al.",
            "year": "2023",
            "title": "Cognitive Load Analysis",
            "doi": "10.1000/example.123"
        }],
        variables={
            "dependent_var": "learning_performance",
            "independent_var": "cognitive_load_level",
            "effect_size": 0.75
        },
        ai_model={
            "name": "GPT-4",
            "version": "1.0",
            "provider": "OpenAI"
        },
        ai_interaction_history=[],
        content_provenance={}
    )

# def create_generator(config: dict = None) -> UnifiedResearchGenerator:
#     """
#     Create a preconfigured UnifiedResearchGenerator instance
    
#     Args:
#         config: Optional configuration override
        
#     Returns:
#         Configured UnifiedResearchGenerator instance
#     """
#     default_config = {
#         'template_settings': TEMPLATE_DEFAULTS,
#         'research_domains': RESEARCH_DOMAINS,
#         'edge_cases': EDGE_CASE_CONFIG,
#         'patterns': PATTERN_CONFIG
#     }
    
#     if config:
#         # Deep merge configuration
#         merged_config = _deep_merge_configs(default_config, config)
#     else:
#         merged_config = default_config
    
#     return UnifiedResearchGenerator(merged_config)

def create_generator(config: dict = None) -> UnifiedResearchGeneratorA:
    """
    Create a preconfigured UnifiedResearchGenerator instance
    
    Args:
        config: Optional configuration override
        
    Returns:
        Configured UnifiedResearchGenerator instance
    """
    default_config = {
        'template_settings': TEMPLATE_DEFAULTS,
        'research_domains': RESEARCH_DOMAINS,
        'edge_cases': EDGE_CASE_CONFIG,
        'patterns': PATTERN_CONFIG,
        # Add generation configuration
        'generation': {
            'size': 5000,
            'min_length': 5,
            'max_length': 20,
            'domains': ['education', 'psychology', 'stem'],
            'complexity_levels': ['basic', 'medium', 'complex'],
            'edge_case_ratio': 0.2
        }
    }
    
    if config:
        # Deep merge configuration
        merged_config = _deep_merge_configs(default_config, config)
    else:
        merged_config = default_config
    
    return UnifiedResearchGeneratorA(merged_config)

def create_pattern_manager(config: dict = None) -> PatternManager:
    """
    Create a preconfigured PatternManager instance
    
    Args:
        config: Optional pattern configuration override
        
    Returns:
        Configured PatternManager instance
    """
    pattern_config = config if config else PATTERN_CONFIG
    return PatternManager(pattern_config)

def create_edge_case_manager(config: dict = None) -> EdgeCaseManager:
    """
    Create a preconfigured EdgeCaseManager instance
    
    Args:
        config: Optional edge case configuration override
        
    Returns:
        Configured EdgeCaseManager instance
    """
    edge_config = config if config else EDGE_CASE_CONFIG
    return EdgeCaseManager(edge_config)

def _deep_merge_configs(base: dict, override: dict) -> dict:
    """
    Deep merge two configuration dictionaries
    
    Args:
        base: Base configuration
        override: Override configuration
        
    Returns:
        Merged configuration dictionary
    """
    merged = base.copy()
    
    for key, value in override.items():
        if (
            key in merged and 
            isinstance(merged[key], dict) and 
            isinstance(value, dict)
        ):
            merged[key] = _deep_merge_configs(merged[key], value)
        else:
            merged[key] = value
    
    return merged

# Configuration validation
def validate_config(config: dict) -> bool:

    """
    Validate configuration dictionary
    """
    required_keys = [
        'template_settings',
        'research_domains',
        'edge_cases',
        'patterns',
        'generation'  # Add generation configuration check
    ]

    """
    Validate configuration dictionary
    
    Args:
        config: Configuration to validate
        
    Returns:
        bool indicating if configuration is valid
        
    Raises:
        ValueError: If configuration is invalid
    """
    required_keys = [
        'template_settings',
        'research_domains',
        'edge_cases',
        'patterns'
    ]
    
    # Check required keys
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required configuration key: {key}")
    
    # Validate template settings
    template_settings = config['template_settings']
    if not isinstance(template_settings.get('context_sensitivity'), (int, float)):
        raise ValueError("Template context_sensitivity must be a number")
    
    # Validate research domains
    for domain, settings in config['research_domains'].items():
        if 'patterns' not in settings or 'complexity_range' not in settings:
            raise ValueError(f"Invalid domain configuration for {domain}")
    
    # Validate edge cases
    edge_cases = config['edge_cases']
    if 'probability_ranges' not in edge_cases or 'severity_levels' not in edge_cases:
        raise ValueError("Invalid edge case configuration")

    # Validate generation configuration
    generation_config = config.get('generation', {})
    required_generation_keys = ['size', 'min_length', 'max_length', 'domains']
    for key in required_generation_keys:
        if key not in generation_config:
            raise ValueError(f"Missing required generation configuration key: {key}")
    
    return True

# Export public interface
__all__ = [
    'UnifiedResearchGenerator',
    'ResearchContext',
    'PatternManager',
    'ConversationPhase',
    'ConversationStyle',
    'ResearchTemplate',
    'EdgeCaseManager',
    'EdgeCase',
    'EdgeCaseType',
    'MetricsCalculator',
    'create_generator',
    'create_pattern_manager',
    'create_edge_case_manager',
    'validate_config',
    'TEMPLATE_DEFAULTS',
    'RESEARCH_DOMAINS',
    'EDGE_CASE_CONFIG',
    'PATTERN_CONFIG'
]

# from research_generator import UnifiedResearchGenerator


# default_config = {
#     'size': 1000,                # Number of conversations
#     'min_length': 5,             # Minimum messages per conversation
#     'max_length': 20,            # Maximum messages per conversation
#     'edge_case_ratio': 0.2,      # Proportion of conversations with edge cases
#     'domains': ['all'],          # Research domains to include
#     'complexity_levels': ['basic', 'medium', 'complex']
# }

# advanced_config = {
#     'size': 1000,
#     'min_length': 5,
#     'max_length': 20,
#     'edge_case_ratio': 0.2,
#     'domains': ['education', 'psychology', 'stem'],
#     'complexity_levels': ['basic', 'medium', 'complex'],
#     'metrics': {
#         'authenticity': {
#             'weights': {
#                 'methodology_score': 0.3,
#                 'theoretical_alignment': 0.3,
#                 'citation_quality': 0.2,
#                 'consistency': 0.2
#             }
#         },
#         'edge_cases': {
#             'min_probability': 0.05,
#             'max_probability': 0.15
#         }
#     }
# }

# # Initialize generator
# generator = UnifiedResearchGenerator(config={
#     'size': 1000,
#     'edge_case_ratio': 0.2,
#     'domains': ['education', 'psychology', 'stem'],
#     'complexity_levels': ['basic', 'medium', 'complex']
# })

# # Generate dataset
# conversations, metrics = generator.generate_dataset()

# # Save results
# generator.save_dataset(conversations, metrics, base_filename='research_data')