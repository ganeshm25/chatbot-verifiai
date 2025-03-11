# File: research_generator/config/default_config.py

"""
Enhanced configuration with AI and C2PA support
"""

# Default configuration with AI and C2PA settings
DEFAULT_CONFIG = {
    # Core generation settings
    'size': 1000,
    'min_length': 5,
    'max_length': 20,
    'edge_case_ratio': 0.2,
    'domains': ['education', 'psychology', 'stem'],
    'complexity_levels': ['basic', 'medium', 'complex'],
    
    # Template settings
    'template_settings': {
        'use_dynamic_templates': True,
        'allow_nested_templates': True,
        'context_sensitivity': 0.8
    },
    
    # Edge case settings
    'edge_case_settings': {
        'severity_threshold': 0.7,
        'max_per_conversation': 2
    },
    
    # AI settings - NEW
    'ai_settings': {
        'models': {
            'GPT-4': {
                'version': '1.0',
                'provider': 'OpenAI',
                'capabilities': ['text_generation', 'analysis', 'research_assistance']
            },
            'Claude': {
                'version': '2.0',
                'provider': 'Anthropic',
                'capabilities': ['research_assistance', 'summarization']
            },
            'PaLM': {
                'version': '1.0',
                'provider': 'Google',
                'capabilities': ['text_generation', 'analysis']
            }
        },
        'interaction_logging': True,
        'verification_required': True,
        'model_selection_strategy': 'capability_match',  # 'random', 'capability_match', 'specified'
        'default_model': 'GPT-4'
    },
    
    # C2PA settings - NEW
    'c2pa_settings': {
        'provenance_tracking': True,
        'verification_level': 'standard',  # 'basic', 'standard', 'advanced'
        'manifest_generation': True,
        'watermarking': True,
        'required_metadata': [
            'model_info',
            'interaction_history',
            'user_actions'
        ]
    },
    
    # Domain-specific settings
    'domain_settings': {
        'education': {
            'ai_models': ['GPT-4', 'Claude'],  # Preferred models for this domain
            'verification_level': 'advanced',  # Override default verification level
            'topics': [
                'Cognitive Load in Online Learning',
                'Social-Emotional Learning Impact',
                'Digital Literacy Development',
                'Inclusive Education Practices',
                'Assessment Methods Innovation'
            ]
        },
        'psychology': {
            'ai_models': ['Claude'],
            'verification_level': 'standard',
            'topics': [
                'Behavioral Intervention Efficacy',
                'Cognitive Development Patterns',
                'Mental Health Interventions',
                'Social Psychology Dynamics',
                'Neuropsychological Assessment'
            ]
        },
        'stem': {
            'ai_models': ['GPT-4', 'PaLM'],
            'verification_level': 'standard',
            'topics': [
                'Machine Learning Ethics',
                'Quantum Computing Applications',
                'Renewable Energy Systems',
                'Biotechnology Advances',
                'Data Science Methods'
            ]
        }
    }
}

# Configuration validation function
def validate_config(config: dict) -> bool:
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
        'ai_settings',  # New required key
        'c2pa_settings'  # New required key
    ]
    
    # Check required keys
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required configuration key: {key}")
    
    # Validate AI settings
    ai_settings = config['ai_settings']
    if not ai_settings.get('models'):
        raise ValueError("AI settings must include at least one model")
    
    # Validate C2PA settings
    c2pa_settings = config['c2pa_settings']
    if c2pa_settings.get('provenance_tracking') and not c2pa_settings.get('verification_level'):
        raise ValueError("C2PA settings must include verification_level when provenance_tracking is enabled")
    
    return True

# Helper functions for configuration
def get_domain_specific_config(config: dict, domain: str) -> dict:
    """
    Get domain-specific configuration
    
    Args:
        config: Main configuration
        domain: Domain to get configuration for
        
    Returns:
        Domain-specific configuration merged with defaults
    """
    domain_config = config.get('domain_settings', {}).get(domain, {})
    
    # Apply domain-specific overrides
    result = {**config}
    
    # Override AI settings if defined for domain
    if 'ai_models' in domain_config:
        result['ai_settings'] = {
            **result['ai_settings'],
            'domain_preferred_models': domain_config['ai_models']
        }
    
    # Override C2PA settings if defined for domain
    if 'verification_level' in domain_config:
        result['c2pa_settings'] = {
            **result['c2pa_settings'],
            'verification_level': domain_config['verification_level']
        }
    
    return result