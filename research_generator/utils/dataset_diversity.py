import random
import statistics
from typing import List, Dict, Union

# Modify the import to use absolute import
from research_generator.data_generation.models import (
    ResearchContext, 
    ConversationPhase, 
    ConversationStyle
)
def classify_content_authenticity(
    ai_interaction_count: int, 
    message_count: int, 
    complexity: float
) -> str:
    """
    Sophisticated content authenticity classification
    
    Args:
        ai_interaction_count: Number of AI interactions
        message_count: Total number of messages
        complexity: Research complexity score
    
    Returns:
        Content authenticity classification
    """
    ai_influence_ratio = ai_interaction_count / max(message_count / 2, 1)
    
    # Incorporate research complexity as a modifier
    complexity_modifier = 1 + (complexity - 0.5)
    
    # Enhanced classification logic
    if ai_influence_ratio * complexity_modifier > 0.7:
        return "ai_generated"
    elif ai_influence_ratio * complexity_modifier > 0.3:
        return "ai_assisted"
    else:
        return "human_generated"

def calculate_trust_score(
    context: ResearchContext,
    verification_status: str,
    ai_interactions: List[Dict]
) -> float:
    """
    Advanced trust score calculation
    
    Args:
        context: Research context
        verification_status: Verification status
        ai_interactions: List of AI interactions
    
    Returns:
        Nuanced trust score between 0.1 and 0.9
    """
    # Base trust factors
    base_factors = {
        'citation_quality': len(context.citations) / 5,
        'research_complexity': context.complexity,
        'ai_interaction_diversity': len(set(
            interaction.get('ai_model', {}).get('name', '') 
            for interaction in ai_interactions
        )) / max(len(ai_interactions), 1),
        'verification_bonus': {
            'verified': 1.0,
            'partially_verified': 0.7,
            'unverified': 0.3
        }.get(verification_status, 0.5)
    }
    
    # Weighted calculation
    trust_score = sum([
        base_factors['citation_quality'] * 0.2,
        base_factors['research_complexity'] * 0.2,
        base_factors['ai_interaction_diversity'] * 0.3,
        base_factors['verification_bonus'] * 0.3
    ])
    
    # Add controlled randomness
    trust_score += random.uniform(-0.1, 0.1)
    
    return round(max(0.1, min(0.9, trust_score)), 2)

def validate_dataset_diversity(conversations: List[Dict]) -> Dict:
    """
    Validate dataset diversity and provide insights
    
    Args:
        conversations: Generated conversations
    
    Returns:
        Diversity metrics and potential recommendations
    """
    # Analyze distributions
    verification_status = [
        conv.get('c2pa_provenance', {}).get('verification_status', 'unverified') 
        for conv in conversations
    ]
    
    authenticity_types = [
        conv.get('content_authenticity', 'unknown') 
        for conv in conversations
    ]
    
    trust_scores = [
        conv.get('trust_score', 0.0) 
        for conv in conversations
    ]
    
    diversity_report = {
        'verification_status_distribution': {
            status: verification_status.count(status) / len(conversations) * 100 
            for status in set(verification_status)
        },
        'content_authenticity_distribution': {
            auth: authenticity_types.count(auth) / len(conversations) * 100 
            for auth in set(authenticity_types)
        },
        'trust_score_stats': {
            'mean': statistics.mean(trust_scores) if trust_scores else 0,
            'median': statistics.median(trust_scores) if trust_scores else 0,
            'min': min(trust_scores) if trust_scores else 0,
            'max': max(trust_scores) if trust_scores else 0,
            'std_dev': statistics.stdev(trust_scores) if len(trust_scores) > 1 else 0
        },
        'recommendations': []
    }
    
    # Add recommendations based on distribution
    if diversity_report['verification_status_distribution'].get('verified', 0) < 20:
        diversity_report['recommendations'].append(
            "Consider increasing the proportion of verified conversations"
        )
    
    if diversity_report['content_authenticity_distribution'].get('human_generated', 0) < 15:
        diversity_report['recommendations'].append(
            "Increase diversity by generating more human-generated content"
        )
    
    return diversity_report

def get_verification_status() -> str:
    """
    Generate verification status with controlled distribution
    
    Returns:
        Verification status string
    """
    verification_probability_weights = {
        "verified": 0.3,        # 30% verified
        "partially_verified": 0.4,  # 40% partially verified 
        "unverified": 0.3       # 30% unverified
    }
    
    return random.choices(
        list(verification_probability_weights.keys()),
        weights=list(verification_probability_weights.values())
    )[0]