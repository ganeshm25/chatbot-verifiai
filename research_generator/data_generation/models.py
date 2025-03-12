# File: research_generator/data_generation/models.py

from enum import Enum
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Union

# Existing enums
class ConversationPhase(Enum):
    INTRODUCTION = "introduction"
    LITERATURE_REVIEW = "literature_review"
    METHODOLOGY = "methodology"
    ANALYSIS = "analysis"
    FINDINGS = "findings"
    DISCUSSION = "discussion"
    CONCLUSION = "conclusion"

class ConversationStyle(Enum):
    FORMAL = "formal"
    ANALYTICAL = "analytical"
    EXPLORATORY = "exploratory"
    CRITICAL = "critical"
    COLLABORATIVE = "collaborative"

# New enum for AI interaction types
class AIInteractionType(Enum):
    TEXT_GENERATION = "text_generation"
    CONTENT_ANALYSIS = "content_analysis"
    RESEARCH_ASSISTANCE = "research_assistance"
    CITATION_CHECK = "citation_check"
    SUMMARIZATION = "summarization"

# Enhanced research context 
@dataclass
class ResearchContext:
    # Existing fields
    domain: str
    topic: str
    methodology: str
    theoretical_framework: str
    complexity: float
    phase: ConversationPhase
    style: ConversationStyle
    research_questions: List[str]
    citations: List[Dict[str, str]]
    variables: Dict[str, Union[str, float]]
    
    # New fields for AI and C2PA
    ai_model: Dict[str, str] = None
    ai_interaction_history: List[Dict] = None
    content_provenance: Dict = None
    
    def __post_init__(self):
        # Initialize default values for new fields if not provided
        if self.ai_model is None:
            self.ai_model = {}
        if self.ai_interaction_history is None:
            self.ai_interaction_history = []
        if self.content_provenance is None:
            self.content_provenance = {}

# New AI Interaction model
@dataclass
class AIInteraction:
    interaction_id: str
    timestamp: datetime
    content_id: str
    interaction_type: AIInteractionType
    input: Dict[str, str]
    output: Dict[str, str]
    user_actions: List[Dict[str, Union[str, datetime]]]
    ai_model: Dict[str, str]
    metadata: Dict[str, any]

class VerificationStatus(Enum):
    VERIFIED = "verified"
    PARTIALLY_VERIFIED = "partially_verified"
    UNVERIFIED = "unverified"
    DISPUTED = "disputed"
    PENDING = "pending"

@dataclass
class ContentProvenance:
    content_id: str
    user_id: str
    publication_timestamp: datetime
    interaction_summary: Dict[str, any]
    content_metadata: Dict[str, str]
    verification_status: VerificationStatus  # Change from string to enum
    verification_details: Dict[str, any] = None  # Add more detailed verification information

    def __post_init__(self):
        # Ensure verification status is an enum
        if isinstance(self.verification_status, str):
            self.verification_status = VerificationStatus(self.verification_status)
        
        # Initialize verification details if not provided
        if self.verification_details is None:
            self.verification_details = {
                "ai_interaction_score": 0.0,
                "citation_quality": 0.0,
                "methodology_rigor": 0.0
            }