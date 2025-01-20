"""
Model components for advanced research analysis
"""

from .training import (
    ResearchAnalysisModel, 
    TrainingPipeline, 
    TrainingConfig,
    CurriculumSampler
)
from .evaluation import (
    EnhancedModelEvaluator, 
    EvaluationMetrics,
    MultiTaskLearningMetrics,
    CurriculumLearningMetrics,
    ContentProvenance,
    RealTimeMetrics,
    BiasMetrics,
    ResearchMetrics
)

__all__ = [
    # Model and Training Components
    "ResearchAnalysisModel",
    "TrainingPipeline", 
    "TrainingConfig",
    "CurriculumSampler",
    
    # Evaluation Components
    "ModelEvaluator",
    "EvaluationMetrics",
    
    # Advanced Learning Metrics
    "MultiTaskLearningMetrics",
    "CurriculumLearningMetrics",
    
    # Detailed Metric Components
    "ContentProvenance",
    "RealTimeMetrics", 
    "BiasMetrics",
    "ResearchMetrics"
]