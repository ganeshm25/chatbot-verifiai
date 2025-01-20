"""
Comprehensive test suite for research metrics evaluation
"""

import pytest
import torch
import numpy as np
from typing import Dict, List
from research_generator.models.evaluation import (
    EnhancedModelEvaluator,
    EvaluationMetrics,
    ContentProvenance,
    RealTimeMetrics,
    BiasMetrics,
    ResearchMetrics
)
from research_generator.data_generation.generator import UnifiedResearchGenerator
from torch.utils.data import Dataset, DataLoader

# Test Fixtures
@pytest.fixture
def mock_conversation():
    """Generate a mock research conversation"""
    return {
        'id': 'test-conv-001',
        'messages': [
            {
                'role': 'researcher',
                'content': 'How does cognitive load affect online learning outcomes according to Smith et al. (2023)?',
                'citations': ['Smith et al. (2023)'],
                'methodology': 'qualitative inquiry'
            },
            {
                'role': 'assistant',
                'content': 'According to Smith et al. (2023), cognitive load significantly impacts learning outcomes...',
                'citations': ['Smith et al. (2023)', 'Johnson (2022)'],
                'methodology_references': ['experimental design', 'statistical analysis']
            }
        ],
        'metadata': {
            'domain': 'education',
            'methodology': 'mixed methods',
            'theoretical_framework': 'cognitive load theory'
        }
    }

@pytest.fixture
def mock_dataset():
    """Create mock dataset for testing"""
    class MockResearchDataset(Dataset):
        def __init__(self, size=100):
            self.size = size
            self.generator = UnifiedResearchGenerator({
                'size': size,
                'min_length': 3,
                'max_length': 5
            })
            self.conversations, self.metrics = self.generator.generate_dataset()
        
        def __len__(self):
            return self.size
        
        def __getitem__(self, idx):
            return self.conversations[idx], self.metrics[idx]
    
    return MockResearchDataset()

@pytest.fixture
def mock_dataloader(mock_dataset):
    """Create mock dataloader"""
    return DataLoader(mock_dataset, batch_size=16, shuffle=True)

class TestContentProvenanceMetrics:
    """Test cases for content provenance tracking"""
    
    def test_citation_chain_completeness(self, test_evaluator, mock_conversation):
        """Test citation chain tracking and completeness"""
        metrics = test_evaluator._evaluate_single_provenance(mock_conversation)
        
        assert 0 <= metrics.citation_chain_completeness <= 1
        assert len(metrics.citation_chain) > 0
        assert all(isinstance(cite, str) for cite in metrics.citation_chain)
    
    def test_source_verification(self, test_evaluator, mock_conversation):
        """Test source verification scoring"""
        metrics = test_evaluator._evaluate_single_provenance(mock_conversation)
        
        # Check verification scores
        assert 0 <= metrics.source_verification_score <= 1
        assert hasattr(metrics, 'verification_details')
        assert 'verified_sources' in metrics.verification_details
    
    @pytest.mark.parametrize("transformation_type", [
        "paraphrase",
        "summary",
        "synthesis",
        "critical_analysis"
    ])
    def test_content_transformations(self, test_evaluator, mock_conversation, transformation_type):
        """Test different types of content transformations"""
        mock_conversation['messages'][1]['transformation_type'] = transformation_type
        metrics = test_evaluator._evaluate_single_provenance(mock_conversation)
        
        assert transformation_type in metrics.transformation_history
        assert 0 <= metrics.transformation_coverage <= 1

class TestRealTimeAnalysisMetrics:
    """Test cases for real-time analysis capabilities"""
    
    @pytest.mark.parametrize("response_type", [
        "immediate",
        "batch",
        "streaming"
    ])
    def test_response_latency(self, test_evaluator, mock_dataloader, response_type):
        """Test response time metrics for different types"""
        metrics = test_evaluator._evaluate_realtime(mock_dataloader, mode=response_type)
        
        assert metrics.response_latency > 0
        assert metrics.response_latency < test_evaluator.config['thresholds']['latency']
        
        if response_type == "immediate":
            assert metrics.response_latency < 50  # ms
    
    def test_trust_scoring_weights(self, test_evaluator, mock_conversation):
        """Test weighted trust score calculation"""
        weights = test_evaluator.config['weights']
        metrics = test_evaluator._evaluate_realtime_single(mock_conversation)
        
        expected_score = (
            metrics.citation_verification * weights['citation'] +
            metrics.methodology_consistency * weights['methodology'] +
            metrics.theoretical_alignment * weights['theoretical']
        )
        
        assert abs(metrics.trust_score - expected_score) < 0.01

class TestBiasDetectionMetrics:
    """Test cases for bias detection"""
    
    @pytest.mark.parametrize("bias_type,expected_indicators", [
        ("confirmation_bias", ["selective_citation", "contrary_evidence_omission"]),
        ("selection_bias", ["sample_bias", "data_exclusion"]),
        ("reporting_bias", ["positive_results_emphasis", "negative_results_omission"]),
        ("methodology_bias", ["method_misalignment", "inappropriate_analysis"])
    ])
    def test_specific_bias_detection(
        self,
        test_evaluator,
        mock_conversation,
        bias_type,
        expected_indicators
    ):
        """Test detection of specific bias types"""
        metrics = test_evaluator._evaluate_bias_single(mock_conversation)
        
        bias_score = getattr(metrics, bias_type)
        assert 0 <= bias_score <= 1
        
        bias_indicators = metrics.bias_indicators[bias_type]
        assert all(indicator in bias_indicators for indicator in expected_indicators)

class TestResearchQualityMetrics:
    """Test cases for research quality assessment"""
    
    def test_methodology_rigor(self, test_evaluator, mock_conversation):
        """Test methodology rigor scoring"""
        mock_conversation['metadata']['methodology_details'] = {
            'design': 'randomized controlled trial',
            'sample_size': 200,
            'analysis_methods': ['ANOVA', 'regression']
        }
        
        metrics = test_evaluator._evaluate_research_quality_single(mock_conversation)
        
        assert 0 <= metrics.methodology_rigor <= 1
        assert hasattr(metrics, 'methodology_subscores')
        assert 'design_quality' in metrics.methodology_subscores
    
    @pytest.mark.parametrize("evidence_type", [
        "empirical",
        "theoretical",
        "mixed"
    ])
    def test_evidence_strength(self, test_evaluator, mock_conversation, evidence_type):
        """Test evidence strength evaluation"""
        mock_conversation['metadata']['evidence_type'] = evidence_type
        metrics = test_evaluator._evaluate_research_quality_single(mock_conversation)
        
        assert 0 <= metrics.empirical_evidence <= 1
        assert hasattr(metrics, 'evidence_details')
        assert metrics.evidence_details['type'] == evidence_type

class TestComprehensiveEvaluation:
    """Test cases for comprehensive evaluation pipeline"""
    
    def test_full_evaluation_pipeline(self, test_evaluator, mock_dataloader):
        """Test complete evaluation pipeline"""
        metrics = test_evaluator.evaluate(mock_dataloader)
        
        # Check all metric components
        assert isinstance(metrics.provenance, ContentProvenance)
        assert isinstance(metrics.realtime, RealTimeMetrics)
        assert isinstance(metrics.bias, BiasMetrics)
        assert isinstance(metrics.research, ResearchMetrics)
        
        # Verify metric ranges
        assert all(0 <= getattr(metrics.realtime, field) <= 1
                  for field in ['trust_score', 'citation_verification',
                              'methodology_consistency', 'theoretical_alignment'])
        
        assert all(0 <= getattr(metrics.bias, field) <= 1
                  for field in ['confirmation_bias', 'selection_bias',
                              'reporting_bias', 'methodology_bias'])
    
    @pytest.mark.parametrize("evaluation_mode", [
        "strict",
        "lenient",
        "balanced"
    ])
    def test_evaluation_modes(self, test_evaluator, mock_dataloader, evaluation_mode):
        """Test different evaluation strictness modes"""
        metrics = test_evaluator.evaluate(
            mock_dataloader,
            mode=evaluation_mode
        )
        
        if evaluation_mode == "strict":
            assert metrics.research.methodology_rigor > 0.8
            assert metrics.bias.overall_bias_score < 0.2
        elif evaluation_mode == "lenient":
            assert metrics.research.methodology_rigor > 0.6
            assert metrics.bias.overall_bias_score < 0.3
    
    def test_report_generation(self, test_evaluator, mock_dataloader):
        """Test evaluation report generation"""
        metrics = test_evaluator.evaluate(mock_dataloader)
        report = test_evaluator.generate_comprehensive_report(metrics)
        
        # Verify report structure
        required_sections = [
            'Content Provenance',
            'Real-time Analysis',
            'Bias Detection',
            'Research Quality',
            'Overall Metrics'
        ]
        
        assert all(section in report.sections for section in required_sections)
        assert 'recommendations' in report.meta
        assert len(report.summary) > 0
    
    @pytest.mark.asyncio
    async def test_streaming_evaluation(self, test_evaluator, mock_dataloader):
        """Test streaming evaluation capabilities"""
        async for batch_metrics in test_evaluator.evaluate_streaming(mock_dataloader):
            assert isinstance(batch_metrics, EvaluationMetrics)
            assert hasattr(batch_metrics.realtime, 'streaming_latency')
            assert batch_metrics.realtime.streaming_latency < 50  # ms

if __name__ == '__main__':
    pytest.main([__file__])