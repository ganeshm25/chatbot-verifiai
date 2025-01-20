"""
Tests for data generation components
"""

import pytest
import torch
import numpy as np
from typing import Dict, List
from research_generator.data_generation.generator import UnifiedResearchGenerator
from research_generator.data_generation.edge_cases import EdgeCaseManager
from research_generator.data_generation.patterns import PatternManager

@pytest.fixture
def test_config():
    """Test configuration fixture"""
    return {
        'size': 10,
        'min_length': 3,
        'max_length': 5,
        'edge_case_ratio': 0.2,
        'domains': ['education', 'psychology'],
        'complexity_levels': ['basic', 'medium']
    }

@pytest.fixture
def generator(test_config):
    """Generator fixture"""
    return UnifiedResearchGenerator(test_config)

class TestUnifiedResearchGenerator:
    """Test cases for UnifiedResearchGenerator"""
    
    def test_initialization(self, generator):
        """Test generator initialization"""
        assert isinstance(generator.pattern_manager, PatternManager)
        assert isinstance(generator.edge_case_manager, EdgeCaseManager)
        assert generator.config['size'] == 10
    
    def test_research_context_generation(self, generator):
        """Test research context generation"""
        context = generator._generate_research_context()
        
        assert context.domain in generator.config['domains']
        assert 0 <= context.complexity <= 1
        assert len(context.research_questions) >= 2
        assert context.methodology in generator.domains[context.domain]['methodologies']
    
    def test_conversation_generation(self, generator):
        """Test conversation generation"""
        context = generator._generate_research_context()
        conversation = generator._generate_conversation(context)
        
        assert 'id' in conversation
        assert 'timestamp' in conversation
        assert 'context' in conversation
        assert 'messages' in conversation
        
        messages = conversation['messages']
        assert len(messages) >= generator.config['min_length'] * 2
        assert len(messages) <= generator.config['max_length'] * 2
    
    def test_edge_case_injection(self, generator):
        """Test edge case injection"""
        conversations, _ = generator.generate_dataset()
        
        edge_cases = sum(
            1 for conv in conversations
            if any(msg.get('type', '').startswith('edge_') for msg in conv['messages'])
        )
        
        expected_edge_cases = int(generator.config['size'] * generator.config['edge_case_ratio'])
        assert abs(edge_cases - expected_edge_cases) <= 1
    
    def test_dataset_generation(self, generator):
        """Test complete dataset generation"""
        conversations, metrics = generator.generate_dataset()
        
        assert len(conversations) == generator.config['size']
        assert len(metrics) == generator.config['size']
        
        # Check conversation structure
        for conv in conversations:
            assert self._validate_conversation_structure(conv)
        
        # Check metrics structure
        for metric in metrics:
            assert self._validate_metrics_structure(metric)
    
    def test_domain_specific_content(self, generator):
        """Test domain-specific content generation"""
        for domain in generator.config['domains']:
            context = generator._generate_research_context()
            assert context.topic in generator.domains[domain]['topics']
            assert context.methodology in generator.domains[domain]['methodologies']
            assert context.theoretical_framework in generator.domains[domain]['frameworks']
    
    def test_conversation_patterns(self, generator):
        """Test conversation pattern generation"""
        context = generator._generate_research_context()
        conversation = generator._generate_conversation(context)
        
        # Check researcher-assistant alternation
        messages = conversation['messages']
        for i in range(0, len(messages), 2):
            assert messages[i]['role'] == 'researcher'
            if i + 1 < len(messages):
                assert messages[i + 1]['role'] == 'assistant'
    
    @pytest.mark.parametrize("complexity", [0.3, 0.7, 1.0])
    def test_complexity_levels(self, generator, complexity):
        """Test different complexity levels"""
        context = generator._generate_research_context()
        context.complexity = complexity
        conversation = generator._generate_conversation(context)
        
        if complexity < 0.5:
            assert len(conversation['messages']) <= generator.config['max_length']
        else:
            assert any(msg.get('content', '').count('.') > 2 for msg in conversation['messages'])
    
    def _validate_conversation_structure(self, conversation: Dict) -> bool:
        """Validate conversation structure"""
        required_fields = {'id', 'timestamp', 'context', 'messages'}
        assert all(field in conversation for field in required_fields)
        
        for message in conversation['messages']:
            assert 'id' in message
            assert 'timestamp' in message
            assert 'role' in message
            assert 'content' in message
            assert message['role'] in {'researcher', 'assistant'}
        
        return True
    
    def _validate_metrics_structure(self, metric: Dict) -> bool:
        """Validate metrics structure"""
        required_metrics = {
            'methodology_metrics',
            'theoretical_metrics',
            'authenticity_metrics',
            'edge_case_metrics'
        }
        assert all(field in metric for field in required_metrics)
        
        # Check metric values
        for category in required_metrics:
            assert isinstance(metric[category], dict)
            assert all(0 <= v <= 1 for v in metric[category].values())
        
        return True
    
    def test_save_and_load(self, generator, tmp_path):
        """Test dataset saving and loading"""
        # Generate dataset
        conversations, metrics = generator.generate_dataset()
        
        # Save dataset
        save_path = tmp_path / "test_dataset"
        generator.save_dataset(conversations, metrics, save_path)
        
        # Check saved files
        assert (save_path.with_suffix('.csv')).exists()
        assert (save_path.with_name(f"{save_path.stem}_metrics.csv")).exists()
        assert (save_path.with_suffix('.json')).exists()
    
    @pytest.mark.parametrize("invalid_config", [
        {'size': -1},
        {'min_length': 0},
        {'edge_case_ratio': 1.5},
        {'domains': ['invalid_domain']}
    ])
    def test_invalid_configurations(self, invalid_config):
        """Test handling of invalid configurations"""
        with pytest.raises(ValueError):
            UnifiedResearchGenerator(invalid_config)
    
    def test_reproducibility(self, test_config):
        """Test reproducibility with same random seed"""
        generator1 = UnifiedResearchGenerator(test_config)
        generator2 = UnifiedResearchGenerator(test_config)
        
        conv1, metrics1 = generator1.generate_dataset()
        conv2, metrics2 = generator2.generate_dataset()
        
        # Compare conversations
        assert len(conv1) == len(conv2)
        for c1, c2 in zip(conv1, conv2):
            assert c1['context'] == c2['context']
            assert len(c1['messages']) == len(c2['messages'])
    
    @pytest.mark.asyncio
    async def test_async_generation(self, generator):
        """Test asynchronous dataset generation"""
        conversations, metrics = await generator.generate_dataset_async()
        assert len(conversations) == generator.config['size']
        assert len(metrics) == generator.config['size']