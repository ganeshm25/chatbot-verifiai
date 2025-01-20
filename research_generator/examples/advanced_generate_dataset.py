import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Any
from data_generation import EDGE_CASE_CONFIG, PATTERN_CONFIG, RESEARCH_DOMAINS, UnifiedResearchGenerator

from data_generation import (
    create_generator, 
    validate_config
)

class AdvancedDatasetGenerator:
    def __init__(self, base_path: Path = Path('research_outputs')):
        self.base_path = base_path
        self.base_path.mkdir(exist_ok=True)
        logging.basicConfig(level=logging.INFO)
    
    async def generate_comprehensive_dataset(self) -> Dict[str, Any]:
        """Generate advanced dataset with sophisticated configurations"""
        config = {
            'template_settings': {  # Add this key
                'use_dynamic_templates': True,
                'context_sensitivity': 0.8
            },
            'generation': {
                'size': 20,
                'min_length': 5,
                'max_length': 20,
                'domains': ['education', 'psychology', 'stem'],
                'complexity_levels': ['basic', 'medium', 'complex']
            },
            'research_domains': RESEARCH_DOMAINS,  # Ensure this is included
            'edge_cases': EDGE_CASE_CONFIG,
            'patterns': PATTERN_CONFIG,
            'patterns': {
                'edge_case_ratio': 0.2,
                'noise_injection': 0.1,
                'conversation_flow': 'dynamic',
                'complexity_distribution': {
                    'basic': 0.3,
                    'medium': 0.5,
                    'complex': 0.2
                }
            },
            'authenticity': {
                'min_citations': 2,
                'required_methodology': True,
                'theoretical_framework': True,
                'source_verification': True
            },
            'content': {
                'research_phases': [
                    'literature_review',
                    'methodology_discussion',
                    'data_analysis',
                    'findings_interpretation'
                ],
                'interaction_patterns': [
                    'question_answer',
                    'elaboration_request',
                    'clarification_exchange',
                    'critical_discussion'
                ]
            }
        }
        
        # Validate configuration
        try:
            validate_config(config)
        except ValueError as e:
            logging.error(f"Invalid configuration: {e}")
            return {}
        
        # Create generator with advanced configuration
        generator = create_generator(config)
        
        # Progress tracking callback
        async def progress_callback(progress: float, status: str):
            logging.info(f"Progress: {progress:.2f}% - {status}")
        
        # Generate dataset
        conversations, metrics = await generator.generate_dataset()
        #            callback=progress_callback
        # Save dataset
        output_path = self.base_path / "advanced_dataset"
        output_path.mkdir(exist_ok=True)
        generator.save_dataset(
            conversations, 
            metrics, 
            base_filename=str(output_path / "comprehensive_research_data")
        )
        
        logging.info(f"Generated {len(conversations)} conversations")
        
        return {
            'conversations': conversations,
            'metrics': metrics,
            'config': config
        }
    
    async def generate_domain_specific_datasets(self) -> Dict[str, List[Dict]]:
        """Generate domain-specific datasets"""
        domain_configs = {
            'education': {
                'size': 20,
                'domains': ['education'],
                'complexity_levels': ['basic', 'medium']
            },
            'psychology': {
                'size': 20,
                'domains': ['psychology'],
                'complexity_levels': ['medium', 'complex']
            },
            'stem': {
                'size': 20,
                'domains': ['stem'],
                'complexity_levels': ['intermediate', 'complex']
            }
        }
        
        domain_datasets = {}
        
        for domain, config in domain_configs.items():
            generator = create_generator(config)
            conversations, metrics = await generator.generate_dataset()
            
            output_path = self.base_path / f"{domain}_dataset"
            output_path.mkdir(exist_ok=True)
            generator.save_dataset(
                conversations, 
                metrics, 
                base_filename=str(output_path / f"{domain}_research_data")
            )
            
            domain_datasets[domain] = {
                'conversations': conversations,
                'metrics': metrics
            }
        
        return domain_datasets

async def main():
    generator = AdvancedDatasetGenerator()
    
    # Generate comprehensive dataset
    await generator.generate_comprehensive_dataset()
    
    # Generate domain-specific datasets
    await generator.generate_domain_specific_datasets()

if __name__ == "__main__":
    asyncio.run(main())