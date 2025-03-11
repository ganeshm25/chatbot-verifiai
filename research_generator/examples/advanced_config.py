"""
Advanced configuration and usage scenarios for the Research Generator system
"""

import asyncio
import torch
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Callable, Union
from data_generation import UnifiedResearchGeneratorA
from models import ResearchAnalysisModel, TrainingPipeline
from models.evaluation import EnhancedModelEvaluator
from utils.helpers import load_config, save_dataset

class AdvancedResearchScenarios:
    """Advanced usage scenarios for research generation and analysis"""
    
    def __init__(self, base_path: Optional[Union[Path, str]] = None):
        self.base_path = Path(base_path) if base_path else Path("research_output")
        self.base_path.mkdir(exist_ok=True)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    async def scenario_1_advanced_generation(self):
        """Scenario 1: Advanced dataset generation with sophisticated patterns"""
        config = {
            'generation': {
                'size': 50,
                'min_length': 5,
                'max_length': 20,
                'domains': ['education', 'psychology', 'stem'],
                'complexity_levels': ['basic', 'medium', 'complex']
            },
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
        
        generator = UnifiedResearchGeneratorA(config)
        
        # Generate with progress tracking
        async def progress_callback(progress: float, status: str):
            print(f"Progress: {progress:.2f}% - {status}")
        
        conversations, metrics = await generator.generate_dataset()
        
        # Save with comprehensive metadata
        await self._save_advanced_dataset(conversations, metrics)
        return conversations, metrics
    
    async def scenario_2_curriculum_training(self):
        """Scenario 2: Advanced training with curriculum learning"""
        training_config = {
            'model': {
                'architecture': {
                    'type': 'transformer',
                    'hidden_size': 768,
                    'num_layers': 12,
                    'num_heads': 12,
                    'dropout': 0.1
                },
                'optimization': {
                    'optimizer': 'adamw',
                    'learning_rate': 2e-5,
                    'weight_decay': 0.01,
                    'gradient_clipping': 1.0,
                    'warmup_steps': 1000
                }
            },
            'curriculum': {
                'stages': [
                    {
                        'name': 'basic_understanding',
                        'epochs': 2,
                        'difficulty': 0.3,
                        'focus': ['trust_scoring', 'citation_verification']
                    },
                    {
                        'name': 'methodology_analysis',
                        'epochs': 3,
                        'difficulty': 0.6,
                        'focus': ['methodology_assessment', 'theoretical_alignment']
                    },
                    {
                        'name': 'advanced_analysis',
                        'epochs': 5,
                        'difficulty': 0.9,
                        'focus': ['bias_detection', 'quality_assessment']
                    }
                ],
                'advancement_criteria': {
                    'min_accuracy': 0.8,
                    'max_loss': 0.2,
                    'min_epochs': 2
                }
            },
            'evaluation': {
                'metrics': ['accuracy', 'f1', 'precision', 'recall'],
                'validation_frequency': 100,
                'early_stopping_patience': 3
            }
        }
        
        # Initialize training pipeline
        pipeline = TrainingPipeline(training_config)
        model = await pipeline.train_with_curriculum(
            callback=self._training_callback
        )
        
        return model
    
    async def scenario_3_comprehensive_evaluation(self):
        """Scenario 3: Comprehensive model evaluation"""
        evaluation_config = {
            'metrics': {
                'trust': {
                    'weights': {
                        'citation_quality': 0.3,
                        'methodology_rigor': 0.3,
                        'theoretical_alignment': 0.2,
                        'evidence_strength': 0.2
                    },
                    'thresholds': {
                        'high_trust': 0.8,
                        'medium_trust': 0.6,
                        'low_trust': 0.4
                    }
                },
                'bias': {
                    'types': [
                        'confirmation_bias',
                        'selection_bias',
                        'reporting_bias',
                        'methodology_bias'
                    ],
                    'detection_threshold': 0.15,
                    'pattern_recognition': True
                },
                'quality': {
                    'aspects': [
                        'methodology_rigor',
                        'theoretical_grounding',
                        'empirical_evidence',
                        'analytical_depth'
                    ],
                    'scoring': 'weighted_average'
                }
            },
            'analysis': {
                'granularity': 'detailed',
                'include_recommendations': True,
                'track_confidence': True,
                'assessment_level': 'comprehensive'
            }
        }
        
        model = await self._load_model()
        evaluator = EnhancedModelEvaluator(model, self.device, evaluation_config)
        
        # Run comprehensive evaluation
        metrics = await evaluator.evaluate_comprehensive()
        report = await self._generate_advanced_report(metrics)
        
        return metrics, report
    
    async def scenario_4_realtime_processing(self):
        """Scenario 4: Advanced real-time processing"""
        realtime_config = {
            'processing': {
                'max_latency': 100,  # ms
                'batch_size': 1,
                'streaming': True,
                'update_frequency': 'message'
            },
            'analysis': {
                'trust_threshold': 0.85,
                'bias_threshold': 0.15,
                'quality_threshold': 0.75
            },
            'monitoring': {
                'track_latency': True,
                'track_quality': True,
                'track_consistency': True
            }
        }
        
        model = await self._load_model()
        analyzer = EnhancedModelEvaluator(model, self.device, realtime_config)
        
        # Simulate real-time conversation
        async def conversation_simulator():
            scenarios = [
                {
                    'role': 'researcher',
                    'content': 'How does cognitive load theory apply to online learning environments?',
                    'context': {'domain': 'education', 'complexity': 0.7}
                },
                {
                    'role': 'assistant',
                    'content': 'According to recent studies by Smith et al. (2023)...',
                    'citations': ['Smith et al. (2023)']
                },
                {
                    'role': 'researcher',
                    'content': 'What methodology was used in these studies?',
                    'context': {'focus': 'methodology'}
                }
            ]
            
            for scenario in scenarios:
                yield scenario
                await asyncio.sleep(1)  # Simulate natural conversation flow
        
        print("\nReal-time Analysis:")
        async for message in conversation_simulator():
            # Real-time analysis with quality tracking
            analysis = await analyzer.analyze_streaming(
                message,
                track_quality=True
            )
            
            # Generate immediate recommendations
            recommendations = analyzer.generate_recommendations(analysis)
            
            # Print real-time insights
            self._print_realtime_analysis(message, analysis, recommendations)
    
    async def scenario_5_advanced_bias_analysis(self):
        """Scenario 5: Advanced bias detection and analysis"""
        bias_config = {
            'detection': {
                'pattern_types': {
                    'confirmation_bias': {
                        'indicators': [
                            'selective_citation',
                            'contrary_evidence_omission',
                            'preferred_interpretation'
                        ],
                        'threshold': 0.7
                    },
                    'methodology_bias': {
                        'indicators': [
                            'inappropriate_analysis',
                            'sample_bias',
                            'measurement_bias'
                        ],
                        'threshold': 0.6
                    },
                    'reporting_bias': {
                        'indicators': [
                            'positive_result_emphasis',
                            'negative_result_omission',
                            'selective_reporting'
                        ],
                        'threshold': 0.65
                    }
                },
                'analysis_depth': 'comprehensive',
                'tracking': {
                    'pattern_evolution': True,
                    'bias_interactions': True,
                    'context_sensitivity': True
                }
            },
            'validation': {
                'cross_validation': True,
                'minimum_confidence': 0.8,
                'evidence_requirement': 'strong'
            }
        }
        
        model = await self._load_model()
        evaluator = EnhancedModelEvaluator(model, self.device, {'bias': bias_config})
        
        # Analyze conversation for sophisticated bias patterns
        conversation = await self._load_sample_conversation()
        bias_analysis = await evaluator.analyze_bias_patterns(conversation)
        
        return self._generate_bias_report(bias_analysis)
    
    async def _save_advanced_dataset(self, conversations: List, metrics: List):
        """Save dataset with comprehensive metadata"""
        dataset_path = self.base_path / "advanced_dataset"
        dataset_path.mkdir(exist_ok=True)
        
        # Flatten and preprocess conversations
        def preprocess_conversation(conv):
            # Convert context values to strings
            if 'context' in conv:
                conv['context'] = {k: str(v) if hasattr(v, 'value') else v 
                                    for k, v in conv['context'].items()}
            return conv
        
        # Ensure conversations are properly formatted
        formatted_conversations = [
            preprocess_conversation(conv) 
            for conv in conversations 
            if isinstance(conv, dict)
        ]
        
        # Create a dictionary with conversations and metrics
        dataset = {
            'conversations': formatted_conversations,
            'metrics': metrics
        }
        
        # Save in multiple formats with metadata
        save_dataset(
            dataset,  # Pass the entire dataset dictionary
            output_path=dataset_path,
            formats=['csv', 'json']  # Remove parquet for now
        )
        
    async def _load_model(self) -> ResearchAnalysisModel:
        """Load pretrained model"""
        model_path = self.base_path / "models" / "research_model.pt"
        if not model_path.exists():
            raise FileNotFoundError("Model not found. Run training first.")
        return ResearchAnalysisModel.load(model_path)
    
    async def _generate_advanced_report(self, metrics: Dict) -> Dict:
        """Generate comprehensive analysis report"""
        report = {
            'trust_analysis': self._analyze_trust_metrics(metrics),
            'bias_analysis': self._analyze_bias_metrics(metrics),
            'quality_assessment': self._analyze_quality_metrics(metrics),
            'recommendations': self._generate_recommendations(metrics)
        }
        return report
    
    def _training_callback(self, stage: str, epoch: int, metrics: Dict):
        """Training progress callback"""
        print(f"\nStage: {stage}, Epoch: {epoch}")
        print("Metrics:", {k: f"{v:.4f}" for k, v in metrics.items()})
    
    def _print_realtime_analysis(self, message: Dict, analysis: Dict, recommendations: List):
        """Print real-time analysis results"""
        print(f"\nMessage: {message['content'][:100]}...")
        print(f"Trust Score: {analysis['trust_score']:.2f}")
        print(f"Quality Score: {analysis['quality_score']:.2f}")
        print("Recommendations:", recommendations)

async def main():
    """Run advanced usage scenarios"""
    scenarios = AdvancedResearchScenarios()
    
    print("\n1. Advanced Dataset Generation")
    await scenarios.scenario_1_advanced_generation()
    
    print("\n2. Curriculum Training")
    await scenarios.scenario_2_curriculum_training()
    
    print("\n3. Comprehensive Evaluation")
    await scenarios.scenario_3_comprehensive_evaluation()
    
    print("\n4. Real-time Processing")
    await scenarios.scenario_4_realtime_processing()
    
    print("\n5. Advanced Bias Analysis")
    await scenarios.scenario_5_advanced_bias_analysis()

if __name__ == "__main__":
    asyncio.run(main())