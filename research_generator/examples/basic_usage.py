"""
Basic usage examples for the Research Generator system
"""

import torch
from data_generation import UnifiedResearchGenerator
from models import ResearchAnalysisModel, TrainingPipeline
from models.evaluation import EnhancedModelEvaluator

def generate_basic_dataset():
    """Example 1: Basic dataset generation"""
    # Initialize generator with basic configuration
    config = {
        'size': 10,
        'min_length': 5,
        'max_length': 10,
        'edge_case_ratio': 0.2,
        'domains': ['education', 'psychology', 'stem']
    }
    
    generator = UnifiedResearchGenerator(config)
    
    # Generate dataset
    conversations, metrics = generator.generate_dataset()
    
    # Save the dataset
    generator.save_dataset(
        conversations,
        metrics,
        base_filename="basic_research_data"
    )
    
    print(f"Generated {len(conversations)} conversations")
    print("\nSample conversation:")
    print_sample_conversation(conversations[0])

def train_basic_model():
    """Example 2: Basic model training"""
    # Model configuration
    model_config = {
        'hidden_size': 768,
        'num_layers': 12,
        'num_heads': 12
    }
    
    # Training configuration
    training_config = {
        'batch_size': 32,
        'learning_rate': 2e-5,
        'epochs': 10,
        'validation_split': 0.2
    }
    
    # Initialize model and training pipeline
    model = ResearchAnalysisModel(model_config)
    pipeline = TrainingPipeline({
        'model': model_config,
        'training': training_config
    })
    
    # Generate training data
    generator = UnifiedResearchGenerator({'size': 1000})
    conversations, metrics = generator.generate_dataset()
    
    # Train model
    pipeline.train(conversations, metrics)
    
    print("\nModel training completed")
    print("Model saved to: research_model.pt")

def evaluate_basic_model():
    """Example 3: Basic model evaluation"""
    # Load model
    model = ResearchAnalysisModel.load("research_model.pt")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize evaluator
    evaluator = EnhancedModelEvaluator(model, device)
    
    # Generate test data
    generator = UnifiedResearchGenerator({'size': 100})
    test_conversations, test_metrics = generator.generate_dataset()
    
    # Evaluate model
    metrics = evaluator.evaluate(test_conversations)
    
    print("\nEvaluation Results:")
    print_evaluation_results(metrics)

def analyze_single_conversation():
    """Example 4: Single conversation analysis"""
    # Load trained model
    model = ResearchAnalysisModel.load("research_model.pt")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    evaluator = EnhancedModelEvaluator(model, device)
    
    # Create sample conversation
    conversation = {
        'messages': [
            {
                'role': 'researcher',
                'content': 'How does cognitive load affect online learning outcomes?',
            },
            {
                'role': 'assistant',
                'content': 'Studies by Smith et al. (2023) show that cognitive load significantly impacts...',
            }
        ],
        'metadata': {
            'domain': 'education',
            'topic': 'Cognitive Load in Online Learning'
        }
    }
    
    # Analyze conversation
    analysis = evaluator.evaluate_single_conversation(conversation)
    
    print("\nSingle Conversation Analysis:")
    print_conversation_analysis(analysis)

def real_time_analysis_example():
    """Example 5: Real-time analysis"""
    model = ResearchAnalysisModel.load("research_model.pt")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    evaluator = EnhancedModelEvaluator(model, device)
    
    # Simulate real-time conversation
    messages = [
        "Could you analyze recent research on student engagement in online learning?",
        "Based on Smith et al. (2023), student engagement is significantly affected by...",
        "What methodologies were used in these studies?",
        "The studies primarily employed mixed-methods approaches, combining surveys..."
    ]
    
    print("\nReal-time Analysis:")
    for msg in messages:
        # Analyze each message
        analysis = evaluator.analyze_message({
            'content': msg,
            'timestamp': torch.cuda.Event()
        })
        
        print(f"\nMessage: {msg[:50]}...")
        print("Trust Score:", analysis['trust_score'])
        print("Citation Status:", analysis['citation_verification'])
        print("Response Latency:", analysis['latency'])

def print_sample_conversation(conversation):
    """Utility function to print conversation"""
    print(f"\nConversation ID: {conversation['id']}")
    print(f"Domain: {conversation['context']['domain']}")
    print("\nMessages:")
    for msg in conversation['messages'][:4]:  # Print first 4 messages
        print(f"\n{msg['role'].upper()}: {msg['content'][:100]}...")

def print_evaluation_results(metrics):
    """Utility function to print evaluation results"""
    print("\nTrust Score:", metrics.realtime.trust_score)
    print("Bias Score:", metrics.bias.overall_bias_score)
    print("Research Quality:", metrics.research.overall_quality)
    print("\nDetailed Metrics:")
    print("- Methodology Rigor:", metrics.research.methodology_rigor)
    print("- Theoretical Alignment:", metrics.research.theoretical_alignment)
    print("- Evidence Strength:", metrics.research.empirical_evidence)

def print_conversation_analysis(analysis):
    """Utility function to print conversation analysis"""
    print("\nTrust Score:", analysis['trust_score'])
    print("Citation Verification:", analysis['citation_verification'])
    print("Methodology Consistency:", analysis['methodology_consistency'])
    print("\nBias Detection:")
    for bias_type, score in analysis['bias_scores'].items():
        print(f"- {bias_type}: {score:.2f}")

def main():
    """Run all basic examples"""
    print("Running Basic Usage Examples...")
    
    print("\n1. Generating Dataset")
    generate_basic_dataset()
    
    print("\n2. Training Model")
    train_basic_model()
    
    print("\n3. Evaluating Model")
    evaluate_basic_model()
    
    print("\n4. Single Conversation Analysis")
    analyze_single_conversation()
    
    print("\n5. Real-time Analysis")
    real_time_analysis_example()

if __name__ == "__main__":
    main()