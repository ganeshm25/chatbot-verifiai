# research_generator/data_generation/generator.py

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Tuple, Optional
import random
import json
import uuid
from datetime import datetime, timedelta
import pandas as pd

class ResearchDomain(Enum):
    EDUCATION = "education"
    PSYCHOLOGY = "psychology"
    SOCIOLOGY = "sociology"
    STEM = "stem"
    HUMANITIES = "humanities"

class ResearchPhase(Enum):
    LITERATURE_REVIEW = "literature_review"
    METHODOLOGY_DISCUSSION = "methodology_discussion"
    DATA_ANALYSIS = "data_analysis"
    FINDINGS_INTERPRETATION = "findings_interpretation"
    IMPLICATIONS_DISCUSSION = "implications_discussion"

@dataclass
class EdgeCase:
    type: str
    probability: float
    patterns: List[str]
    triggers: List[str]

@dataclass
class ResearchContext:
    domain: ResearchDomain
    topic: str
    methodology: str
    theoretical_framework: str
    research_questions: List[str]
    phase: ResearchPhase
    complexity: float
    uncertainty: float

class UnifiedResearchGenerator:
    """Enhanced research conversation generator with comprehensive features"""
    
    def __init__(self, config: Dict):
        self.config = self._merge_with_defaults(config)
        self.initialize_components()
        
    def _merge_with_defaults(self, config: Dict) -> Dict:
        """Merge provided config with defaults"""
        defaults = {
            'size': 20,
            'min_length': 5,
            'max_length': 20,
            'edge_case_ratio': 0.2,
            'domains': ['all'],
            'complexity_levels': ['basic', 'medium', 'complex']
        }
        return {**defaults, **config}
    
    def initialize_components(self):
        """Initialize all generator components"""
        self.topics = self._load_research_topics()
        self.methodologies = self._load_methodologies()
        self.frameworks = self._load_theoretical_frameworks()
        self.edge_cases = self._initialize_edge_cases()
        self.patterns = self._initialize_patterns()
        self.metrics = self._initialize_metrics()
        
    def _load_research_topics(self) -> Dict[str, List[str]]:
        """Load research topics by domain"""
        return {
            "education": [
                "Cognitive Load in Online Learning",
                "Social-Emotional Learning Impact",
                "Digital Literacy Development",
                "Inclusive Education Practices",
                "Assessment Methods Innovation"
            ],
            "psychology": [
                "Behavioral Intervention Efficacy",
                "Cognitive Development Patterns",
                "Mental Health Interventions",
                "Social Psychology Dynamics",
                "Learning Psychology"
            ],
            "stem": [
                "Machine Learning Ethics",
                "Quantum Computing Applications",
                "Renewable Energy Systems",
                "Biotechnology Advances",
                "Space Exploration Methods"
            ]
        }
    
    def _initialize_edge_cases(self) -> Dict[str, EdgeCase]:
        """Initialize edge case definitions"""
        return {
            "contradictory_claims": EdgeCase(
                type="contradiction",
                probability=0.15,
                patterns=["claim_a -> opposite_claim"],
                triggers=["however", "contrary", "despite"]
            ),
            "methodology_mismatch": EdgeCase(
                type="method_mismatch",
                probability=0.10,
                patterns=["qual_method_quant_conclusion"],
                triggers=["significant results", "statistical evidence"]
            ),
            "theoretical_inconsistency": EdgeCase(
                type="theory_conflict",
                probability=0.08,
                patterns=["framework_a_concept_b"],
                triggers=["based on theory", "theoretical framework"]
            )
        }
    
    def generate_dataset(self) -> Tuple[List, List]:
        """Generate complete research conversation dataset"""
        conversations = []
        metrics = []
        
        for i in range(self.config['size']):
            # Generate research context
            context = self._generate_research_context()
            
            # Generate conversation
            conversation = self._generate_conversation(context)
            conversations.append(conversation)
            
            # Generate metrics
            metric = self._generate_metrics(conversation, context)
            metrics.append(metric)
        
        return conversations, metrics
    
    def _generate_research_context(self) -> ResearchContext:
        """Generate complete research context"""
        domain = random.choice(list(ResearchDomain))
        topic = random.choice(self.topics[domain.value])
        
        return ResearchContext(
            domain=domain,
            topic=topic,
            methodology=random.choice(self.methodologies),
            theoretical_framework=random.choice(self.frameworks),
            research_questions=self._generate_research_questions(topic),
            phase=random.choice(list(ResearchPhase)),
            complexity=random.uniform(0.1, 1.0),
            uncertainty=random.uniform(0.1, 1.0)
        )
    
    def _generate_conversation(self, context: ResearchContext) -> Dict:
        """Generate single research conversation"""
        conversation = {
            "id": str(uuid.uuid4()),
            "timestamp": datetime.now().isoformat(),
            "context": self._context_to_dict(context),
            "messages": []
        }
        
        # Generate base messages
        num_exchanges = random.randint(
            self.config['min_length'],
            self.config['max_length']
        )
        
        messages = []
        for i in range(num_exchanges):
            # Researcher message
            researcher_msg = self._generate_researcher_message(context, i)
            messages.append(researcher_msg)
            
            # Assistant response
            assistant_msg = self._generate_assistant_response(context, researcher_msg)
            messages.append(assistant_msg)
        
        # Inject edge cases if needed
        if random.random() < self.config['edge_case_ratio']:
            messages = self._inject_edge_cases(messages)
        
        conversation["messages"] = messages
        return conversation
    
    def _generate_metrics(self, conversation: Dict, context: ResearchContext) -> Dict:
        """Generate comprehensive metrics for conversation"""
        return {
            "conversation_id": conversation["id"],
            "methodology_metrics": self._calculate_methodology_metrics(conversation),
            "theoretical_metrics": self._calculate_theoretical_metrics(conversation),
            "authenticity_metrics": self._calculate_authenticity_metrics(conversation),
            "edge_case_metrics": self._calculate_edge_case_metrics(conversation)
        }
    
    def save_dataset(self, conversations: List, metrics: List, base_filename: str):
        """Save generated dataset to files"""
        # Save conversations
        conv_df = pd.DataFrame(self._flatten_conversations(conversations))
        conv_df.to_csv(f"{base_filename}_conversations.csv", index=False)
        
        # Save metrics
        metrics_df = pd.DataFrame(self._flatten_metrics(metrics))
        metrics_df.to_csv(f"{base_filename}_metrics.csv", index=False)
        
        # Save complete data as JSON
        with open(f"{base_filename}_complete.json", 'w') as f:
            json.dump({
                "conversations": conversations,
                "metrics": metrics
            }, f, indent=2, default=str)
    
    def _flatten_conversations(self, conversations: List) -> List[Dict]:
        """Flatten conversation structure for DataFrame"""
        flattened = []
        for conv in conversations:
            for msg in conv["messages"]:
                flat_msg = {
                    "conversation_id": conv["id"],
                    "timestamp": msg["timestamp"],
                    **msg,
                    **conv["context"]
                }
                flattened.append(flat_msg)
        return flattened
    
    def _flatten_metrics(self, metrics: List) -> List[Dict]:
        """Flatten metrics structure for DataFrame"""
        flattened = []
        for metric in metrics:
            flat_metric = {
                "conversation_id": metric["conversation_id"],
                **metric["methodology_metrics"],
                **metric["theoretical_metrics"],
                **metric["authenticity_metrics"],
                **metric["edge_case_metrics"]
            }
            flattened.append(flat_metric)
        return flattened

def main():
    # Example usage
    config = {
        'size': 100,
        'edge_case_ratio': 0.2,
        'domains': ['education', 'psychology']
    }
    
    generator = UnifiedResearchGenerator(config)
    conversations, metrics = generator.generate_dataset()
    generator.save_dataset(conversations, metrics, "research_data")
    
    # Print sample
    print("\nSample Conversation:")
    sample_conv = conversations[0]
    for msg in sample_conv["messages"][:4]:
        print(f"\n{msg['role'].upper()}: {msg['content']}")
        print(f"Context: {sample_conv['context']['topic']} - {sample_conv['context']['phase']}")
    
    print("\nSample Metrics:")
    print(json.dumps(metrics[0], indent=2))

if __name__ == "__main__":
    main()