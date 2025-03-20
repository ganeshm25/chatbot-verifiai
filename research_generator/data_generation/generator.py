"""
Enhanced unified generator for research conversations with sophisticated patterns
"""

import asyncio
from typing import Dict, List, Tuple, Optional, Union
import random
from datetime import datetime, timedelta
import uuid
import json
import re
from dataclasses import dataclass, asdict
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime, timedelta
import uuid
from dataclasses import asdict

from .ai_logger import AIInteractionLogger
from .c2pa_manager import C2PAManager
from .models import (
    ResearchContext,
    AIInteraction,
    AIInteractionType,
    ContentProvenance,
    ConversationPhase,
    ConversationStyle,
    VerificationStatus
)

from .patterns import PatternManager
from .edge_cases import EdgeCaseManager
from .metrics import MetricsCalculator

from .patterns import (
    PatternManager,
    ConversationPhase,
    ConversationStyle,
    ResearchTemplate
)
from .edge_cases import EdgeCaseManager, EdgeCaseType
from .metrics import MetricsCalculator
from research_generator.utils.dataset_diversity import (
    classify_content_authenticity, 
    calculate_trust_score, 
    validate_dataset_diversity,
    get_verification_status
)

from enum import Enum

# @dataclass
# class ResearchContext:
#     """Enhanced research context"""
#     domain: str
#     topic: str
#     methodology: str
#     theoretical_framework: str
#     complexity: float
#     phase: ConversationPhase
#     style: ConversationStyle
#     research_questions: List[str]
#     citations: List[Dict[str, str]]
#     variables: Dict[str, Union[str, float]]

class UnifiedResearchGeneratorA:
    """Enhanced research conversation generator"""
    
    def __init__(self, config: Dict):
        self.config = self._merge_with_defaults(config)
        self.pattern_manager = PatternManager()
        self.edge_case_manager = EdgeCaseManager()
        self.metrics_calculator = MetricsCalculator()
        self._initialize_components()
    
    def _merge_with_defaults(self, config: Dict) -> Dict:
        """Merge provided config with defaults"""
        defaults = {
            'size': 50,
            'min_length': 5,
            'max_length': 20,
            'edge_case_ratio': 0.2,
            'domains': ['education', 'psychology', 'stem'],
            'complexity_levels': ['basic', 'medium', 'complex'],
            'template_settings': {
                'use_dynamic_templates': True,
                'allow_nested_templates': True,
                'context_sensitivity': 0.8
            },
            'edge_case_settings': {
                'severity_threshold': 0.7,
                'max_per_conversation': 2
            },
            'quality_settings': {
                'min_citations': 2,
                'required_methodology': True,
                'theoretical_framework': True
            }
        }
        return {**defaults, **config}
    
    def _initialize_components(self):
        """Initialize generation components"""
        self.domains = {
            'education': {
                'topics': [
                    "Cognitive Load in Online Learning",
                    "Social-Emotional Learning Impact",
                    "Digital Literacy Development",
                    "Inclusive Education Practices",
                    "Assessment Methods Innovation"
                ],
                'methodologies': [
                    "Mixed Methods Research",
                    "Action Research",
                    "Case Study Analysis",
                    "Longitudinal Study",
                    "Experimental Design"
                ],
                'frameworks': [
                    "Constructivist Learning Theory",
                    "Social Cognitive Theory",
                    "Transformative Learning Theory",
                    "Communities of Practice",
                    "Experiential Learning Model"
                ]
            },
            'psychology': {
                'topics': [
                    "Behavioral Intervention Efficacy",
                    "Cognitive Development Patterns",
                    "Mental Health Interventions",
                    "Social Psychology Dynamics",
                    "Neuropsychological Assessment"
                ],
                'methodologies': [
                    "Experimental Psychology",
                    "Clinical Trials",
                    "Cross-sectional Studies",
                    "Longitudinal Research",
                    "Meta-analysis"
                ],
                'frameworks': [
                    "Cognitive Behavioral Theory",
                    "Psychodynamic Framework",
                    "Social Learning Theory",
                    "Humanistic Psychology",
                    "Neuropsychological Theory"
                ]
            },
            'stem': {
                'topics': [
                    "Machine Learning Ethics",
                    "Quantum Computing Applications",
                    "Renewable Energy Systems",
                    "Biotechnology Advances",
                    "Data Science Methods"
                ],
                'methodologies': [
                    "Empirical Analysis",
                    "Computational Modeling",
                    "Laboratory Experimentation",
                    "Statistical Analysis",
                    "Simulation Studies"
                ],
                'frameworks': [
                    "Systems Theory",
                    "Information Processing",
                    "Computational Theory",
                    "Scientific Method",
                    "Engineering Design"
                ]
            }
        }
        
        # Initialize citation database
        self.citations = self._initialize_citations()
    
    def _initialize_citations(self) -> Dict[str, List[Dict]]:
        """Initialize domain-specific citation database"""
        return {
            domain: [
                {
                    'author': f"{last_name} et al.",
                    'year': str(random.randint(2018, 2024)),
                    'title': f"Research on {topic.lower()}",
                    'journal': "Journal of Research",
                    'doi': f"10.1000/jr.{random.randint(1000, 9999)}"
                }
                for last_name in ["Smith", "Johnson", "Williams", "Brown", "Jones"]
                for topic in self.domains[domain]['topics']
            ]
            for domain in self.domains
        }
    
    async def generate_dataset(self) -> Tuple[List[Dict], List[Dict]]:
        """Generate research conversation dataset with advanced patterns"""
        conversations = []
        metrics = []
        
        for i in range(self.config['size']):
            # Generate research context
            context = await self._generate_research_context()
            
            # Generate conversation
            conversation = await self._generate_conversation(context)
            
            # Inject edge cases if appropriate
            if random.random() < self.config['edge_case_ratio']:
                conversation = self.edge_case_manager.inject_edge_cases(
                    conversation,
                    context.phase,
                    context.style,
                    asdict(context)
                )
            
            # Calculate metrics
            metric = await self._calculate_metrics(conversation, context)
            
            # Only add conversations with valid metrics
            if conversation and metric:
                conversations.append(conversation)
                metrics.append(metric)
            
            if (i + 1) % 100 == 0:
                print(f"Generated {i + 1}/{self.config['size']} conversations")
        
        return conversations, metrics
    
    async def _generate_research_context(self) -> ResearchContext:
        """Generate enhanced research context"""
        domain = random.choice(self.config['domains'])
        domain_content = self.domains[domain]
        
        # Select topic and related content
        topic = random.choice(domain_content['topics'])
        methodology = random.choice(domain_content['methodologies'])
        framework = random.choice(domain_content['frameworks'])
        
        # Generate research questions
        research_questions = await self._generate_research_questions(topic)
        
        # Select relevant citations
        citations = random.sample(self.citations[domain], 
                                k=random.randint(2, 5))
        
        # Generate research variables
        variables = await self._generate_research_variables(topic)
        
        return ResearchContext(
            domain=domain,
            topic=topic,
            methodology=methodology,
            theoretical_framework=framework,
            complexity=random.uniform(0.3, 1.0),
            phase=random.choice(list(ConversationPhase)),
            style=random.choice(list(ConversationStyle)),
            research_questions=research_questions,
            citations=citations,
            variables=variables
        )
    
    async def _generate_conversation(self, context: ResearchContext) -> Dict:
        """Generate enhanced research conversation"""
        conversation = {
            "id": str(uuid.uuid4()),
            "timestamp": datetime.now().isoformat(),
            "context": asdict(context),
            "messages": []
        }
        
        # Generate conversation flow
        num_exchanges = random.randint(
            self.config['min_length'],
            self.config['max_length']
        )
        
        current_phase = context.phase
        messages = []
        
        for i in range(num_exchanges):
            # Generate researcher message
            researcher_msg = await self._generate_researcher_message(
                context,
                current_phase,
                i
            )
            messages.append(researcher_msg)
            
            # Generate assistant response
            assistant_msg = await self._generate_assistant_response(
                context,
                researcher_msg,
                current_phase
            )
            messages.append(assistant_msg)
            
            # Update phase if needed
            if random.random() < 0.3:  # 30% chance to change phase
                current_phase = self._get_next_phase(current_phase)
        
        conversation["messages"] = messages
        return conversation
    
    async def _generate_researcher_message(
        self,
        context: ResearchContext,
        phase: ConversationPhase,
        position: int
    ) -> Dict:
        """Generate enhanced researcher message"""
        try:
            # Fallback template generation
            message_content = self.pattern_manager.generate_conversation_flow(
                phase,
                context.style,
                context.complexity,
                asdict(context)
            )
            
            # Ensure non-empty content
            if not message_content:
                message_content = [f"Default researcher message about {context.topic}"]
            
            return {
                "id": str(uuid.uuid4()),
                "timestamp": datetime.now() + timedelta(minutes=position*5),
                "role": "researcher",
                "content": " ".join(message_content),
                "metadata": {
                    "phase": phase.value,
                    "style": context.style.value,
                    "position": position,
                    "context_variables": list(context.variables.keys())
                }
            }
        except Exception as e:
            return {
                "id": str(uuid.uuid4()),
                "timestamp": datetime.now() + timedelta(minutes=position*5),
                "role": "researcher",
                "content": f"Fallback message: {context.topic} research exploration",
                "metadata": {}
            }
    
    async def _generate_assistant_response(
        self,
        context: ResearchContext,
        researcher_msg: Dict,
        phase: ConversationPhase
    ) -> Dict:
        try:
            # Generate response using templates
            response_content = self.pattern_manager.generate_conversation_flow(
                phase,
                context.style,
                context.complexity,
                {
                    **asdict(context),
                    "previous_message": researcher_msg["content"]
                }
            )
            
            # Ensure non-empty content
            if not response_content:
                response_content = [f"Default assistant response about {context.topic}"]
            
            # Add citations if appropriate
            if self.config['quality_settings']['min_citations']:
                response_content = await self._add_citations(
                    response_content,
                    context.citations
                )
            
            return {
                "id": str(uuid.uuid4()),
                "timestamp": datetime.now() + timedelta(
                    minutes=researcher_msg["metadata"]["position"]*5 + 2
                ),
                "role": "assistant",
                "content": " ".join(response_content),
                "metadata": {
                    "phase": phase.value,
                    "style": context.style.value,
                    "citations_used": [c["doi"] for c in context.citations],
                    "theoretical_framework": context.theoretical_framework
                }
            }
        except Exception as e:
            return {
                "id": str(uuid.uuid4()),
                "timestamp": datetime.now() + timedelta(
                    minutes=researcher_msg["metadata"]["position"]*5 + 2
                ),
                "role": "assistant",
                "content": f"Fallback assistant response: {context.topic} insights",
                "metadata": {}
            }
    
    def _get_next_phase(self, current_phase: ConversationPhase) -> ConversationPhase:
        """Get next logical conversation phase"""
        phase_order = {
            ConversationPhase.INTRODUCTION: ConversationPhase.LITERATURE_REVIEW,
            ConversationPhase.LITERATURE_REVIEW: ConversationPhase.METHODOLOGY,
            ConversationPhase.METHODOLOGY: ConversationPhase.ANALYSIS,
            ConversationPhase.ANALYSIS: ConversationPhase.FINDINGS,
            ConversationPhase.FINDINGS: ConversationPhase.DISCUSSION,
            ConversationPhase.DISCUSSION: ConversationPhase.CONCLUSION,
            ConversationPhase.CONCLUSION: ConversationPhase.CONCLUSION
        }
        return phase_order.get(current_phase, current_phase)
    
    async def _add_citations(
        self,
        content: List[str],
        citations: List[Dict]
    ) -> List[str]:
        """Add relevant citations to content"""
        if not citations:
            return content
        
        cited_content = []
        for text in content:
            if random.random() < 0.3:  # 30% chance to add citation
                citation = random.choice(citations)
                text += f" ({citation['author']}, {citation['year']})"
            cited_content.append(text)
        
        return cited_content
    
    async def _generate_research_questions(self, topic: str) -> List[str]:
        """Generate research questions"""
        templates = [
            f"How does {topic.lower()} impact research outcomes?",
            f"What are the key factors influencing {topic.lower()}?",
            f"How can {topic.lower()} be effectively implemented?",
            f"What role does technology play in {topic.lower()}?",
            f"How do different contexts affect {topic.lower()}?"
        ]
        return random.sample(templates, k=random.randint(2, 4))
    
    async def _generate_research_variables(self, topic: str) -> Dict[str, Union[str, float]]:
        """Generate research variables"""
        return {
            'dependent_var': f"{topic.lower()}_outcome",
            'independent_var': f"{topic.lower()}_intervention",
            'control_var': f"{topic.lower()}_baseline",
            'effect_size': round(random.uniform(0.1, 0.8), 2),
            'sample_size': random.randint(50, 500)
        }
    
    async def _calculate_metrics(
        self,
        conversation: Dict,
        context: ResearchContext
    ) -> Dict:
        """Calculate comprehensive metrics"""
        try:

                # Handle string input
                if isinstance(conversation, str):
                    print(f"Invalid conversation type(not str): {type(conversation)}")
                    return {}
        
                # Handle list input
                if isinstance(conversation, list):
                    # Ensure list is not empty and contains dictionaries
                    conversation = [
                        conv for conv in conversation 
                        if isinstance(conv, dict)
                    ]
                    conversation = conversation[0] if conversation else {}
                
                # Ensure conversation is a dictionary
                if not isinstance(conversation, dict):
                    print(f"Invalid conversation type(not dict): {type(conversation)}")
                    return {}
                
                # Validate context
                if not isinstance(context, ResearchContext):
                    context = ResearchContext(
                        domain='unknown',
                        topic='unknown',
                        methodology='unknown',
                        theoretical_framework='unknown',
                        complexity=0.5,
                        phase=random.choice(list(ConversationPhase)),
                        style=random.choice(list(ConversationStyle)),
                        research_questions=[],
                        citations=[],
                        variables={}
                    )
                
                # Calculate metrics safely
                try:
                    metrics = self.metrics_calculator.calculate_metrics(conversation, asdict(context))
                    return metrics
                except Exception as metric_error:
                    print(f"Metrics calculation error: {metric_error}")
                    return {}
            
        except Exception as e:
            print(f"Unexpected error in _calculate_metrics: {e}")
            return {}
#        return self.metrics_calculator.calculate_metrics(conversation, asdict(context))
    
    # def save_dataset(
    #     self,
    #     conversations: List[Dict],
    #     metrics: List[Dict],
    #     base_filename: str
    # ) -> None:
    #     """Save generated dataset"""
    #     # Save conversations
    #     conv_df = pd.DataFrame(self._flatten_conversations(conversations))
    #     conv_df.to_csv(f"{base_filename}_conversations.csv", index=False)
        
    #     # Save metrics
    #     metrics_df = pd.DataFrame(metrics)
    #     metrics_df.to_csv(f"{base_filename}_metrics.csv", index=False)
        
    #     # Save complete data
    #     with open(f"{base_filename}_complete.json", 'w') as f:
    #         json.dump({
    #             "conversations": conversations,
    #             "metrics": metrics,
    #             "config": self.config
    #         }, f, indent=2, default=str)

    def save_dataset(
        self,
        conversations: List[Union[Dict, List]],
        metrics: List[Dict],
        base_filename: str
    ) -> None:
        """Save generated dataset with robust error handling"""
        try:
            # Normalize conversations
            if not conversations:
                print("No conversations to save.")
                return
            
            # Flatten nested lists if needed
            if isinstance(conversations[0], list):
                conversations = [
                    conv for sublist in conversations 
                    for conv in (sublist if isinstance(sublist, list) else [sublist])
                ]
            
            # Ensure we have a list of dictionaries
            conversations = [
                conv for conv in conversations 
                if isinstance(conv, dict)
            ]
            
            # Flatten conversations
            flattened_convs = self._flatten_conversations(conversations)
            
            # Save conversations
            conv_df = pd.DataFrame(flattened_convs)
            conv_df.to_csv(f"{base_filename}_conversations.csv", index=False)
            
            # Save metrics
            metrics_df = pd.DataFrame(metrics)
            metrics_df.to_csv(f"{base_filename}_metrics.csv", index=False)
            
            # Save complete data
            with open(f"{base_filename}_complete.json", 'w') as f:
                json.dump({
                    "conversations": conversations,
                    "metrics": metrics,
                    "config": self.config
                }, f, indent=2, default=str)
            
            print(f"Dataset saved successfully. Conversations: {len(conversations)}")
        
        except Exception as e:
            print(f"Error saving dataset: {e}")

    def _flatten_conversations(self, conversations: List[Dict]) -> List[Dict]:
        """Flatten conversations for DataFrame format"""
        flattened = []
        for conv in conversations:
            context = conv["context"]
            for msg in conv["messages"]:
                flat_msg = {
                    "conversation_id": conv["id"],
                    "timestamp": msg["timestamp"],
                    **msg,
                    **{f"context_{k}": v for k, v in context.items()
                       if not isinstance(v, (list, dict))}
                }
                flattened.append(flat_msg)
        return flattened
    
    async def _generate_templated_content(
        self,
        context: ResearchContext,
        phase: ConversationPhase,
        complexity: float,
        previous_content: Optional[str] = None
    ) -> List[str]:
        """Generate content using advanced templating system"""
        templates = {
            'literature_review': {
                'synthesis': [
                    "A systematic review by {author} ({year}) analyzed {num_studies} studies on {topic}, "
                    "revealing {key_finding}. This {methodology} approach demonstrated {evidence}.",
                    
                    "Integrating findings from {author1} ({year1}) and {author2} ({year2}), "
                    "we observe {pattern} in {research_area}. This suggests {implication}."
                ],
                'critique': [
                    "While {author} ({year}) proposed {theory}, subsequent research by "
                    "{critic} ({critic_year}) identified {limitation}. This raises questions about {aspect}.",
                    
                    "Critical analysis of {methodology} studies reveals {gap}, particularly "
                    "when considering {factor} as highlighted by {author} ({year})."
                ]
            },
            'methodology': {
                'design': [
                    "The {study_type} employed a {design_type} design with {sample_size} participants, "
                    "utilizing {instrument} for data collection. {analysis_method} was used to analyze {data_type}.",
                    
                    "To address {research_question}, researchers developed {approach} incorporating "
                    "{feature}. This enabled {advantage} while mitigating {limitation}."
                ],
                'implementation': [
                    "Data collection occurred over {duration}, involving {procedure}. "
                    "Participants were {sampling_method} based on {criteria}.",
                    
                    "The {intervention} was implemented using {protocol}, with {control_measure} "
                    "to ensure {quality_aspect}."
                ]
            }
        }

        # Select appropriate templates based on phase and complexity
        phase_templates = templates.get(phase.value, {})
        style_templates = []
        
        for style, temp_list in phase_templates.items():
            if complexity >= 0.7 and style in ['critique', 'implementation']:
                style_templates.extend(temp_list)
            elif complexity >= 0.4 and style in ['synthesis', 'design']:
                style_templates.extend(temp_list)
        
        if not style_templates:
            return [self._generate_fallback_content(context, phase)]
        
        # Fill templates with context
        content = []
        for template in random.sample(style_templates, k=min(2, len(style_templates))):
            filled_content = await self._fill_advanced_template(template, context)
            content.append(filled_content)
        
        return content

    async def _fill_advanced_template(
        self,
        template: str,
        context: ResearchContext
    ) -> str:
        """Fill template with advanced contextual awareness"""
        template_vars = {
            'author': lambda: random.choice([c['author'] for c in context.citations]),
            'year': lambda: random.choice([c['year'] for c in context.citations]),
            'num_studies': lambda: str(random.randint(10, 50)),
            'topic': lambda: context.topic,
            'methodology': lambda: context.methodology,
            'research_area': lambda: f"{context.domain} research",
            'theory': lambda: context.theoretical_framework,
            'sample_size': lambda: str(context.variables.get('sample_size', random.randint(50, 500))),
            'study_type': lambda: random.choice([
                "investigation", "research", "study", "analysis", "examination"
            ]),
            'design_type': lambda: random.choice([
                "randomized controlled", "quasi-experimental", "longitudinal",
                "cross-sectional", "mixed-methods"
            ]),
            'analysis_method': lambda: random.choice([
                "thematic analysis", "statistical analysis", "content analysis",
                "regression analysis", "phenomenological analysis"
            ]),
            'research_question': lambda: random.choice(context.research_questions)
        }

        try:
            # Extract needed variables from template
            needed_vars = {
                var.strip('{}') for var in re.findall(r'\{([^}]+)\}', template)
            }

            # Generate values for each variable
            values = {}
            for var in needed_vars:
                if var in template_vars:
                    values[var] = template_vars[var]()
                else:
                    values[var] = self._generate_variable_value(var, context)

            return template.format(**values)
        except KeyError as e:
            return f"Error filling template: missing value for {e}"

    def _generate_variable_value(
        self,
        variable: str,
        context: ResearchContext
    ) -> str:
        """Generate contextually appropriate values for template variables"""
        
        # Research-specific value generators
        research_values = {
            'evidence': lambda: random.choice([
                "significant correlations", "strong associations",
                "causal relationships", "meaningful patterns"
            ]),
            'implication': lambda: random.choice([
                "important implications for practice",
                "significant theoretical contributions",
                "methodological considerations",
                "potential applications"
            ]),
            'limitation': lambda: random.choice([
                "methodological constraints",
                "sampling limitations",
                "theoretical gaps",
                "contextual factors"
            ]),
            'key_finding': lambda: f"a {random.choice(['significant', 'notable', 'important'])} "
                                 f"relationship between {context.variables.get('independent_var', 'variables')}"
        }

        # Quality-related value generators
        quality_values = {
            'quality_aspect': lambda: random.choice([
                "methodological rigor", "data quality",
                "analytical precision", "reliability"
            ]),
            'control_measure': lambda: random.choice([
                "validation procedures", "quality controls",
                "verification methods", "reliability checks"
            ])
        }

        # Method-specific value generators
        method_values = {
            'procedure': lambda: random.choice([
                "structured interviews", "standardized assessments",
                "systematic observations", "controlled experiments"
            ]),
            'sampling_method': lambda: random.choice([
                "randomly selected", "purposively sampled",
                "systematically recruited", "strategically chosen"
            ]),
            'protocol': lambda: random.choice([
                "standardized protocol", "validated procedure",
                "systematic approach", "structured methodology"
            ])
        }

        # Check all value generators
        all_generators = {
            **research_values,
            **quality_values,
            **method_values
        }

        if variable in all_generators:
            return all_generators[variable]()
        
        # Generate a generic contextual value if no specific generator exists
        return f"{variable.replace('_', ' ')} in {context.domain} research"

    async def generate_template_variations(
        self,
        base_template: str,
        context: ResearchContext,
        num_variations: int = 3
    ) -> List[str]:
        """Generate variations of a base template"""
        variations = []
        
        # Extract template structure
        structure = re.findall(r'\{([^}]+)\}', base_template)
        
        for _ in range(num_variations):
            variation = base_template
            for var in structure:
                # Generate new value for each variable
                new_value = await self._get_variation_value(var, context)
                variation = variation.replace(f"{{{var}}}", new_value)
            variations.append(variation)
        
        return variations

    async def _get_variation_value(
        self,
        variable: str,
        context: ResearchContext
    ) -> str:
        """Get variation value for template variable"""
        variation_generators = {
            'author': lambda: random.choice([
                f"{author['author']} and colleagues",
                f"the team led by {author['author']}",
                f"researchers {author['author']}"
            ] for author in context.citations),
            'methodology': lambda: random.choice([
                f"using {context.methodology}",
                f"employing {context.methodology}",
                f"through {context.methodology}"
            ]),
            'finding': lambda: random.choice([
                f"discovered that {context.variables.get('dependent_var')}",
                f"demonstrated that {context.variables.get('dependent_var')}",
                f"revealed that {context.variables.get('dependent_var')}"
            ])
        }

        if variable in variation_generators:
            return variation_generators[variable]()
        return self._generate_variable_value(variable, context)

    def _generate_fallback_content(
        self,
        context: ResearchContext,
        phase: ConversationPhase
    ) -> str:
        """Generate fallback content when templates are not available"""
        fallback_templates = {
            ConversationPhase.LITERATURE_REVIEW: 
                "Research in {domain} has explored {topic}, with studies showing {finding}.",
            ConversationPhase.METHODOLOGY:
                "The study employed {methodology} to investigate {topic}.",
            ConversationPhase.ANALYSIS:
                "Analysis of {topic} revealed {finding} in the context of {domain}.",
            ConversationPhase.DISCUSSION:
                "These findings suggest important implications for {topic} in {domain}."
        }

        template = fallback_templates.get(
            phase,
            "Research on {topic} in {domain} continues to evolve."
        )

        return template.format(
            domain=context.domain,
            topic=context.topic,
            methodology=context.methodology,
            finding=f"significant relationships with {context.variables.get('dependent_var', 'outcomes')}"
        )

# Example usage in main generator class:
    async def _generate_researcher_message(
        self,
        context: ResearchContext,
        phase: ConversationPhase,
        position: int
    ) -> Dict:
        """Generate enhanced researcher message with templates"""
        # Generate templated content
        content = await self._generate_templated_content(
            context,
            phase,
            context.complexity
        )
        
        return {
            "id": str(uuid.uuid4()),
            "timestamp": datetime.now() + timedelta(minutes=position*5),
            "role": "researcher",
            "content": " ".join(content),
            "metadata": {
                "phase": phase.value,
                "style": context.style.value,
                "position": position,
                "template_variables": list(context.variables.keys()),
                "complexity": context.complexity
            }
        }
class UnifiedResearchGenerator:
    """Enhanced research conversation generator with AI and C2PA support"""
    
    def __init__(self, config: Dict):
        self.logger = logging.getLogger(__name__)
        self.config = self._merge_with_defaults(config)
        
        # Initialize core components
        self.pattern_manager = PatternManager()
        self.edge_case_manager = EdgeCaseManager()
        self.metrics_calculator = MetricsCalculator()
        
        # Initialize new components
        self.ai_logger = AIInteractionLogger()
        self.c2pa_manager = C2PAManager()
        
        self._initialize_components()
    
    def _merge_with_defaults(self, config: Dict) -> Dict:
        """Merge provided config with defaults including AI and C2PA settings"""
        defaults = {
            'size': 50,
            'min_length': 5,
            'max_length': 20,
            'edge_case_ratio': 0.2,
            'domains': ['education', 'psychology', 'stem'],
            'complexity_levels': ['basic', 'medium', 'complex'],
            'ai_settings': {
                'models': {
                    'GPT-4': {
                        'version': '1.0',
                        'provider': 'OpenAI',
                        'capabilities': ['text_generation', 'analysis']
                    }
                },
                'interaction_logging': True,
                'verification_required': True
            },
            'c2pa_settings': {
                'provenance_tracking': True,
                'verification_level': 'standard',
                'manifest_generation': True
            }
        }
        return {**defaults, **config}

    def _initialize_components(self):
        """Initialize generation components"""
        self.domains = {
            'education': {
                'topics': [
                    "Cognitive Load in Online Learning",
                    "Social-Emotional Learning Impact",
                    "Digital Literacy Development",
                    "Inclusive Education Practices",
                    "Assessment Methods Innovation"
                ],
                'methodologies': [
                    "Mixed Methods Research",
                    "Action Research",
                    "Case Study Analysis",
                    "Longitudinal Study",
                    "Experimental Design"
                ],
                'frameworks': [
                    "Constructivist Learning Theory",
                    "Social Cognitive Theory",
                    "Transformative Learning Theory",
                    "Communities of Practice",
                    "Experiential Learning Model"
                ]
            },
            'psychology': {
                'topics': [
                    "Behavioral Intervention Efficacy",
                    "Cognitive Development Patterns",
                    "Mental Health Interventions",
                    "Social Psychology Dynamics",
                    "Neuropsychological Assessment"
                ],
                'methodologies': [
                    "Experimental Psychology",
                    "Clinical Trials",
                    "Cross-sectional Studies",
                    "Longitudinal Research",
                    "Meta-analysis"
                ],
                'frameworks': [
                    "Cognitive Behavioral Theory",
                    "Psychodynamic Framework",
                    "Social Learning Theory",
                    "Humanistic Psychology",
                    "Neuropsychological Theory"
                ]
            },
            'stem': {
                'topics': [
                    "Machine Learning Ethics",
                    "Quantum Computing Applications",
                    "Renewable Energy Systems",
                    "Biotechnology Advances",
                    "Data Science Methods"
                ],
                'methodologies': [
                    "Empirical Analysis",
                    "Computational Modeling",
                    "Laboratory Experimentation",
                    "Statistical Analysis",
                    "Simulation Studies"
                ],
                'frameworks': [
                    "Systems Theory",
                    "Information Processing",
                    "Computational Theory",
                    "Scientific Method",
                    "Engineering Design"
                ]
            }
        }
        
        # Initialize citation database
        self.citations = self._initialize_citations()
    
    def _initialize_citations(self) -> Dict[str, List[Dict]]:
        """Initialize domain-specific citation database"""
        return {
            domain: [
                {
                    'author': f"{last_name} et al.",
                    'year': str(random.randint(2018, 2024)),
                    'title': f"Research on {topic.lower()}",
                    'journal': "Journal of Research",
                    'doi': f"10.1000/jr.{random.randint(1000, 9999)}"
                }
                for last_name in ["Smith", "Johnson", "Williams", "Brown", "Jones"]
                for topic in self.domains[domain]['topics']
            ]
            for domain in self.domains
        }

    async def generate_dataset(self) -> Tuple[List[Dict], List[Dict]]:
        """Generate enhanced dataset with AI interactions and C2PA provenance"""
        try:
            conversations = []
            metrics = []
            
            # Track diversity metrics
            authenticity_distribution = {
                "human_generated": 0,
                "human_assisted": 0,
                "ai_assisted": 0,
                "ai_generated": 0
            }
            
            # Domain distribution tracking
            domain_distribution = {
                "education": 0,
                "psychology": 0,
                "stem": 0
            }

            # Track verification status distribution
            verification_distribution = {
                VerificationStatus.VERIFIED: 0,
                VerificationStatus.PARTIALLY_VERIFIED: 0,
                VerificationStatus.UNVERIFIED: 0,
                VerificationStatus.DISPUTED: 0,
                VerificationStatus.PENDING: 0
            }
            
            # Updated authenticity distribution to include new category
            authenticity_distribution = {
                "human_generated": 0,
                "human_assisted": 0,  # Added this line
                "ai_assisted": 0,
                "ai_generated": 0
            }
            
            for i in range(self.config['size']):
                # Generate enhanced context
                context = await self._generate_research_context()
                
                # Generate conversation with AI interactions
                conversation = await self._generate_conversation_with_ai(context)
                
                # Determine verification status with controlled randomness
                verification_probabilities = {
                    VerificationStatus.VERIFIED: 0.3,
                    VerificationStatus.PARTIALLY_VERIFIED: 0.4,
                    VerificationStatus.UNVERIFIED: 0.2,
                    VerificationStatus.DISPUTED: 0.05,
                    VerificationStatus.PENDING: 0.05
                }
                
                verification_status = random.choices(
                    list(verification_probabilities.keys()), 
                    weights=list(verification_probabilities.values())
                )[0]
                verification_distribution[verification_status] += 1
                
                # Determine content authenticity
                ai_interactions = conversation.get("ai_interactions", [])
                total_messages = len(conversation.get("messages", []))
                
                # Classify content authenticity with domain context
                content_authenticity = classify_content_authenticity(
                    len(conversation.get('ai_interactions', [])),
                    len(conversation.get('messages', [])),
                    context.complexity,
                    context.domain
                )
                conversation['content_authenticity'] = content_authenticity
                # Update distribution tracking
                authenticity_distribution[content_authenticity] += 1
                domain_distribution[context.domain] += 1
                
                # Calculate verification details
                ai_influence_ratio = len(ai_interactions) / max(total_messages, 1)
                verification_details = {
                    "ai_interaction_score": min(ai_influence_ratio, 1.0),
                    "citation_quality": random.uniform(0.3, 1.0),
                    "methodology_rigor": random.uniform(0.4, 1.0)
                }
                
                # Calculate trust score
                trust_score = calculate_trust_score(
                    context, 
                    verification_status.value, 
                    ai_interactions
                )
                
                # Add attributes to conversation
                conversation["content_authenticity"] = content_authenticity
                conversation["trust_score"] = trust_score
                
                # Add provenance with enriched verification
                provenance = ContentProvenance(
                    content_id=conversation["id"],
                    user_id=str(uuid.uuid4()),
                    publication_timestamp=datetime.now(),
                    interaction_summary={},  # Populate from existing code
                    content_metadata={},     # Populate from existing code
                    verification_status=verification_status,
                    verification_details=verification_details
                )
                
                conversation["c2pa_provenance"] = asdict(provenance)
                
                # Calculate metrics
                metric = await self._calculate_enhanced_metrics(conversation, context)

                
                conversations.append(conversation)
                metrics.append(metric)
                
                if (i + 1) % 10 == 0:
                    self.logger.info(f"Generated {i + 1}/{self.config['size']} conversations")
            
            # Validate and report dataset diversity
            diversity_report = validate_dataset_diversity(conversations)
            
            # Log distribution insights
            self.logger.info("\nVerification Status Distribution:")
            for status, count in verification_distribution.items():
                self.logger.info(f"{status.value}: {count} ({count/self.config['size']*100:.2f}%)")
            
            # Log diversity insights
            self.logger.info("\nContent Authenticity Distribution:")
            for category, count in authenticity_distribution.items():
                self.logger.info(f"{category}: {count} ({count/self.config['size']*100:.2f}%)")
            
            self.logger.info("\nDomain Distribution:")
            for domain, count in domain_distribution.items():
                self.logger.info(f"{domain}: {count} ({count/self.config['size']*100:.2f}%)")
            
            # Optional: Log diversity report details
            self.logger.info("\nDataset Diversity Report:")
            for key, value in diversity_report.items():
                self.logger.info(f"{key}: {value}")
            
            return conversations, metrics
            
        except Exception as e:
            self.logger.error(f"Error generating dataset: {str(e)}")
            raise

    async def _generate_research_context(self) -> ResearchContext:
        """Generate enhanced research context with AI model information"""
        try:
            # Select AI model from config
            ai_model = self._select_ai_model()
            
            # Generate base context
            domain = self._select_random_domain()
            context = ResearchContext(
                domain=domain,
                topic=self._generate_topic(domain),
                methodology=self._select_methodology(domain),
                theoretical_framework=self._select_framework(domain),
                complexity=self._generate_complexity(),
                phase=self._select_initial_phase(),
                style=self._select_conversation_style(),
                research_questions=await self._generate_research_questions(domain),
                citations=await self._generate_citations(domain),
                variables=self._generate_research_variables(),
                ai_model=ai_model,
                ai_interaction_history=[],
                content_provenance={}
            )
            
            return context
            
        except Exception as e:
            self.logger.error(f"Error generating research context: {str(e)}")
            raise

    async def _generate_conversation_with_ai(
        self,
        context: ResearchContext
    ) -> Dict:
        """Generate conversation with AI interaction tracking"""
        try:
            conversation = {
                "id": str(uuid.uuid4()),
                "timestamp": datetime.now().isoformat(),
                "context": asdict(context),
                "messages": [],
                "ai_interactions": []
            }
            
            num_exchanges = self._get_conversation_length()
            current_phase = context.phase
            
            for i in range(num_exchanges):
                # Generate researcher message
                researcher_msg = await self._generate_researcher_message(
                    context,
                    current_phase,
                    i
                )
                
                # Log AI interaction
                ai_interaction = await self.ai_logger.log_interaction(
                    AIInteractionType.RESEARCH_ASSISTANCE,
                    {"message": researcher_msg["content"]},
                    {},  # Will be filled after assistant response
                    context
                )
                
                # Generate assistant response
                assistant_msg = await self._generate_assistant_response(
                    context,
                    researcher_msg,
                    ai_interaction
                )
                
                # Update AI interaction with response
                ai_interaction.output = {"message": assistant_msg["content"]}
                
                # Log user action (simulated acceptance)
                await self.ai_logger.log_user_action(
                    ai_interaction.interaction_id,
                    "accept",
                    {"confidence": 0.9}
                )
                
                conversation["messages"].extend([researcher_msg, assistant_msg])
                conversation["ai_interactions"].append(asdict(ai_interaction))
                
                # Update phase if needed
                if self._should_change_phase():
                    current_phase = self._get_next_phase(current_phase)
            
            return conversation
            
        except Exception as e:
            self.logger.error(f"Error generating conversation: {str(e)}")
            raise

    async def _add_c2pa_provenance(
        self,
        conversation: Dict,
        context: ResearchContext
    ) -> Dict:
        """Add C2PA provenance to conversation"""
        try:
            # Ensure context is converted to dictionary if it's an object
            context_dict = asdict(context) if hasattr(context, '__dict__') else context
            
            # Convert AI interactions back to objects
            ai_interactions = [
                AIInteraction(**interaction)
                for interaction in conversation["ai_interactions"]
            ]
            
            # Generate provenance
            provenance = await self.c2pa_manager.generate_provenance(
                conversation,
                context_dict,  # Pass dictionary representation
                ai_interactions
            )
            
            # Generate manifest
            manifest = await self.c2pa_manager.generate_manifest(
                conversation,
                provenance,
                ai_interactions
            )
            
            # Add to conversation
            conversation["c2pa_provenance"] = asdict(provenance)
            conversation["c2pa_manifest"] = manifest
            
            return conversation
            
        except Exception as e:
            self.logger.error(f"Error adding C2PA provenance: {str(e)}")
            return conversation  # Return original conversation if C2PA fails

    async def _calculate_enhanced_metrics(
        self,
        conversation: Dict,
        context: ResearchContext
    ) -> Dict:
        """Calculate comprehensive metrics including AI and C2PA aspects"""
        try:
            # Ensure context is a dictionary if it's a ResearchContext object
            context_dict = asdict(context) if hasattr(context, '__dict__') else context
            
            # Calculate base metrics
            try:
                # Check if calculate_metrics is a coroutine
                if asyncio.iscoroutinefunction(self.metrics_calculator.calculate_metrics):
                    base_metrics = await self.metrics_calculator.calculate_metrics(
                        conversation, 
                        context_dict
                    )
                else:
                    base_metrics = self.metrics_calculator.calculate_metrics(
                        conversation, 
                        context_dict
                    )
            except Exception as metric_error:
                print(f"Metrics calculation error: {metric_error}")
                base_metrics = {}
                    
            
            # Get AI interaction summary
            try:
                # Check if generate_interaction_summary is a coroutine
                if asyncio.iscoroutinefunction(self.ai_logger.generate_interaction_summary):
                    ai_metrics = await self.ai_logger.generate_interaction_summary(
                        conversation["id"]
                    )
                else:
                    ai_metrics = self.ai_logger.generate_interaction_summary(
                        conversation["id"]
                    )
            except Exception as ai_metric_error:
                print(f"AI metrics calculation error: {ai_metric_error}")
                ai_metrics = {}
            
            # Get C2PA verification if available
            try:
                c2pa_metrics = await self._calculate_c2pa_metrics(conversation)
            except Exception as c2pa_error:
                print(f"C2PA metrics calculation error: {c2pa_error}")
                c2pa_metrics = {}
            
            return {
                "base_metrics": base_metrics,
                "ai_interaction_metrics": ai_metrics,
                "c2pa_metrics": c2pa_metrics
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating metrics: {str(e)}")
            return {}

    async def _calculate_c2pa_metrics(self, conversation: Dict) -> Dict:
        """Calculate C2PA-specific metrics"""
        if not conversation.get("c2pa_manifest"):
            return {}
            
        try:
            # Make sure verification is awaitable 
            if hasattr(self.c2pa_manager.verify_content, '__await__'):
                verification = await self.c2pa_manager.verify_content(
                    conversation,
                    conversation["c2pa_manifest"]
                )
            else:
                # If not awaitable, call it directly
                verification = self.c2pa_manager.verify_content(
                    conversation,
                    conversation["c2pa_manifest"]
                )
            
            return {
                "verification_status": verification["verified"],
                "verification_details": verification["details"],
                "verified_at": verification["timestamp"]
            }
        except Exception as e:
            self.logger.error(f"Provenance calculation error: {type(e)}: {str(e)}")
            return {}

    def _select_ai_model(self) -> Dict:
        """Select AI model from configured models"""
        models = self.config['ai_settings']['models']
        model_name = next(iter(models))  # Get first model for now
        return {**models[model_name], "name": model_name}

    def _get_conversation_length(self) -> int:
        """Determine conversation length"""
        return random.randint(
            self.config['min_length'],
            self.config['max_length']
        )

    def _should_change_phase(self) -> bool:
        """Determine if conversation phase should change"""
        return random.random() < 0.3  # 30% chance to change phase

    # Add these helper methods to the UnifiedResearchGenerator class in generator.py

    # Methods for selecting domain, methodology, etc.
    def _select_random_domain(self):
        """Select a random domain from configured domains"""
        return random.choice(self.config['domains'])

    def _generate_topic(self, domain: str) -> str:
        """
        Generate a topic for a specific domain using predefined topics
        
        Args:
            domain: Research domain
        
        Returns:
            A randomly selected research topic from the domain
        """
        # Use topics from self.domains if the domain exists
        if domain in self.domains:
            return random.choice(self.domains[domain]['topics'])
        
        # Fallback if domain is not found
        return f"Research Topic in {domain}"

    def _select_methodology(self, domain):
        """Select appropriate methodology for the domain"""
        methodologies = {
            'education': ['Mixed Methods Research', 'Action Research', 'Case Study Analysis'],
            'psychology': ['Experimental Research', 'Survey Research', 'Longitudinal Study'],
            'stem': ['Empirical Analysis', 'Computational Modeling', 'Statistical Analysis']
        }
        domain_methods = methodologies.get(domain, ['Research Methodology'])
        return random.choice(domain_methods)

    def _select_framework(self, domain):
        """Select theoretical framework for the domain"""
        frameworks = {
            'education': ['Constructivist Learning Theory', 'Social Cognitive Theory'],
            'psychology': ['Cognitive Behavioral Theory', 'Humanistic Psychology'],
            'stem': ['Systems Theory', 'Information Processing Theory']
        }
        domain_frameworks = frameworks.get(domain, ['Theoretical Framework'])
        return random.choice(domain_frameworks)

    def _generate_complexity(self):
        """Generate complexity score between 0.3 and 1.0"""
        return random.uniform(0.3, 1.0)

    def _select_initial_phase(self):
        """Select initial conversation phase"""
        return random.choice(list(ConversationPhase))

    def _select_conversation_style(self):
        """Select conversation style"""
        return random.choice(list(ConversationStyle))

    async def _generate_research_questions(
        self, 
        domain: Optional[str] = None, 
        topic: Optional[str] = None
    ) -> List[str]:
        """
        Generate diverse research questions with domain and topic-specific templates
        
        Args:
            domain: Research domain (optional)
            topic: Specific research topic (optional)
        
        Returns:
            List of research questions
        """
        # Domain-specific templates
        domain_templates = {
            'education': [
                "How does {topic} impact student learning outcomes?",
                "What factors influence {topic} in educational settings?",
                "How can {topic} be effectively implemented in classrooms?",
                "What pedagogical strategies enhance {topic}?",
                "How does {topic} affect student engagement and motivation?"
            ],
            'psychology': [
                "What is the relationship between {topic} and behavior?",
                "How does {topic} affect cognitive development?",
                "What interventions are effective for {topic}?",
                "How do individual differences mediate {topic}?",
                "What psychological mechanisms underlie {topic}?"
            ],
            'stem': [
                "How can {topic} be optimized for better performance?",
                "What are the key variables affecting {topic}?",
                "How does {topic} compare to existing methodologies?",
                "What technological innovations can improve {topic}?",
                "What are the computational challenges in {topic}?"
            ],
            # Fallback templates for any domain
            'default': [
                "How does {topic} impact research outcomes?",
                "What are the key factors influencing {topic}?",
                "How can {topic} be effectively implemented?",
                "What role does technology play in {topic}?",
                "How do different contexts affect {topic}?"
            ]
        }

        # Determine the domain and topic
        if domain is None:
            domain = random.choice(list(self.domains.keys()))
        
        if topic is None:
            topic = self._generate_topic(domain)

        # Select templates
        templates = domain_templates.get(domain, domain_templates['default'])
        
        # Determine number of questions
        num_questions = random.randint(2, min(4, len(templates)))
        
        # Generate questions
        questions = [
            template.format(topic=topic)
            for template in random.sample(templates, num_questions)
        ]
        
        return questions

    async def _generate_citations(self, domain):
        """Generate citations for the domain"""
        authors = ["Smith et al.", "Johnson et al.", "Williams et al.", "Brown et al."]
        years = [str(year) for year in range(2018, 2024)]
        
        citations = []
        for _ in range(random.randint(2, 4)):
            citations.append({
                "author": random.choice(authors),
                "year": random.choice(years),
                "title": f"Research on {domain}",
                "journal": "Journal of Research",
                "doi": f"10.1000/jr.{random.randint(1000, 9999)}"
            })
        
        return citations

    def _generate_research_variables(self):
        """Generate research variables"""
        return {
            'dependent_var': "research_outcome",
            'independent_var': "research_intervention",
            'control_var': "research_baseline",
            'effect_size': round(random.uniform(0.1, 0.8), 2),
            'sample_size': random.randint(50, 500)
        }

    def _get_next_phase(self, current_phase):
        """Get next logical conversation phase"""
        phase_order = {
            ConversationPhase.INTRODUCTION: ConversationPhase.LITERATURE_REVIEW,
            ConversationPhase.LITERATURE_REVIEW: ConversationPhase.METHODOLOGY,
            ConversationPhase.METHODOLOGY: ConversationPhase.ANALYSIS,
            ConversationPhase.ANALYSIS: ConversationPhase.FINDINGS,
            ConversationPhase.FINDINGS: ConversationPhase.DISCUSSION,
            ConversationPhase.DISCUSSION: ConversationPhase.CONCLUSION,
            ConversationPhase.CONCLUSION: ConversationPhase.CONCLUSION
        }
        return phase_order.get(current_phase, current_phase)

    async def _generate_researcher_message(self, context, phase, position):
        """Generate researcher message"""
        return {
            "id": str(uuid.uuid4()),
            "timestamp": (datetime.now() + timedelta(minutes=position*5)).isoformat(),
            "role": "researcher",
            "content": f"Research question about {context.topic} in the {phase.value} phase",
            "metadata": {
                "phase": phase.value,
                "style": context.style.value,
                "position": position,
                "context_variables": list(context.variables.keys())
            }
        }

    async def _generate_assistant_response(self, context, researcher_msg, ai_interaction):
        """Generate assistant response"""
        return {
            "id": str(uuid.uuid4()),
            "timestamp": (datetime.now() + timedelta(minutes=researcher_msg["metadata"]["position"]*5 + 2)).isoformat(),
            "role": "assistant",
            "content": f"Response about {context.topic} using {context.methodology} methodology",
            "metadata": {
                "phase": researcher_msg["metadata"]["phase"],
                "style": context.style.value,
                "citations_used": [c["doi"] for c in context.citations[:2]],
                "theoretical_framework": context.theoretical_framework
            }
        }

# Example usage:
async def main():
    # Configuration
    config = {
        'size': 2,
        'domains': ['education'],
        'ai_settings': {
            'models': {
                'GPT-4': {
                    'version': '1.0',
                    'provider': 'OpenAI',
                    'capabilities': ['text_generation', 'analysis']
                }
            }
        }
    }
    
    # Initialize generator
    generator = UnifiedResearchGenerator(config)
    
    # Generate dataset
    conversations, metrics = await generator.generate_dataset()
    
    # Print sample results
    print(f"\nGenerated {len(conversations)} conversations")
    if conversations:
        print("\nSample conversation:")
        conversation = conversations[0]
        print(f"ID: {conversation['id']}")
        print(f"AI Interactions: {len(conversation['ai_interactions'])}")
        if 'c2pa_provenance' in conversation:
            print(f"C2PA Status: {conversation['c2pa_provenance']['verification_status']}")


# async def main():
#     """Example usage"""
#     config = {
#         'size': 10,
#         'min_length': 3,
#         'max_length': 5,
#         'domains': ['education', 'psychology'],
#         'template_settings': {
#             'use_dynamic_templates': True
#         }
#     }
    
#     generator = UnifiedResearchGeneratorA(config)
#     conversations, metrics = await generator.generate_dataset()
    
#     print(f"\nGenerated {len(conversations)} conversations")
#     print("\nSample conversation:")
#     sample_conv = conversations[0]
#     for msg in sample_conv["messages"][:4]:
#         print(f"\n{msg['role'].upper()}: {msg['content'][:100]}")
#     print(f"Context: {sample_conv['context']['topic']} - {sample_conv['context']['phase']}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())