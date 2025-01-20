"""
Enhanced conversation patterns and templates for research generation
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Union
import random
from enum import Enum
from typing import Optional

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

@dataclass
class ResearchTemplate:
    """Template for research conversation generation"""
    phase: ConversationPhase
    style: ConversationStyle
    template: str
    variables: List[str]
    constraints: Dict[str, str]
    complexity: float

class PatternManager:
    """Manager for research conversation patterns and templates"""
    
    def __init__(self):
        self.templates = self._initialize_templates()
        self.patterns = self._initialize_patterns()
        self.transitions = self._initialize_transitions()
    
    def _initialize_templates(self) -> Dict[str, Dict[str, List[ResearchTemplate]]]:
        """Initialize comprehensive template repository"""
        return {
            'literature_review': {
                'formal': [
                    ResearchTemplate(
                        phase=ConversationPhase.LITERATURE_REVIEW,
                        style=ConversationStyle.FORMAL,
                        template="According to {author}'s ({year}) research on {topic}, {finding}. "
                                "This {methodology} study demonstrated that {result}.",
                        variables=['author', 'year', 'topic', 'finding', 'methodology', 'result'],
                        constraints={'year': 'range(2000, 2024)', 'methodology': 'methods_list'},
                        complexity=0.7
                    ),
                    ResearchTemplate(
                        phase=ConversationPhase.LITERATURE_REVIEW,
                        style=ConversationStyle.FORMAL,
                        template="The systematic review by {author} et al. ({year}) synthesized "
                                "{number} studies on {topic}, revealing {pattern}.",
                        variables=['author', 'year', 'number', 'topic', 'pattern'],
                        constraints={'number': 'range(10, 100)'},
                        complexity=0.8
                    )
                ],
                'analytical': [
                    ResearchTemplate(
                        phase=ConversationPhase.LITERATURE_REVIEW,
                        style=ConversationStyle.ANALYTICAL,
                        template="Critically examining {author}'s ({year}) findings, we observe {observation}. "
                                "This suggests {implication} when considering {context}.",
                        variables=['author', 'year', 'observation', 'implication', 'context'],
                        constraints={},
                        complexity=0.9
                    )
                ]
            },
            'methodology': {
                'formal': [
                    ResearchTemplate(
                        phase=ConversationPhase.METHODOLOGY,
                        style=ConversationStyle.FORMAL,
                        template="The study employed a {design} design with {sample_size} participants, "
                                "utilizing {method} for data collection. {analysis} was used to analyze the data.",
                        variables=['design', 'sample_size', 'method', 'analysis'],
                        constraints={'sample_size': 'range(50, 1000)'},
                        complexity=0.8
                    )
                ],
                'exploratory': [
                    ResearchTemplate(
                        phase=ConversationPhase.METHODOLOGY,
                        style=ConversationStyle.EXPLORATORY,
                        template="To investigate {research_question}, researchers developed {instrument} "
                                "which incorporated {feature}. This approach allowed for {advantage}.",
                        variables=['research_question', 'instrument', 'feature', 'advantage'],
                        constraints={},
                        complexity=0.7
                    )
                ]
            },
            'findings': {
                'analytical': [
                    ResearchTemplate(
                        phase=ConversationPhase.FINDINGS,
                        style=ConversationStyle.ANALYTICAL,
                        template="Analysis revealed {finding}, with a significance level of {p_value}. "
                                "The effect size ({effect_size}) indicates {interpretation}.",
                        variables=['finding', 'p_value', 'effect_size', 'interpretation'],
                        constraints={'p_value': 'float<0.05', 'effect_size': 'float<1.0'},
                        complexity=0.9
                    )
                ],
                'critical': [
                    ResearchTemplate(
                        phase=ConversationPhase.FINDINGS,
                        style=ConversationStyle.CRITICAL,
                        template="While the data supports {main_finding}, several limitations merit consideration: "
                                "{limitation_1}, {limitation_2}, and {limitation_3}.",
                        variables=['main_finding', 'limitation_1', 'limitation_2', 'limitation_3'],
                        constraints={},
                        complexity=0.8
                    )
                ]
            }
        }
    
    def _initialize_patterns(self) -> Dict[str, List[str]]:
        """Initialize conversation flow patterns"""
        return {
            'question_response': [
                "Could you elaborate on {topic}?",
                "What does the literature say about {aspect} of {topic}?",
                "How was {variable} measured in {study}?",
                "What are the implications of {finding} for {context}?"
            ],
            'analysis_patterns': [
                "Analyzing {study}'s methodology reveals {insight}.",
                "Comparing results across studies shows {pattern}.",
                "The evidence suggests {conclusion} because {reasoning}."
            ],
            'synthesis_patterns': [
                "Synthesizing findings from {study1} and {study2} indicates {synthesis}.",
                "Integrating perspectives from {field1} and {field2} suggests {insight}.",
                "Combined analysis of {methods} demonstrates {conclusion}."
            ]
        }
    
    def _initialize_transitions(self) -> Dict[str, List[str]]:
        """Initialize phase transition patterns"""
        return {
            'general': [
                "Building on this,",
                "Furthermore,",
                "Additionally,",
                "In contrast,",
                "However,"
            ],
            'phase_specific': {
                ConversationPhase.LITERATURE_REVIEW: [
                    "Examining the literature further,",
                    "Recent studies also show",
                    "Prior research indicates"
                ],
                ConversationPhase.METHODOLOGY: [
                    "Regarding methodology,",
                    "In terms of research design,",
                    "The analytical approach"
                ],
                ConversationPhase.FINDINGS: [
                    "The analysis revealed",
                    "Key findings indicate",
                    "Results demonstrate"
                ]
            }
        }
    
    def generate_conversation_flow(
        self,
        phase: ConversationPhase,
        style: ConversationStyle,
        complexity: float,
        context: Dict
    ) -> List[str]:
        """Generate conversation flow based on phase and style"""
        templates = self.templates.get(phase.value, {}).get(style.value, [])
        if not templates:
            return [f"Default template for {phase.value} phase"]
        
        # Filter by complexity
        suitable_templates = [t for t in templates if t.complexity <= complexity]
        if not suitable_templates:
            suitable_templates = templates
        
        # Generate conversation elements
        conversation = []
        
        # Add appropriate transition
        transition = self._select_transition(phase)
        if transition:
            conversation.append(transition)
        
        # Add main content using template
        template = random.choice(suitable_templates)
        content = self._fill_template(template, context)
        conversation.append(content)
        
        return conversation
    
    def _select_transition(self, phase: ConversationPhase) -> Optional[str]:
        """Select appropriate transition for the phase"""
        if random.random() < 0.7:  # 70% chance to add transition
            if random.random() < 0.5:  # 50% chance for phase-specific transition
                phase_transitions = self.transitions['phase_specific'].get(phase, [])
                if phase_transitions:
                    return random.choice(phase_transitions)
            
            return random.choice(self.transitions['general'])
        return None
    
    def _fill_template(self, template: ResearchTemplate, context: Dict) -> str:
        """Fill template with context values"""
        try:
            # Get values for all required variables
            values = {}
            for var in template.variables:
                if var in context:
                    values[var] = context[var]
                else:
                    values[var] = self._generate_value(var, template.constraints.get(var))
            
            return template.template.format(**values)
        except KeyError as e:
            return f"Error filling template: missing value for {e}"
    
    def _generate_value(self, variable: str, constraint: Optional[str]) -> str:
        """Generate appropriate value based on variable and constraints"""
        if not constraint:
            return f"{{{variable}}}"
        
        try:
            # Handle float constraints with comparison symbols
            if constraint.startswith('float<'):
                # Remove '<' and convert the rest to float
                limit = float(constraint[6:])
                return f"{random.random() * limit:.2f}"
            
            if constraint.startswith('range'):
                start, end = map(int, constraint[6:-1].split(','))
                return str(random.randint(start, end))
            
            elif constraint == 'methods_list':
                methods = ['quantitative', 'qualitative', 'mixed-methods', 'experimental']
                return random.choice(methods)
            
            return f"{{{variable}}}"
        
        except (ValueError, TypeError):
            # Fallback to default if conversion fails
            return f"{{{variable}}}"
        
        def get_templates_for_phase(self, phase: ConversationPhase) -> List[ResearchTemplate]:
            """Get all templates for a specific phase"""
            phase_templates = []
            for style_templates in self.templates.get(phase.value, {}).values():
                phase_templates.extend(style_templates)
            return phase_templates
        
        def get_patterns_for_style(self, style: ConversationStyle) -> List[str]:
            """Get conversation patterns for a specific style"""
            patterns = []
            for pattern_list in self.patterns.values():
                patterns.extend([p for p in pattern_list if style.value in p.lower()])
            return patterns

def main():
    """Example usage of PatternManager"""
    manager = PatternManager()
    
    # Example context
    context = {
        'author': 'Smith',
        'year': '2023',
        'topic': 'cognitive load in online learning',
        'finding': 'significant correlation between task complexity and learning outcomes',
        'methodology': 'mixed-methods',
        'result': 'increased cognitive load significantly impacted retention'
    }
    
    # Generate conversation flow
    flow = manager.generate_conversation_flow(
        phase=ConversationPhase.LITERATURE_REVIEW,
        style=ConversationStyle.ANALYTICAL,
        complexity=0.8,
        context=context
    )
    
    print("\nGenerated Conversation Flow:")
    for element in flow:
        print(f"\n{element}")

if __name__ == "__main__":
    main()