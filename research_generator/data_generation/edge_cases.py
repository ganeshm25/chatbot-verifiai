"""
Enhanced edge case generation for research conversations
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import random
import re
from enum import Enum
from .patterns import ConversationPhase, ConversationStyle, ResearchTemplate

class EdgeCaseType(Enum):
    CONTRADICTION = "contradiction"
    METHODOLOGY_MISMATCH = "methodology_mismatch"
    CITATION_ERROR = "citation_error"
    THEORETICAL_INCONSISTENCY = "theoretical_inconsistency"
    LOGICAL_FALLACY = "logical_fallacy"
    SAMPLING_BIAS = "sampling_bias"
    REPORTING_BIAS = "reporting_bias"
    MEASUREMENT_ERROR = "measurement_error"
    CONFOUNDING_VARIABLE = "confounding_variable"
    EXTERNAL_VALIDITY = "external_validity"

@dataclass
class EdgeCase:
    """Edge case definition with enhanced attributes"""
    type: EdgeCaseType
    probability: float
    templates: List[str]
    triggers: List[str]
    severity: float
    phase_applicability: List[ConversationPhase]
    style_compatibility: List[ConversationStyle]
    detection_patterns: List[str]

class EdgeCaseManager:
    """Enhanced manager for edge case generation and injection"""
    
    def __init__(self):
        self.edge_cases = self._initialize_edge_cases()
        self.severity_thresholds = {
            'low': 0.3,
            'medium': 0.6,
            'high': 0.8
        }
    
    def _initialize_edge_cases(self) -> Dict[str, EdgeCase]:
        """Initialize comprehensive edge case definitions"""
        return {
            'contradiction': EdgeCase(
                type=EdgeCaseType.CONTRADICTION,
                probability=0.15,
                templates=[
                    "While {initial_claim}, later evidence suggests {contrary_claim}",
                    "Despite claiming {claim_a}, the methodology shows {claim_b}",
                    "{source_a} argues {position_a}, but {source_b} demonstrates {position_b}",
                    "The initial hypothesis suggested {hypothesis}, yet the findings indicate {opposite_result}"
                ],
                triggers=['however', 'contrary', 'despite', 'nevertheless', 'yet', 'although'],
                severity=0.8,
                phase_applicability=[
                    ConversationPhase.LITERATURE_REVIEW,
                    ConversationPhase.FINDINGS,
                    ConversationPhase.DISCUSSION
                ],
                style_compatibility=[
                    ConversationStyle.ANALYTICAL,
                    ConversationStyle.CRITICAL
                ],
                detection_patterns=[
                    r"(\w+)\s+but\s+(\w+)",
                    r"despite.*,\s*(\w+)",
                    r"contrary to.*,\s*(\w+)"
                ]
            ),
            'methodology_mismatch': EdgeCase(
                type=EdgeCaseType.METHODOLOGY_MISMATCH,
                probability=0.12,
                templates=[
                    "Using {qual_method} to derive {quant_conclusion}",
                    "Applying {method} inappropriately to {context}",
                    "Drawing {conclusion_type} conclusions from {incompatible_data}",
                    "The study employs {inappropriate_method} despite {data_requirement}"
                ],
                triggers=['statistically significant', 'quantitative evidence', 'correlation'],
                severity=0.7,
                phase_applicability=[
                    ConversationPhase.METHODOLOGY,
                    ConversationPhase.ANALYSIS
                ],
                style_compatibility=[
                    ConversationStyle.ANALYTICAL,
                    ConversationStyle.CRITICAL,
                    ConversationStyle.FORMAL
                ],
                detection_patterns=[
                    r"qualitative.*quantitative",
                    r"statistical.*interview",
                    r"correlation.*case study"
                ]
            ),
            'citation_error': EdgeCase(
                type=EdgeCaseType.CITATION_ERROR,
                probability=0.1,
                templates=[
                    "As {misattributed_author} ({wrong_year}) suggests...",
                    "Building on work by {nonexistent_author}...",
                    "The landmark study by {author} ({year}) [no such study exists]...",
                    "According to {source}, although this finding appears in {different_source}"
                ],
                triggers=['according to', 'states that', 'suggests', 'argues'],
                severity=0.6,
                phase_applicability=[
                    ConversationPhase.LITERATURE_REVIEW,
                    ConversationPhase.DISCUSSION
                ],
                style_compatibility=[
                    ConversationStyle.FORMAL,
                    ConversationStyle.ANALYTICAL
                ],
                detection_patterns=[
                    r"\(\d{4}\)",
                    r"et al\.",
                    r"according to.*\(\d{4}\)"
                ]
            ),
            'theoretical_inconsistency': EdgeCase(
                type=EdgeCaseType.THEORETICAL_INCONSISTENCY,
                probability=0.08,
                templates=[
                    "Using {framework_a} concepts to explain {framework_b_phenomenon}",
                    "Applying {theory} in contradiction to its core assumptions about {concept}",
                    "Mixing incompatible theoretical frameworks: {framework_1} and {framework_2}"
                ],
                triggers=['theoretical framework', 'theory suggests', 'conceptual basis'],
                severity=0.75,
                phase_applicability=[
                    ConversationPhase.LITERATURE_REVIEW,
                    ConversationPhase.DISCUSSION
                ],
                style_compatibility=[
                    ConversationStyle.ANALYTICAL,
                    ConversationStyle.CRITICAL
                ],
                detection_patterns=[
                    r"theory of.*while.*theory of",
                    r"framework.*contradicts",
                    r"theoretical.*inconsistent"
                ]
            ),
            'sampling_bias': EdgeCase(
                type=EdgeCaseType.SAMPLING_BIAS,
                probability=0.1,
                templates=[
                    "The study's findings from {limited_sample} are generalized to {broad_population}",
                    "Despite using {biased_sample}, conclusions are drawn about {target_population}",
                    "Results based on {convenience_sample} are applied to {diverse_group}"
                ],
                triggers=['representative sample', 'population', 'generalizability'],
                severity=0.7,
                phase_applicability=[
                    ConversationPhase.METHODOLOGY,
                    ConversationPhase.FINDINGS
                ],
                style_compatibility=[
                    ConversationStyle.CRITICAL,
                    ConversationStyle.ANALYTICAL
                ],
                detection_patterns=[
                    r"sample of.*generalized to",
                    r"participants were.*conclusions about",
                    r"based on.*applied to"
                ]
            ),
            'confounding_variable': EdgeCase(
                type=EdgeCaseType.CONFOUNDING_VARIABLE,
                probability=0.09,
                templates=[
                    "Attributing {outcome} to {variable}, while ignoring {confounding_factor}",
                    "Claiming causation between {var_a} and {var_b} without controlling for {confounder}",
                    "Overlooking the role of {hidden_variable} in the relationship between {x} and {y}"
                ],
                triggers=['causes', 'leads to', 'results in', 'affects'],
                severity=0.75,
                phase_applicability=[
                    ConversationPhase.ANALYSIS,
                    ConversationPhase.FINDINGS
                ],
                style_compatibility=[
                    ConversationStyle.ANALYTICAL,
                    ConversationStyle.CRITICAL
                ],
                detection_patterns=[
                    r"causes.*without considering",
                    r"affects.*ignoring",
                    r"relationship between.*overlooking"
                ]
            )
        }
    
    def inject_edge_cases(
        self,
        conversation: List[Dict],
        phase: ConversationPhase,
        style: ConversationStyle,
        context: Dict
    ) -> List[Dict]:
        """Inject edge cases into conversation"""
        injected_conversation = []
        
        for message in conversation:
            injected_conversation.append(message)
            
            # Decide whether to inject edge case after this message
            if self._should_inject_edge_case(phase, style):
                edge_case = self._generate_edge_case(phase, style, context)
                if edge_case:
                    injected_conversation.append(edge_case)
        
        return injected_conversation
    
    def _should_inject_edge_case(
        self,
        phase: ConversationPhase,
        style: ConversationStyle
    ) -> bool:
        """Determine if edge case should be injected"""
        # Get applicable edge cases for this phase and style
        applicable_cases = [
            case for case in self.edge_cases.values()
            if phase in case.phase_applicability
            and style in case.style_compatibility
        ]
        
        if not applicable_cases:
            return False
        
        # Use highest probability among applicable cases
        max_probability = max(case.probability for case in applicable_cases)
        return random.random() < max_probability
    
    def _generate_edge_case(
        self,
        phase: ConversationPhase,
        style: ConversationStyle,
        context: Dict
    ) -> Optional[Dict]:
        """Generate appropriate edge case for given phase and style"""
        # Filter applicable edge cases
        applicable_cases = [
            case for case in self.edge_cases.values()
            if phase in case.phase_applicability
            and style in case.style_compatibility
        ]
        
        if not applicable_cases:
            return None
        
        # Select edge case based on probabilities
        weights = [case.probability for case in applicable_cases]
        selected_case = random.choices(applicable_cases, weights=weights, k=1)[0]
        
        # Generate edge case content
        template = random.choice(selected_case.templates)
        content = self._fill_edge_case_template(template, context)
        
        return {
            'type': 'edge_case',
            'edge_case_type': selected_case.type.value,
            'content': content,
            'severity': selected_case.severity,
            'phase': phase.value,
            'style': style.value
        }
    
    def _fill_edge_case_template(self, template: str, context: Dict) -> str:
        """Fill edge case template with appropriate values"""
        try:
            # Extract required variables from template
            variables = re.findall(r'\{(\w+)\}', template)
            
            # Generate values for each variable
            values = {}
            for var in variables:
                if var in context:
                    values[var] = context[var]
                else:
                    values[var] = self._generate_edge_case_value(var)
            
            return template.format(**values)
        except KeyError as e:
            return f"Error generating edge case: missing value for {e}"
    
    def _generate_edge_case_value(self, variable: str) -> str:
        """Generate appropriate value for edge case variable"""
        # Add specific value generators for different variable types
        value_generators = {
            'author': lambda: random.choice(['Smith', 'Johnson', 'Williams', 'Brown', 'Jones']),
            'year': lambda: str(random.randint(2000, 2024)),
            'method': lambda: random.choice(['survey', 'interview', 'observation', 'experiment']),
            'sample': lambda: f"sample of {random.randint(10, 1000)} participants",
            'theory': lambda: random.choice([
                'Social Learning Theory',
                'Cognitive Load Theory',
                'Self-Determination Theory',
                'Activity Theory'
            ])
        }
        
        if variable in value_generators:
            return value_generators[variable]()
        
        return f"{{{variable}}}"
    
    def detect_edge_cases(self, conversation: List[Dict]) -> List[Dict]:
        """Detect and analyze edge cases in conversation"""
        detected_cases = []
        
        for message in conversation:
            content = message.get('content', '')
            
            # Check each edge case type
            for case_name, case in self.edge_cases.items():
                # Check for triggers
                trigger_matches = any(
                    trigger.lower() in content.lower()
                    for trigger in case.triggers
                )
                
                # Check for patterns
                pattern_matches = any(
                    re.search(pattern, content, re.IGNORECASE)
                    for pattern in case.detection_patterns
                )
                
                if trigger_matches or pattern_matches:
                    detected_cases.append({
                        'message_id': message.get('id'),
                        'edge_case_type': case_name,
                        'severity': case.severity,
                        'triggers_found': [
                            t for t in case.triggers
                            if t.lower() in content.lower()
                        ],
                        'patterns_matched': [
                            p for p in case.detection_patterns
                            if re.search(p, content, re.IGNORECASE)
                        ]
                    })
        
        return detected_cases

def main():
    """Example usage of EdgeCaseManager"""
    manager = EdgeCaseManager()
    
    # Example context
    context = {
        'author': 'Smith',
        'year': '2023',
        'method': 'qualitative interviews',
        'sample': '15 participants',
        'theory': 'Grounded Theory'
    }
    
    # Generate edge case
    edge_case = manager._generate_edge_case(
        phase=ConversationPhase.METHODOLOGY,
        style=ConversationStyle.ANALYTICAL,
        context=context
    )
    
    print("\nGenerated Edge Case:")
    print(f"Type: {edge_case['edge_case_type']}")
    print(f"Content: {edge_case['content']}")
    print(f"Severity: {edge_case['severity']}")

if __name__ == "__main__":
    main()