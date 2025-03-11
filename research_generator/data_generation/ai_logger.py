# File: research_generator/data_generation/ai_logger.py

import logging
from typing import Dict, List, Optional, Union
from datetime import datetime
import uuid
import json
from dataclasses import asdict

from .models import AIInteraction, AIInteractionType, ResearchContext

class AIInteractionLogger:
    """Component for logging and managing AI interactions"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.interactions: List[AIInteraction] = []
        self.session_metadata: Dict = {}
    
    async def log_interaction(
        self,
        interaction_type: AIInteractionType,
        input_data: Dict,
        output_data: Dict,
        context: ResearchContext
    ) -> AIInteraction:
        """Log a single AI interaction"""
        try:
            interaction = AIInteraction(
                interaction_id=str(uuid.uuid4()),
                timestamp=datetime.now(),
                content_id=str(uuid.uuid4()),  # or from context if available
                interaction_type=interaction_type,
                input=input_data,
                output=output_data,
                user_actions=[],  # Will be populated as actions occur
                ai_model=context.ai_model,
                metadata={
                    "domain": context.domain,
                    "phase": context.phase.value,
                    "complexity": context.complexity,
                    "logged_at": datetime.now().isoformat()
                }
            )
            
            self.interactions.append(interaction)
            self.logger.info(f"Logged interaction: {interaction.interaction_id}")
            
            return interaction
            
        except Exception as e:
            self.logger.error(f"Error logging interaction: {str(e)}")
            raise

    async def log_user_action(
        self,
        interaction_id: str,
        action: str,
        details: Optional[Dict] = None
    ) -> bool:
        """Log user action for a specific interaction"""
        try:
            interaction = self._find_interaction(interaction_id)
            if interaction:
                user_action = {
                    "action": action,
                    "timestamp": datetime.now().isoformat(),
                    "details": details or {}
                }
                interaction.user_actions.append(user_action)
                self.logger.info(f"Logged user action for interaction: {interaction_id}")
                return True
            return False
        except Exception as e:
            self.logger.error(f"Error logging user action: {str(e)}")
            return False

    async def generate_interaction_summary(
        self,
        content_id: Optional[str] = None
    ) -> Dict:
        """Generate summary of AI interactions"""
        try:
            # Filter interactions if content_id provided
            relevant_interactions = [
                i for i in self.interactions
                if not content_id or i.content_id == content_id
            ]
            
            if not relevant_interactions:
                return self._generate_empty_summary()
            
            return {
                "total_interactions": len(relevant_interactions),
                "interaction_types": self._count_interaction_types(relevant_interactions),
                "temporal_analysis": self._analyze_temporal_patterns(relevant_interactions),
                "user_engagement": self._analyze_user_engagement(relevant_interactions),
                "model_usage": self._analyze_model_usage(relevant_interactions),
                "generated_at": datetime.now().isoformat()
            }
        
        except Exception as e:
            self.logger.error(f"Error generating interaction summary: {str(e)}")
            return self._generate_empty_summary()

    def _find_interaction(self, interaction_id: str) -> Optional[AIInteraction]:
        """Find interaction by ID"""
        return next(
            (i for i in self.interactions if i.interaction_id == interaction_id),
            None
        )

    def _count_interaction_types(self, interactions: List[AIInteraction]) -> Dict[str, int]:
        """Count occurrences of each interaction type"""
        type_counts = {}
        for interaction in interactions:
            type_name = interaction.interaction_type.value
            type_counts[type_name] = type_counts.get(type_name, 0) + 1
        return type_counts

    def _analyze_temporal_patterns(self, interactions: List[AIInteraction]) -> Dict:
        """Analyze temporal patterns in interactions"""
        if not interactions:
            return {}
            
        sorted_interactions = sorted(interactions, key=lambda x: x.timestamp)
        
        return {
            "first_interaction": sorted_interactions[0].timestamp.isoformat(),
            "last_interaction": sorted_interactions[-1].timestamp.isoformat(),
            "total_duration_seconds": (
                sorted_interactions[-1].timestamp - 
                sorted_interactions[0].timestamp
            ).total_seconds(),
            "interaction_timeline": [
                {
                    "timestamp": i.timestamp.isoformat(),
                    "type": i.interaction_type.value,
                    "has_user_actions": bool(i.user_actions)
                }
                for i in sorted_interactions
            ]
        }

    def _analyze_user_engagement(self, interactions: List[AIInteraction]) -> Dict:
        """Analyze user engagement patterns"""
        total_actions = sum(len(i.user_actions) for i in interactions)
        
        return {
            "total_user_actions": total_actions,
            "average_actions_per_interaction": (
                total_actions / len(interactions) if interactions else 0
            ),
            "action_types": self._count_user_action_types(interactions)
        }

    def _analyze_model_usage(self, interactions: List[AIInteraction]) -> Dict:
        """Analyze AI model usage patterns"""
        model_usage = {}
        
        for interaction in interactions:
            model_name = interaction.ai_model.get("name", "unknown")
            if model_name not in model_usage:
                model_usage[model_name] = {
                    "count": 0,
                    "successful_interactions": 0,
                    "version": interaction.ai_model.get("version", "unknown")
                }
            
            model_usage[model_name]["count"] += 1
            # Assuming success if there are user actions or non-empty output
            if interaction.user_actions or interaction.output:
                model_usage[model_name]["successful_interactions"] += 1
        
        return model_usage

    def _count_user_action_types(self, interactions: List[AIInteraction]) -> Dict[str, int]:
        """Count different types of user actions"""
        action_counts = {}
        for interaction in interactions:
            for action in interaction.user_actions:
                action_type = action["action"]
                action_counts[action_type] = action_counts.get(action_type, 0) + 1
        return action_counts

    def _generate_empty_summary(self) -> Dict:
        """Generate empty summary structure"""
        return {
            "total_interactions": 0,
            "interaction_types": {},
            "temporal_analysis": {},
            "user_engagement": {
                "total_user_actions": 0,
                "average_actions_per_interaction": 0,
                "action_types": {}
            },
            "model_usage": {},
            "generated_at": datetime.now().isoformat()
        }

    async def export_interactions(
        self,
        format: str = "json",
        content_id: Optional[str] = None
    ) -> Union[str, Dict]:
        """Export interactions in specified format"""
        try:
            # Filter interactions if content_id provided
            relevant_interactions = [
                asdict(i) for i in self.interactions
                if not content_id or i.content_id == content_id
            ]
            
            if format.lower() == "json":
                return {
                    "interactions": relevant_interactions,
                    "metadata": self.session_metadata,
                    "exported_at": datetime.now().isoformat()
                }
            else:
                raise ValueError(f"Unsupported export format: {format}")
                
        except Exception as e:
            self.logger.error(f"Error exporting interactions: {str(e)}")
            return {}

# Example usage:
async def main():
    # Initialize logger
    ai_logger = AIInteractionLogger()
    
    # Create sample interaction
    context = ResearchContext(
        domain="education",
        topic="cognitive load",
        methodology="mixed methods",
        theoretical_framework="cognitive load theory",
        complexity=0.8,
        phase=ConversationPhase.LITERATURE_REVIEW,
        style=ConversationStyle.ANALYTICAL,
        research_questions=[],
        citations=[],
        variables={},
        ai_model={
            "name": "GPT-4",
            "version": "1.0",
            "provider": "OpenAI"
        },
        ai_interaction_history=[],
        content_provenance={}
    )
    
    # Log interaction
    interaction = await ai_logger.log_interaction(
        interaction_type=AIInteractionType.RESEARCH_ASSISTANCE,
        input_data={"query": "Analyze cognitive load impact"},
        output_data={"response": "Analysis of cognitive load..."},
        context=context
    )
    
    # Log user action
    await ai_logger.log_user_action(
        interaction_id=interaction.interaction_id,
        action="accept",
        details={"confidence": 0.95}
    )
    
    # Generate summary
    summary = await ai_logger.generate_interaction_summary()
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())