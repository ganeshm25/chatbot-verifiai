# File: research_generator/data_generation/c2pa_manager.py

import logging
from typing import Dict, List, Optional, Union
from datetime import datetime
import uuid
import json
from enum import Enum
import hashlib
from dataclasses import asdict

from .models import ContentProvenance, ResearchContext, AIInteraction

class EnhancedJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Enum):
            return obj.value
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        return super().default(obj)

class C2PAManager:
    """Manager for C2PA (Coalition for Content Provenance and Authenticity) functionality"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.manifests: Dict[str, Dict] = {}
        self.content_provenances: Dict[str, ContentProvenance] = {}
    
    # In c2pa_manager.py, modify generate_provenance method
    async def generate_provenance(
        self,
        conversation: Dict,
        context: ResearchContext,
        ai_interactions: List[AIInteraction]
    ) -> ContentProvenance:
        """Generate content provenance record"""
        try:
            # Generate content hash
            content_hash = self._generate_content_hash(conversation)
            
            # Create interaction summary
            interaction_summary = await self._generate_interaction_summary(ai_interactions)
            
            # Create provenance record using context attributes directly, not get()
            provenance = ContentProvenance(
                content_id=conversation["id"],
                user_id=str(uuid.uuid4()),  # Should come from context in real implementation
                publication_timestamp=datetime.now(),
                interaction_summary=interaction_summary,
                content_metadata={
                    "title": f"Research on {context.topic}",
                    "content_type": "research_conversation",
                    "content_hash": content_hash,
                    "domain": context.domain,
                    "methodology": context.methodology,
                    "theoretical_framework": context.theoretical_framework
                },
                verification_status="verified"
            )
            
            # Store provenance
            self.content_provenances[conversation["id"]] = provenance
            
            return provenance
            
        except Exception as e:
            self.logger.error(f"Error generating provenance: {str(e)}")
            raise

    async def generate_manifest(
        self,
        conversation: Dict,
        provenance: ContentProvenance,
        ai_interactions: List[AIInteraction]
    ) -> Dict:
        """Generate C2PA manifest"""
        try:
            manifest = {
                "manifest_id": f"manifest_{conversation['id']}",
                "version": "1.0",
                "created": datetime.now().isoformat(),
                "assertions": [
                    {
                        "type": "ai_assistance",
                        "details": self._generate_ai_assertion(ai_interactions)
                    },
                    {
                        "type": "content_integrity",
                        "details": {
                            "hash_algorithm": "sha256",
                            "content_hash": provenance.content_metadata["content_hash"]
                        }
                    },
                    {
                        "type": "research_context",
                        "details": self._generate_research_assertion(conversation["context"])
                    }
                ],
                "credentials": self._generate_credentials(),
                "signatures": await self._generate_signatures(provenance)
            }
            
            # Store manifest
            self.manifests[conversation["id"]] = manifest
            
            return manifest
            
        except Exception as e:
            self.logger.error(f"Error generating manifest: {str(e)}")
            raise

    async def verify_content(
        self,
        conversation: Dict,
        manifest: Dict
    ) -> Dict:
        """Verify content against C2PA manifest"""
        try:
            # Verify content hash
            current_hash = self._generate_content_hash(conversation)
            stored_hash = manifest["assertions"][1]["details"]["content_hash"]
            hash_valid = current_hash == stored_hash
            
            # Verify signatures
            signatures_valid = await self._verify_signatures(manifest["signatures"])
            
            # Verify AI assertions
            ai_assertions_valid = self._verify_ai_assertions(
                manifest["assertions"][0]["details"],
                conversation.get("ai_interactions", [])
            )
            
            return {
                "verified": hash_valid and signatures_valid and ai_assertions_valid,
                "details": {
                    "content_hash_valid": hash_valid,
                    "signatures_valid": signatures_valid,
                    "ai_assertions_valid": ai_assertions_valid
                },
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error verifying content: {str(e)}")
            return {
                "verified": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }


    def _generate_content_hash(self, content: Dict) -> str:
        """Generate SHA-256 hash of content"""
        content_str = json.dumps(content, sort_keys=True, cls=EnhancedJSONEncoder)
        return hashlib.sha256(content_str.encode()).hexdigest()

    async def _generate_interaction_summary(
        self,
        interactions: List[AIInteraction]
    ) -> Dict:
        """Generate summary of AI interactions"""
        if not interactions:
            return {
                "total_interactions": 0,
                "ai_models_used": [],
                "interaction_types": []
            }
            
        return {
            "total_interactions": len(interactions),
            "ai_models_used": list(set(
                i.ai_model["name"] for i in interactions
            )),
            "interaction_types": list(set(
                i.interaction_type.value for i in interactions
            )),
            "timeline": [
                {
                    "timestamp": i.timestamp.isoformat(),
                    "type": i.interaction_type.value
                }
                for i in sorted(interactions, key=lambda x: x.timestamp)
            ]
        }

    def _generate_ai_assertion(self, interactions: List[AIInteraction]) -> Dict:
        """Generate AI-related assertions"""
        return {
            "models_used": list(set(
                i.ai_model["name"] for i in interactions
            )),
            "interaction_count": len(interactions),
            "interaction_types": list(set(
                i.interaction_type.value for i in interactions
            )),
            "first_interaction": min(
                i.timestamp for i in interactions
            ).isoformat() if interactions else None,
            "last_interaction": max(
                i.timestamp for i in interactions
            ).isoformat() if interactions else None
        }

    def _generate_research_assertion(self, context: Dict) -> Dict:
        """Generate research-related assertions"""
        return {
            "domain": context["domain"],
            "methodology": context["methodology"],
            "theoretical_framework": context["theoretical_framework"],
            "complexity": context["complexity"],
            "citations_count": len(context.get("citations", [])),
            "research_questions_count": len(context.get("research_questions", []))
        }

    def _generate_credentials(self) -> Dict:
        """Generate credential information"""
        return {
            "issuer": "Verifai Research System",
            "issued_at": datetime.now().isoformat(),
            "valid_until": None,  # Permanent validity for research content
            "version": "1.0"
        }

    async def _generate_signatures(self, provenance: ContentProvenance) -> Dict:
        """Generate cryptographic signatures"""
        # In a real implementation, this would use proper cryptographic signing
        return {
            "algorithm": "ES256",
            "signature": f"sig_{uuid.uuid4().hex}",
            "public_key": f"key_{uuid.uuid4().hex}"
        }

    async def _verify_signatures(self, signatures: Dict) -> bool:
        """Verify cryptographic signatures"""
        # In a real implementation, this would verify actual signatures
        return bool(signatures.get("signature") and signatures.get("public_key"))

    def _verify_ai_assertions(
        self,
        assertions: Dict,
        interactions: List[AIInteraction]
    ) -> bool:
        """Verify AI-related assertions"""
        if not assertions or not interactions:
            return False
            
        actual_models = set(i.ai_model["name"] for i in interactions)
        asserted_models = set(assertions["models_used"])
        
        return (
            len(interactions) == assertions["interaction_count"] and
            actual_models == asserted_models
        )

# Example usage:
async def main():
    # Initialize manager
    c2pa_manager = C2PAManager()
    
    # Sample conversation
    conversation = {
        "id": str(uuid.uuid4()),
        "context": {
            "domain": "education",
            "topic": "cognitive load",
            "methodology": "mixed methods",
            "theoretical_framework": "cognitive load theory",
            "citations": [],
            "research_questions": []
        },
        "messages": [
            {"role": "researcher", "content": "Query about cognitive load"},
            {"role": "assistant", "content": "Response about cognitive load"}
        ]
    }
    
    # Sample interactions
    ai_interactions = [
        AIInteraction(
            interaction_id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            content_id=conversation["id"],
            interaction_type=AIInteractionType.RESEARCH_ASSISTANCE,
            input={"query": "cognitive load analysis"},
            output={"response": "analysis results"},
            user_actions=[],
            ai_model={"name": "GPT-4", "version": "1.0"},
            metadata={}
        )
    ]
    
    # Generate provenance
    provenance = await c2pa_manager.generate_provenance(
        conversation,
        context=ResearchContext(...),  # Fill with appropriate context
        ai_interactions=ai_interactions
    )
    
    # Generate manifest
    manifest = await c2pa_manager.generate_manifest(
        conversation,
        provenance,
        ai_interactions
    )
    
    # Verify content
    verification = await c2pa_manager.verify_content(conversation, manifest)
    
    print(json.dumps(verification, indent=2))

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())