# File: research_generator/examples/generate_with_ai_c2pa.py

"""
Example script demonstrating usage of the enhanced research generator
with AI interaction logging and C2PA provenance tracking
"""

import asyncio
import json
import os
import argparse
from datetime import datetime
from utils.helpers import load_config, save_dataset, serialize_for_json

from research_generator.data_generation.generator import UnifiedResearchGenerator
from research_generator.config.default_config import DEFAULT_CONFIG, get_domain_specific_config
# Add to imports at the top of generate_with_ai_c2pa.py
import pandas as pd
from pathlib import Path

# Add this function to generate CSV files
def save_as_csv(conversations, metrics, output_dir):
    """Save dataset components as CSV files for easier analysis"""
    output_dir = Path(output_dir)
    
    # Ensure output directory exists
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # 1. Save basic conversation data
    conversations_basic = []
    for conv in conversations:
        conversations_basic.append({
            'conversation_id': conv.get('id', ''),
            'domain': conv.get('context', {}).get('domain', ''),
            'topic': conv.get('context', {}).get('topic', ''),
            'methodology': conv.get('context', {}).get('methodology', ''),
            'complexity': conv.get('context', {}).get('complexity', 0),
            'message_count': len(conv.get('messages', [])),
            'ai_interaction_count': len(conv.get('ai_interactions', [])),
            'verification_status': conv.get('c2pa_provenance', {}).get('verification_status', '')
        })
    
    df_conversations = pd.DataFrame(conversations_basic)
    df_conversations.to_csv(output_dir / 'conversations.csv', index=False)
    print(f"Saved {len(conversations_basic)} conversation records to {output_dir / 'conversations.csv'}")
    
    # 2. Save detailed messages data
    messages = []
    for conv in conversations:
        conv_id = conv.get('id', '')
        domain = conv.get('context', {}).get('domain', '')
        for msg in conv.get('messages', []):
            messages.append({
                'conversation_id': conv_id,
                'message_id': msg.get('id', ''),
                'timestamp': msg.get('timestamp', ''),
                'role': msg.get('role', ''),
                'domain': domain,
                'phase': msg.get('metadata', {}).get('phase', ''),
                'content': msg.get('content', '')[:500]  # Truncate long content
            })
    
    df_messages = pd.DataFrame(messages)
    df_messages.to_csv(output_dir / 'messages.csv', index=False)
    print(f"Saved {len(messages)} message records to {output_dir / 'messages.csv'}")
    
    # 3. Save AI interactions data
    ai_interactions = []
    for conv in conversations:
        conv_id = conv.get('id', '')
        domain = conv.get('context', {}).get('domain', '')
        for interaction in conv.get('ai_interactions', []):
            ai_interactions.append({
                'conversation_id': conv_id,
                'interaction_id': interaction.get('interaction_id', ''),
                'domain': domain,
                'timestamp': interaction.get('timestamp', ''),
                'interaction_type': interaction.get('interaction_type', ''),
                'ai_model': interaction.get('ai_model', {}).get('name', ''),
                'user_action_count': len(interaction.get('user_actions', [])),
                'input_content': str(interaction.get('input', {}))[:200],  # Truncate and convert to string
                'output_content': str(interaction.get('output', {}))[:200]  # Truncate and convert to string
            })
    
    df_interactions = pd.DataFrame(ai_interactions)
    df_interactions.to_csv(output_dir / 'ai_interactions.csv', index=False)
    print(f"Saved {len(ai_interactions)} AI interaction records to {output_dir / 'ai_interactions.csv'}")
    
    # 4. Save metrics data
    flattened_metrics = []
    for i, (conv, metric) in enumerate(zip(conversations, metrics)):
        conv_id = conv.get('id', '')
        domain = conv.get('context', {}).get('domain', '')
        
        # Extract and flatten key metrics
        base = metric.get('base_metrics', {})
        ai = metric.get('ai_interaction_metrics', {})
        c2pa = metric.get('c2pa_metrics', {})
        
        flattened_metrics.append({
            'conversation_id': conv_id,
            'domain': domain,
            'methodology_score': base.get('methodology_score', 0),
            'theoretical_score': base.get('theoretical_score', 0),
            'analytical_depth': base.get('analytical_depth', 0),
            'citation_quality': base.get('citation_quality', 0),
            'interaction_quality': ai.get('interaction_quality', 0),
            'user_engagement': ai.get('user_engagement', 0),
            'provenance_completeness': c2pa.get('provenance_completeness', 0),
            'verification_status': c2pa.get('verification_status', ''),
            'transparency_score': c2pa.get('transparency_score', 0)
        })
    
    df_metrics = pd.DataFrame(flattened_metrics)
    df_metrics.to_csv(output_dir / 'metrics.csv', index=False)
    print(f"Saved {len(flattened_metrics)} metric records to {output_dir / 'metrics.csv'}")

async def generate_dataset(config, output_dir):
    """Generate research dataset with AI and C2PA features"""
    print(f"\nInitializing generator with configuration...")
    generator = UnifiedResearchGenerator(config)
    
    print(f"\nGenerating dataset of {config['size']} conversations...")
    start_time = datetime.now()
    conversations, metrics = await generator.generate_dataset()
    end_time = datetime.now()
    
    print(f"\nGeneration completed in {(end_time - start_time).total_seconds():.2f} seconds")
    print(f"Generated {len(conversations)} conversations with AI interactions and C2PA provenance")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save complete dataset
    dataset_path = os.path.join(output_dir, "research_dataset_complete.json")
    with open(dataset_path, "w") as f:
        json.dump({
            "conversations": conversations,
            "metrics": metrics,
            "config": {k: v for k, v in config.items() if k != "template_paths"},
            "generated_at": datetime.now().isoformat()
        }, f, indent=2, default=str)
    print(f"\nSaved complete dataset to {dataset_path}")
    
    # Save a sample conversation for easy viewing
    if conversations:
        sample_path = os.path.join(output_dir, "sample_conversation.json")
        with open(sample_path, "w") as f:
            json.dump(conversations[0], f, indent=2, default=str)
        print(f"Saved sample conversation to {sample_path}")
    
    # Save metrics summary
    metrics_path = os.path.join(output_dir, "metrics_summary.json")
    with open(metrics_path, "w") as f:
        json.dump({
            "average_ai_interactions": sum(len(c.get("ai_interactions", [])) for c in conversations) / len(conversations),
            "verification_success_rate": sum(1 for c in conversations if c.get("c2pa_provenance", {}).get("verification_status") == "verified") / len(conversations),
            "metrics_by_domain": _calculate_metrics_by_domain(conversations, metrics),
            "ai_model_usage": _calculate_ai_model_usage(conversations)
        }, f, indent=2)
    print(f"Saved metrics summary to {metrics_path}")
    
    # Add CSV export
    save_as_csv(conversations, metrics, output_dir)
    
    return conversations, metrics

# async def generate_dataset(config, output_dir):
#     """Generate research dataset with AI and C2PA features"""
#     print(f"\nInitializing generator with configuration...")
#     generator = UnifiedResearchGenerator(config)
    
#     print(f"\nGenerating dataset of {config['size']} conversations...")
#     start_time = datetime.now()
#     conversations, metrics = await generator.generate_dataset()
#     end_time = datetime.now()
    
#     print(f"\nGeneration completed in {(end_time - start_time).total_seconds():.2f} seconds")
#     print(f"Generated {len(conversations)} conversations with AI interactions and C2PA provenance")
    
#     # Create output directory if it doesn't exist
#     os.makedirs(output_dir, exist_ok=True)
    
#     # Save complete dataset
#     dataset_path = os.path.join(output_dir, "research_dataset_complete.json")
#     with open(dataset_path, "w") as f:
#         json.dump({
#             "conversations": serialize_for_json(conversations),
#             "metrics": serialize_for_json(metrics),
#             "config": serialize_for_json({k: v for k, v in config.items() if k != "template_paths"}),
#             "generated_at": datetime.now().isoformat()
#         }, f, indent=2)
#     print(f"\nSaved complete dataset to {dataset_path}")
    
#     # Save a sample conversation for easy viewing
#     if conversations:
#         sample_path = os.path.join(output_dir, "sample_conversation.json")
#         with open(sample_path, "w") as f:
#             json.dump(conversations[0], f, indent=2, default=str)
#         print(f"Saved sample conversation to {sample_path}")
    
#     # Save metrics summary
#     metrics_path = os.path.join(output_dir, "metrics_summary.json")
#     with open(metrics_path, "w") as f:
#         json.dump({
#             "average_ai_interactions": sum(len(c.get("ai_interactions", [])) for c in conversations) / len(conversations),
#             "verification_success_rate": sum(1 for c in conversations if c.get("c2pa_provenance", {}).get("verification_status") == "verified") / len(conversations),
#             "metrics_by_domain": _calculate_metrics_by_domain(conversations, metrics),
#             "ai_model_usage": _calculate_ai_model_usage(conversations)
#         }, f, indent=2)
#     print(f"Saved metrics summary to {metrics_path}")
    
#     return conversations, metrics

def _calculate_metrics_by_domain(conversations, metrics):
    """Calculate metrics aggregated by domain"""
    domain_metrics = {}
    
    for i, conversation in enumerate(conversations):
        domain = conversation.get("context", {}).get("domain")
        if not domain:
            continue
            
        if domain not in domain_metrics:
            domain_metrics[domain] = {
                "count": 0,
                "avg_ai_interactions": 0,
                "avg_quality": 0
            }
        
        domain_metrics[domain]["count"] += 1
        domain_metrics[domain]["avg_ai_interactions"] += len(conversation.get("ai_interactions", []))
        
        # Add quality metric if available
        quality = metrics[i].get("quality", {}).get("overall_quality", 0)
        domain_metrics[domain]["avg_quality"] += quality
    
    # Calculate averages
    for domain, data in domain_metrics.items():
        if data["count"] > 0:
            data["avg_ai_interactions"] /= data["count"]
            data["avg_quality"] /= data["count"]
    
    return domain_metrics

def _calculate_ai_model_usage(conversations):
    """Calculate AI model usage statistics"""
    model_usage = {}
    
    for conversation in conversations:
        for interaction in conversation.get("ai_interactions", []):
            model = interaction.get("ai_model", {}).get("name", "unknown")
            
            if model not in model_usage:
                model_usage[model] = {
                    "count": 0,
                    "domains": {}
                }
            
            model_usage[model]["count"] += 1
            
            # Track domain usage
            domain = conversation.get("context", {}).get("domain", "unknown")
            model_usage[model]["domains"][domain] = model_usage[model]["domains"].get(domain, 0) + 1
    
    return model_usage

async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Generate research dataset with AI and C2PA features")
    parser.add_argument("--size", type=int, default=10, help="Number of conversations to generate")
    parser.add_argument("--domain", type=str, choices=["education", "psychology", "stem"], help="Focus on specific domain")
    parser.add_argument("--output", type=str, default="./output", help="Output directory")
    parser.add_argument("--ai-model", type=str, help="Specific AI model to use")
    parser.add_argument("--csv-only", action="store_true", help="Generate only CSV files, not JSON")
    args = parser.parse_args()
    
    # Start with default configuration
    config = DEFAULT_CONFIG.copy()
    
    # Apply command line overrides
    config["size"] = args.size
    
    # Apply domain-specific settings if requested
    if args.domain:
        config["domains"] = [args.domain]
        config = get_domain_specific_config(config, args.domain)
    
    # Apply AI model override if specified
    if args.ai_model and args.ai_model in config["ai_settings"]["models"]:
        config["ai_settings"]["model_selection_strategy"] = "specified"
        config["ai_settings"]["default_model"] = args.ai_model
    
    # Generate dataset
    conversations, metrics = await generate_dataset(config, args.output)
    
    # If CSV-only flag is set, remove JSON files
    if args.csv_only:
        for json_file in ["research_dataset_complete.json", "sample_conversation.json", "metrics_summary.json"]:
            json_path = os.path.join(args.output, json_file)
            if os.path.exists(json_path):
                os.remove(json_path)
                print(f"Removed JSON file: {json_path}")
#before CSV                
# async def main():
#     """Main entry point"""
#     parser = argparse.ArgumentParser(description="Generate research dataset with AI and C2PA features")
#     parser.add_argument("--size", type=int, default=10, help="Number of conversations to generate")
#     parser.add_argument("--domain", type=str, choices=["education", "psychology", "stem"], help="Focus on specific domain")
#     parser.add_argument("--output", type=str, default="./output", help="Output directory")
#     parser.add_argument("--ai-model", type=str, help="Specific AI model to use")
#     args = parser.parse_args()
    
#     # Start with default configuration
#     config = DEFAULT_CONFIG.copy()
    
#     # Apply command line overrides
#     config["size"] = args.size
    
#     # Apply domain-specific settings if requested
#     if args.domain:
#         config["domains"] = [args.domain]
#         config = get_domain_specific_config(config, args.domain)
    
#     # Apply AI model override if specified
#     if args.ai_model and args.ai_model in config["ai_settings"]["models"]:
#         config["ai_settings"]["model_selection_strategy"] = "specified"
#         config["ai_settings"]["default_model"] = args.ai_model
    
#     # Generate dataset
#     await generate_dataset(config, args.output)

if __name__ == "__main__":
    asyncio.run(main())