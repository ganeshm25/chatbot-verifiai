import asyncio
from data_generation import UnifiedResearchGeneratorA
from examples.advanced_config import AdvancedResearchScenarios
from pathlib import Path

async def generate_dataset():
    # Configuration for 1000 conversations
    config = {
        'size': 1,
        'min_length': 5,
        'max_length': 10,
        'domains': ['education', 'psychology', 'stem']
    }
    
    # Initialize generator
    generator = UnifiedResearchGeneratorA(config)
    
    # Generate dataset using await
    conversations, metrics = await generator.generate_dataset()
    
    # Save dataset
    generator.save_dataset(
        conversations, 
        metrics, 
        base_filename="research_conversations"
    )
    
    # Optional: print some information
    print(f"Generated {len(conversations)} conversations")

# Run the async function
# if __name__ == "__main__":
#     asyncio.run(generate_dataset())



async def main():
    # Initialize the scenarios object with a Path object
    base_path = Path("output_data")
    scenarios = AdvancedResearchScenarios(base_path=base_path)
    
    # Generate advanced dataset (Scenario 1)
    print("Generating advanced dataset...")
    conversations, metrics = await scenarios.scenario_1_advanced_generation()
    
    print(f"Generated {len(conversations)} conversations")
    print(f"First conversation has {len(conversations[0]['messages'])} messages")
    
    # Print sample conversation
    sample_conv = conversations[0]
    print("\nSample Conversation:")
    for msg in sample_conv["messages"][:4]:  # Show first 4 messages
        print(f"\n{msg['role'].upper()}: {msg['content'][:100]}...")

if __name__ == "__main__":
    asyncio.run(main())