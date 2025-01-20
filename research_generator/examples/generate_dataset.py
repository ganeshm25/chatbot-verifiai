import asyncio
from data_generation import UnifiedResearchGenerator

async def generate_dataset():
    # Configuration for 1000 conversations
    config = {
        'size': 1,
        'min_length': 5,
        'max_length': 10,
        'domains': ['education', 'psychology', 'stem']
    }
    
    # Initialize generator
    generator = UnifiedResearchGenerator(config)
    
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
if __name__ == "__main__":
    asyncio.run(generate_dataset())