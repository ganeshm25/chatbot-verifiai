Generate both JSON and CSV outputs by default:
bashCopypython -m research_generator.examples.generate_with_ai_c2pa --size 20 --output ./output

Generate only CSV files (better for students who primarily use tools like Excel or R):
bashCopypython -m research_generator.examples.generate_with_ai_c2pa --size 20 --output ./output --csv-only

(main) $ python generate_dataset.py 
Generating advanced dataset...
No valid messages found
No valid messages found
No valid messages found
No valid messages found
No valid messages found
No valid messages found
No valid messages found
No valid messages found
No valid messages found
No valid messages found
Generated 50 conversations
First conversation has 32 messages

Sample Conversation:

RESEARCHER: Research on Mental Health Interventions in psychology continues to evolve....

ASSISTANT: Default template for conclusion phase...

RESEARCHER: Research on Mental Health Interventions in psychology continues to evolve....

ASSISTANT: Default template for conclusion phase...

=======================

python advanced_generate_dataset.py 


========================

python -m research_generator.examples.generate_with_ai_c2pa --size 10 --output ./output