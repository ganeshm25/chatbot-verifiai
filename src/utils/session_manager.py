
from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Optional

def save_conversation(conversation_id: str, messages: List[Dict]):
    with open(f"conversations/{conversation_id}.json", "w") as f:
        json.dump(messages, f)

def load_conversation(conversation_id: str) -> List[Dict]:
    try:
        with open(f"conversations/{conversation_id}.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return []