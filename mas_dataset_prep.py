import json
from datetime import datetime
from pathlib import Path
import argparse
import asyncio
from typing import Dict, List

# Constants (keep in sync with mas_basic.py)
BASE_DIR = Path('data')
CONVERSATION_DIR = BASE_DIR / 'conversations'
TRAINING_DIR = BASE_DIR / 'training'

# Ensure directories exist
for dir_path in [CONVERSATION_DIR, TRAINING_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

class TrainingDataProcessor:
    def __init__(self):
        self.conversations_dir = CONVERSATION_DIR
        self.training_dir = TRAINING_DIR
        
    def process_conversations(self, output_name: str = "training_dataset", agent_filter: str = None, tool_filter: str = None) -> None:
        """
        Process conversations into training format
        
        Args:
            output_name: Base name for the output file
            agent_filter: Optional name of specific agent to filter for (e.g., 'Operator', 'Subtraction Agent')
            tool_filter: Optional name of specific tool to filter for (e.g., 'delegate_to_agent', 'subtract_two_numbers')
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Build filename based on filters
        filename_parts = [output_name]
        if agent_filter:
            filename_parts.append(f"Agent_{agent_filter.replace(' ', '_')}")
        if tool_filter:
            filename_parts.append(f"Tool_{tool_filter}")
        if not (agent_filter or tool_filter):
            filename_parts.append("full")
        
        filename_parts.append(timestamp)
        final_output_file = self.training_dir / f"{'_'.join(filename_parts)}.json"
        
        training_examples = []
        
        # Process each conversation file
        for conv_file in self.conversations_dir.glob("conversation_*.json"):
            with open(conv_file, 'r') as f:
                conversation = json.load(f)
                examples = self._extract_training_examples(
                    conversation, 
                    agent_filter=agent_filter,
                    tool_filter=tool_filter
                )
                training_examples.extend(examples)
        
        # Save directly to final format
        with open(final_output_file, 'w') as f:
            json.dump(training_examples, f, indent=2)
        
        print(f"\nTraining data saved to: {final_output_file}")
        print(f"Total examples processed: {len(training_examples)}")
        if agent_filter:
            print(f"Filtered for agent: {agent_filter}")
        if tool_filter:
            print(f"Filtered for tool: {tool_filter}")

    def _transform_to_finetune_format(self, input_json_path: Path, output_json_path: Path) -> None:
        """Transform raw training data into fine-tuning format"""
        with open(input_json_path, "r") as f:
            raw_examples = json.load(f)  # Now loading directly as a list
        
        finetune_data = []

        for ex in raw_examples:
            if "input" in ex and "output" in ex:
                # Already in the correct format
                finetune_data.append(ex)
                continue

            # Process examples that need transformation
            context = ex.get("context", [])
            if not context:
                continue
            
            user_msg = None
            for msg in reversed(context):
                if msg["role"] == "user":
                    user_msg = msg["content"]
                    break
            
            if not user_msg:
                continue

            tool_calls = ex.get("tool_calls", [])
            if not tool_calls:
                continue

            for tc in tool_calls:
                training_ex = {
                    "input": user_msg,
                    "output": {
                        "function_call": {
                            "name": tc["name"],
                            "arguments": tc["arguments"]
                        }
                    }
                }
                finetune_data.append(training_ex)

        with open(output_json_path, "w") as out:
            json.dump(finetune_data, out, indent=2)

    def _extract_training_examples(self, data: Dict, agent_filter: str = None, tool_filter: str = None) -> List[Dict]:
        """
        Extract training examples from conversations with optional filtering.
        
        Args:
            data: The conversation data
            agent_filter: Optional agent name to filter for
            tool_filter: Optional tool name to filter for
        """
        examples = []
        
        # Get the conversations list from the data
        conversations = data.get("conversations", [])
        
        for conv in conversations:
            # Skip if agent filter is set and doesn't match
            if agent_filter and conv.get("agent") != agent_filter:
                continue
                
            messages = conv.get("messages", [])
            llm_response = conv.get("llm_response", {})
            
            # Get the tool calls from the LLM response
            tool_calls = llm_response.get("message", {}).get("tool_calls", [])
            
            if tool_calls:
                # Find the last user message before this response
                last_user_msg = None
                for msg in messages:
                    if msg["role"] == "user":
                        last_user_msg = msg["content"]
                
                if last_user_msg:
                    # Create an example for each tool call
                    for call in tool_calls:
                        if call.get("type") == "function":
                            function_info = call.get("function", {})
                            tool_name = function_info.get("name")
                            
                            # Skip if tool filter is set and doesn't match
                            if tool_filter and tool_name != tool_filter:
                                continue
                                
                            ex = {
                                "input": last_user_msg,
                                "output": {
                                    "function_call": {
                                        "name": tool_name,
                                        "arguments": function_info.get("arguments", {})
                                    }
                                }
                            }
                            examples.append(ex)
        
        return examples


async def process_training_data(output_name: str = None, agent_filter: str = None, tool_filter: str = None) -> None:
    """
    Process all conversations into training data with optional filtering
    
    Args:
        output_name: Optional custom name for the output file
        agent_filter: Optional agent name to filter for
        tool_filter: Optional tool name to filter for
    """
    training_processor = TrainingDataProcessor()
    if output_name:
        training_processor.process_conversations(
            output_name,
            agent_filter=agent_filter,
            tool_filter=tool_filter
        )
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        training_processor.process_conversations(
            f"training_dataset_{timestamp}",
            agent_filter=agent_filter,
            tool_filter=tool_filter
        )
    print("\nTraining data processed and saved!")

def main():
    parser = argparse.ArgumentParser(description='Process conversations into training data')
    parser.add_argument('--output', type=str, help='Specify output name for training data')
    parser.add_argument('--agent', type=str, help='Filter by specific agent name')
    parser.add_argument('--tool', type=str, help='Filter by specific tool name')
    args = parser.parse_args()

    asyncio.run(process_training_data(args.output, args.agent, args.tool))

if __name__ == "__main__":
    main() 