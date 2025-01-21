# Ollama Agent Examples

This repository showcases two primary examples of using [Ollama]([https://github.com/jmorganca/ollama](https://ollama.com/library/llama3.2)) for building function-calling chat agents in Python. The first is a **Single Agent** example (a basic math bot), and the second is a **Multi-Agent System** where an “Operator Agent” delegates tasks to specialized math agents (addition, subtraction, multiplication, division). 

---

## Table of Contents

1. [Single Agent](#single-agent)
   - [Key Features](#key-features)
   - [Design](#design)
2. [Description](#description)
3. [Multi-Agent System Example](#multi-agent-system-example)
   - [Overview](#overview)
   - [How It Works](#how-it-works)
   - [Conversation History](#conversation-history)
4. [Usage](#usage)
5. [User & History Management](#user--history-management)
6. [Cleaning Script and Training Data Generation](#cleaning-script-and-training-data-generation)

---

## Single Agent

Single Agent Interaction

![single_agent](https://github.com/user-attachments/assets/4d15af1c-7f59-430a-8a35-7e9bc627e82d)


Single Agent (With Auto Retry for failed Pydantic Output Validation)

![single_agent_retry](https://github.com/user-attachments/assets/ae9138ae-f477-4be8-966a-bf3703f20487)


### Key Features
- **MathAgent class** with built-in tools for:
  - Addition
  - Subtraction
  - Multiplication
  - Division
- **Automatic validation** of tool outputs using [Pydantic](https://docs.pydantic.dev).
- **Retry Mechanism**: If the model or the function call returns invalid JSON or an invalid schema, the agent can retry calling the tool.

### Design

1. **System Prompt**  
   - Provides the agent with instructions on how to respond and which tools are available.

2. **Tools (Functions)**  
   - The agent can call any of the following functions:
     - `add_two_numbers(a, b)`
     - `subtract_two_numbers(a, b)`
     - `multiply_two_numbers(a, b)`
     - `divide_two_numbers(a, b)`

3. **Messages Flow**  
   - A user message is added to the conversation (e.g., “what is 5 plus 3?”).  
   - Ollama is queried with the messages and the tool definitions.  
   - If the agent decides to call a function, it makes a tool call.  
   - The function result is returned, validated against the Pydantic schema, and appended to the conversation as a tool response.  
   - Finally, the agent’s last response is either streamed or displayed in full, returning the final answer (e.g., “The result of 5 plus 3 is 8.”).


---

## Multi-Agent System Example

General User Query

![mas_general](https://github.com/user-attachments/assets/b32b7164-4e7d-4e3f-adae-36b03ac7465c)


Direct Agent Query

![mas_direct](https://github.com/user-attachments/assets/55a455b7-24e9-4c89-b6b4-b526191a4b70)


Agent Workflow (Sequential Agents)

![mas_workflow](https://github.com/user-attachments/assets/0c630479-6a24-4769-a7fe-b8e0e1654b7b)

### Overview

The multi-agent system expands upon the single-agent concept. It includes:

- An **Operator Agent** whose only job is to decide **which** specialized agent should handle the user’s query.
- Specialized math agents:
  - **AdditionAgent**
  - **SubtractionAgent**
  - **MultiplicationAgent**
  - **DivisionAgent**

These specialized agents each have only one corresponding tool (e.g., `add_two_numbers`) and can only perform that single operation. The operator delegates to them using a shared chat approach and the `delegate_to_agent` tool.

### How It Works

1. **OperatorAgent**  
   - Receives user queries and tries to figure out if the user’s request involves addition, subtraction, multiplication, or division.  
   - Calls `delegate_to_agent(agent_name, query, reason)` with the chosen agent name.

2. **Specialized Agents**  
   - Each has its own system prompt and a single function tool (e.g., `add_two_numbers` for the **AdditionAgent**).  
   - Once delegated a query, the specialized agent processes it, calls its tool, and returns the result.

### Conversation History

- The **global conversation** is maintained by the multi-agent system to track the flow of user prompts and operator decisions.  
- **Each individual agent** has its own memory store that can either be cleared or saved after each agent interaction. This means:
  - The OperatorAgent can be “reset” by clearing its messages, so it forgets past queries.
  - Each specialized agent (AdditionAgent, SubtractionAgent, etc.) can likewise have its own short or long-term memory, depending on your design choices.

---

## Usage

1. **Standard Conversation**  
   - When you simply type a math query (e.g., “what is 8 minus 2?”), the **OperatorAgent** determines which specialized agent to route to (using the `delegate_to_agent` function call under the hood).
   - The specialized agent takes the user query, performs a tool call to get the result, and sends the final answer back to the user.

2. **Directed Agent Usage**  
   - Using the `@` symbol with the agent name or alias, the user can bypass the operator and directly interact with the underlying agent system.  
   - Examples:
     - `@add 5 plus 7`
     - `@subtract 12 from 30`

3. **Workflow / Flows**  
   - You can design a sequential workflow where the user sets a chain of inputs/outputs that interact with the specified agents in order.  
   - In the provided example, the code runs a series of operations by passing the result of one agent to another until the final answer is complete.  
   - For instance, `@flow flow1 10` might run a series of math operations: add 10, subtract 5, multiply by 2, and so on.

---

## User & History Management

1. **Conversation Logging**  
   - By default, the multi-agent system logs conversation data (user prompts, agent responses, and tool calls) in the `data/conversations` directory.  
   - If you do **not** want your conversation recorded, you will need to modify or remove the logging behavior.

2. **Agent Memory**  
   - Each agent (and the operator) can maintain its own message history.  
   - The code can truncate (shorten) or clear memory to avoid long prompts or memory bloat. For example, by default, it might only keep the last `n` messages.  
   - Memory management is handled in the `run_agent_with_tools` function calls, where you can set `agent_memory=True` or `False` and decide whether to keep the conversation context.

3. **Clearing / Resetting**  
   - At any time, you can reset an agent’s memory or the entire system’s memory by modifying the code or by calling an appropriate reset function. This ensures the agent “forgets” previous queries.

4. **User Workflow**  
   - The system is designed to allow freeform text queries, specialized “@agent” commands, and curated “flows.”  
   - Consider which approach best fits your use case—simple Q&A vs. directed agent calls vs. multi-step workflows.

---

## Cleaning Script and Training Data Generation

A separate **cleaning script** is provided (the `TrainingDataProcessor` in the second code snippet) to help you:

1. **Process conversation logs** from `data/conversations`.  
2. **Filter** by agent or tool name if desired.  
3. **Transform** the conversation data into training examples for fine-tuning.

This will create a JSON file in data/training containing only the examples where the “Addition Agent” used the add_two_numbers tool.

Example usage (command-line arguments):

```bash
python training_data_processor.py --output my_dataset --agent "Addition Agent" --tool add_two_numbers
