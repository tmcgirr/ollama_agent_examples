import asyncio
from ollama import ChatResponse, AsyncClient
import ollama
from pydantic import BaseModel, ValidationError, validator
from typing import Any, List, Optional, Type, Literal
from json.decoder import JSONDecodeError
import json

# Ollama (https://www.llama.com/docs/model-cards-and-prompt-formats/llama3_2/)
# 3.2 model -> "support custom functions defined in either the system prompt or user prompt
# MODEL = 'llama3.2:1b'
MODEL = 'llama3.2:3b'
MAX_TOOL_RETRIES = 2 

# Data Schema (Expected output from tool) (Add explicit tool names for strict validation)
class NumberOperation(BaseModel):
    result: int
    operation: Literal['add_two_numbers', 'subtract_two_numbers', 'multiply_two_numbers', 'divide_two_numbers']
    numbers_used: list[int]

# Math Agent
class MathAgent:
    def __init__(self, model: str = MODEL):
        self.model = model
        self.client = ollama.AsyncClient()
        
        # Define tools map as part of agent initialization
        self.tools_map = {
            'add_two_numbers': self.add_two_numbers,
            'subtract_two_numbers': self.subtract_two_numbers,
            'multiply_two_numbers': self.multiply_two_numbers,
            'divide_two_numbers': self.divide_two_numbers
        }
        
        self.messages = []
        self.available_tools = [
            {
                "type": "function",
                "function": {
                    "name": "add_two_numbers",
                    "description": "Add two numbers, returning a NumberOperation.",
                    "parameters": {
                        "type": "object",
                        "required": ["a", "b"],
                        "properties": {
                            "a": {"type": "number"},
                            "b": {"type": "number"}
                        }
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "subtract_two_numbers",
                    "description": "Subtract two numbers, returning a NumberOperation.",
                    "parameters": {
                        "type": "object",
                        "required": ["a", "b"],
                        "properties": {
                            "a": {"type": "number"},
                            "b": {"type": "number"}
                        }
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "multiply_two_numbers",
                    "description": "Multiply two numbers, returning a NumberOperation.",
                    "parameters": {
                        "type": "object",
                        "required": ["a", "b"],
                        "properties": {
                            "a": {"type": "number"},
                            "b": {"type": "number"}
                        }
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "divide_two_numbers",
                    "description": "Divide two numbers, returning a NumberOperation.",
                    "parameters": {
                        "type": "object",
                        "required": ["a", "b"],
                        "properties": {
                            "a": {"type": "number"},
                            "b": {"type": "number"}
                        }
                    }
                }
            }
        ]
        
        # Validation Schema Map (Pydantic Model for Each Tool)
        self.validation_schema_map = {
            'add_two_numbers': NumberOperation,
            'subtract_two_numbers': NumberOperation,
            'multiply_two_numbers': NumberOperation,
            'divide_two_numbers': NumberOperation
        }
        
        # System Prompt (Dynaminc Tool Descriptions)
        self.system_prompt = f'''
            <system>
                You are an expert in mathematical operations that uses tools for calculations.
                
                <tools>
                {self.available_tools}
                </tools>
                
                <rules>
                1. ONLY call a tool when you have all required parameters
                2. If operation not available, respond: "Sorry, I can only perform addition, subtraction, multiplication, and division."
                3. If parameters missing, ask for them
                4. For tool responses, extract ONLY the number from "result=X" for the final output
                </rules>
                
                <format>
                    <input>what is 5 plus 3?</input>
                    <tool_call>add_two_numbers(a=5, b=3)</tool_call>
                    <tool_response>result=8 operation='add_two_numbers' numbers_used=[5, 3]</tool_response>
                    <output>The result of 5 plus 3 is 8</output>
                </format>
                
                <instructions>
                DO NOT explain calculations.
                DO NOT make new tool calls after receiving a response.
                ONLY return the number from result=X. 
                DO NOT include the operation name or numbers used in the final output (e.g. "The result of 5 plus 3 is 8")
                </instructions>
                
                <example>
                User: what is 5 plus 3?
                Assistant: The result of 5 plus 3 is 8.
                </example>
                
                <example>
                User: what is 55 minus 3?
                Assistant: The result of 55 minus 3 is 52.
                </example>
                
                <example>
                User: what is 8 times 12?
                Assistant: The result of 8 times 12 is 96.
                </example>
                
                <example>
                User: what is 72 divided by 9?
                Assistant: The result of 72 divided by 9 is 8.
                </example>
                             
            </system>
            '''
        
    # Tool Function 
    async def add_two_numbers(self, a: Any, b: Any) -> NumberOperation:
        """
        Add two numbers, but accept raw input (possibly string)
        and return a validated NumberOperation object.
        
        Args:
            a (Any): The first number
            b (Any): The second number
            
        Returns:
            NumberOperation: A validated NumberOperation object
        """
        # Return an invalid structure to test the retry mechanism
        # return {
        #     'result': "not a number",  # This should fail validation since result should be int
        #     'operation': 'add_two_numbers',
        #     'numbers_used': [1, 2]
        # }
        return NumberOperation(result=int(a) + int(b), operation="add_two_numbers", numbers_used=[int(a), int(b)])
    
    # Tool Function 
    async def subtract_two_numbers(self, a: Any, b: Any) -> NumberOperation:
            """
            Subtract two numbers, returning a NumberOperation.
            
            Args:
                a (Any): The first number
                b (Any): The second number
                
            Returns:
                NumberOperation: A validated NumberOperation object
            """
            return NumberOperation(result=int(a) - int(b), operation="subtract_two_numbers", numbers_used=[int(a), int(b)])
    
    async def multiply_two_numbers(self, a: Any, b: Any) -> NumberOperation:
        """
        Multiply two numbers, returning a NumberOperation.
        
        Args:
            a (Any): The first number
            b (Any): The second number
            
        Returns:
            NumberOperation: A validated NumberOperation object
        """
        return NumberOperation(result=int(a) * int(b), operation="multiply_two_numbers", numbers_used=[int(a), int(b)])
    
    async def divide_two_numbers(self, a: Any, b: Any) -> NumberOperation:
        """
        Divide two numbers, returning a NumberOperation.
        
        Args:
            a (Any): The first number
            b (Any): The second number
            
        Returns:
            NumberOperation: A validated NumberOperation object
        """
        return NumberOperation(result=int(a) / int(b), operation="divide_two_numbers", numbers_used=[int(a), int(b)])

# Model Retry (Custom Retry Handler)
class ModelRetryHandler:
    def __init__(
        self,
        client: AsyncClient,
        model: str,
        available_tools: List[Any],
        output_schema: Type[BaseModel],
        max_format_retries: int = 3,
        max_tool_retries: int = 2
    ):
        self.client = client
        self.model = model
        self.available_tools = available_tools
        self.output_schema = output_schema
        self.max_format_retries = max_format_retries
        self.max_tool_retries = max_tool_retries
        # Update tools_map to handle the tool definitions format
        self.tools_map = {
            tool['function']['name']: tool 
            for tool in available_tools 
            if isinstance(tool, dict) and 'function' in tool
        }

    async def _attempt_format_retry(
        self,
        raw_output: Any,
        expected_model: Type[BaseModel]
    ) -> Optional[BaseModel]:
        """Attempts to reformat invalid output using the format parameter"""
        for attempt in range(self.max_format_retries):
            try:
                print(f'Format attempt {attempt + 1}/{self.max_format_retries}')
                format_messages = [
                    {
                        'role': 'system',
                        'content': 'You are a data formatter. Your only job is to convert the exact values from the input into the required schema format.'
                    },
                    {
                        'role': 'user',
                        'content': f'Convert this exact data into JSON matching this schema (preserve all values, just fix types): \n\n{expected_model.model_json_schema()}\n\nData to format: \n\n{raw_output}'
                    }
                ]
                
                # Send the formatted request to the model with the schema (Structured Output)
                retry_response = await self.client.chat(
                    self.model,
                    messages=format_messages,
                    format=expected_model.model_json_schema(),
                    options={'temperature': 0}
                )
                
                # Validate the response against the schema
                return expected_model.model_validate_json(retry_response.message.content)
                
            except ValidationError as e:
                print(f'Format attempt {attempt + 1} failed: {e}')
                if attempt == self.max_format_retries - 1:
                    return None
        return None

    # Tool Retry (Custom Retry Handler) (If simple formatting fails)
    async def _attempt_tool_retry(
        self,
        messages: List[dict],
        function_to_call: Any
    ) -> Optional[BaseModel]:
        """Attempts to retry the entire tool call"""
        for attempt in range(self.max_tool_retries):
            try:
                print(f'Tool retry attempt {attempt + 1}/{self.max_tool_retries}')
                # Only use the system prompt and user query
                retry_messages = [
                    msg for msg in messages 
                    if msg['role'] in ('system', 'user')
                ][:2]
                
                tool_retry_response = await self.client.chat(
                    self.model,
                    messages=retry_messages,
                    tools=self.available_tools,
                    options={'temperature': 0}
                )
                
                if tool_retry_response.message.tool_calls:
                    retry_tool = tool_retry_response.message.tool_calls[0]
                    args = retry_tool.function.arguments
                    if isinstance(args, str):
                        args = json.loads(args)
                    retry_output = await function_to_call(**args)
                    return self.output_schema.model_validate(retry_output)
                    
            except (ValidationError, JSONDecodeError, Exception) as e:
                print(f'Tool retry attempt {attempt + 1} failed: {e}')
                if attempt == self.max_tool_retries - 1:
                    raise Exception(f"All retry attempts failed: {str(e)}")
        return None

    # Main Method (Execute Tool Call with Automatic Retries)
    async def execute_tool_with_retry(
        self,
        messages: List[dict],
        tool_call: Any,
        function_to_call: Any
    ) -> tuple[BaseModel, List[dict]]:
        """Main method to execute a tool call with automatic retries"""
        try:
            # Await the function call
            raw_output = await function_to_call(**tool_call.function.arguments)
            validated_output = self.output_schema.model_validate(raw_output)
            
            # Properly format the messages
            messages.extend([
                {
                    'role': 'assistant',
                    'content': None,
                    'tool_calls': [tool_call]
                },
                {
                    'role': 'tool',
                    'content': str(validated_output),
                    'name': tool_call.function.name
                }
            ])
            
            return validated_output, messages
            
        except ValidationError:
            print('Validation error, attempting to reformat tool output')
            
            # Try format retry
            if validated_output := await self._attempt_format_retry(raw_output, self.output_schema):
                messages.extend([
                    {
                        'role': 'assistant',
                        'content': None,
                        'tool_calls': [tool_call]
                    },
                    {
                        'role': 'tool',
                        'content': str(validated_output),
                        'name': tool_call.function.name
                    }
                ])
                return validated_output, messages
            
            # If format retry fails, try tool retry
            print('All formatting attempts failed, trying full tool retry')
            if validated_output := await self._attempt_tool_retry(messages, function_to_call):
                messages.extend([
                    {
                        'role': 'assistant',
                        'content': None,
                        'tool_calls': [tool_call]
                    },
                    {
                        'role': 'tool',
                        'content': str(validated_output),
                        'name': tool_call.function.name
                    }
                ])
                return validated_output, messages
            
            raise Exception("All retry attempts failed")


async def run_agent_with_tools(
    client: AsyncClient,
    model: str,
    messages: List[dict],
    tools: List[dict],
    output_schema: Type[BaseModel],
    agent_instance: MathAgent,
    stream: bool = True, 
    max_tool_retries: int = MAX_TOOL_RETRIES
) -> str:
    """
    Run an agent with tools and handle retries using ModelRetryHandler.
    """
    # Initialize the retry handler
    retry_handler = ModelRetryHandler(
        client=client,
        model=model,
        available_tools=tools,
        output_schema=output_schema,
        max_tool_retries=max_tool_retries
    )
    
    response = await client.chat(
        model=model,
        messages=messages,
        tools=tools,
        options={'temperature': 0}
    )

    if response.message.tool_calls:
        for tool_call in response.message.tool_calls:
            function_name = tool_call.function.name
            function_to_call = agent_instance.tools_map.get(function_name)
            
            if function_to_call:
                print(f'\n[TOOL] -> Using [{function_name}] with arguments: [{tool_call.function.arguments}]')
                
                try:
                    # Use the retry handler to execute the tool call
                    validated_output, updated_messages = await retry_handler.execute_tool_with_retry(
                        messages=messages,
                        tool_call=tool_call,
                        function_to_call=function_to_call
                    )
                    
                    # Update messages with the retry handler's updated message history
                    messages = updated_messages
                    
                except Exception as e:
                    print(f"\nAll retry attempts failed: {str(e)}")
                    raise
        
        # Stream the final response (LLM tool call interpretation)
        response_text = ""
        if stream:
            stream_response = await client.chat(
                model=model,
                messages=messages,
                tools=tools,
                options={'temperature': 0},
                stream=True
            )
            
            print("\nAGENT: ", end='', flush=True)
            async for part in stream_response:
                if part.message.content:
                    print(part.message.content, end='', flush=True)
                    response_text += part.message.content
            print()
        else:
            final_response = await client.chat(
                model=model,
                messages=messages,
                tools=tools,
                options={'temperature': 0}
            )
            response_text = final_response.message.content
            print(f"\nAGENT: {response_text}")
        
        return response_text

    return response.message.content


# Main 
async def main():
    # Initialize Math Agent
    agent = MathAgent()
    
    # Print Welcome Message
    print("\n🦙 Ollama Math Agent! (type 'exit' to quit)")
    print("----------------------------------------")
    
    while True:
        try:
            # Get User Input
            user_input = input("\nYou: ").strip()
            
            # Exit Condition
            if user_input.lower() in ['exit', 'quit']:
                print("\nGoodbye!")
                break
            
            # Skip Empty Input
            if not user_input:
                continue
            
            # Print Thinking Message
            print("\nAGENT: Thinking...")
            
            # Create Message History Storage
            messages = [
                {'role': 'system', 'content': agent.system_prompt},
                {'role': 'user', 'content': user_input}
            ]

            # Run Agent with Tools
            response = await run_agent_with_tools(
                client=agent.client,
                model=agent.model,
                messages=messages,
                tools=agent.available_tools,
                output_schema=NumberOperation,
                agent_instance=agent,
                stream=True 
            )
            
            # Print Response (if not streamed)
            if not response.strip():
                print(f"\nAGENT: {response}")
            
        except ValidationError as e:
            # Print Error Message
            print("\nI apologize, but I encountered an error processing your calculation.")
            print("This might be due to an internal problem with my math tools.")
            print("Please try your question again, or try a different calculation.")
            
        except Exception as e:
            # Print Unexpected Error Message
            print(f"\nAn unexpected error occurred: {str(e)}")
            print("Please try again with a different question.")

if __name__ == "__main__":
    asyncio.run(main())
