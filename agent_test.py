"""
AWS Bedrock Agent with Tool Use — Gemma 3 27B
=============================================
Tool: add_numbers — called whenever the user asks to add two numbers.

Prerequisites:
    pip install boto3

AWS credentials must be configured (env vars, ~/.aws/credentials, or IAM role).
Model access for google.gemma-3-27b-it must be enabled in your Bedrock console.
"""

import json
import boto3
from botocore.exceptions import ClientError

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_ID = "google.gemma-3-27b-it"  # Gemma 3 27B Instruct on Bedrock
REGION   = "ap-south-1"              # Change if your model is in another region

# ── Tool definition (JSON Schema) ─────────────────────────────────────────────
TOOL_CONFIG = {
    "tools": [
        {
            "toolSpec": {
                "name": "add_numbers",
                "description": "Adds two numbers together and returns the result. "
                               "Use this tool whenever the user asks to add, sum, or "
                               "find the total of two numbers.",
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "a": {
                                "type": "number",
                                "description": "The first number to add."
                            },
                            "b": {
                                "type": "number",
                                "description": "The second number to add."
                            }
                        },
                        "required": ["a", "b"]
                    }
                }
            }
        }
    ]
}

# ── Tool implementation ────────────────────────────────────────────────────────
def add_numbers(a: float, b: float) -> float:
    """The actual addition logic."""
    return a + b


def execute_tool(tool_name: str, tool_input: dict) -> str:
    """Dispatch a tool call and return the result as a string."""
    if tool_name == "add_numbers":
        result = add_numbers(tool_input["a"], tool_input["b"])
        return json.dumps({"result": result})
    raise ValueError(f"Unknown tool: {tool_name}")


# ── Agentic loop ──────────────────────────────────────────────────────────────
def run_agent(user_message: str) -> str:
    """
    Send a user message to Gemma 3 27B on Bedrock.
    Handles tool calls automatically until the model produces a final answer.
    Returns the final text response.
    """
    client   = boto3.client("bedrock-runtime", region_name=REGION)
    messages = [{"role": "user", "content": [{"text": user_message}]}]

    while True:
        try:
            response = client.converse(
                modelId=MODEL_ID,
                messages=messages,
                toolConfig=TOOL_CONFIG,
                inferenceConfig={"maxTokens": 512, "temperature": 0.1},
            )
        except ClientError as e:
            raise RuntimeError(f"Bedrock API error: {e}") from e

        output_message = response["output"]["message"]
        stop_reason    = response["stopReason"]

        # Add assistant's message to history
        messages.append(output_message)

        # ── Case 1: Model wants to use a tool ─────────────────────────────────
        if stop_reason == "tool_use":
            tool_results = []

            for block in output_message["content"]:
                if block.get("type") != "toolUse" and "toolUse" not in block:
                    continue

                tool_block  = block.get("toolUse", block)
                tool_use_id = tool_block["toolUseId"]
                tool_name   = tool_block["name"]
                tool_input  = tool_block["input"]

                print(f"[Tool call] {tool_name}({tool_input})")

                try:
                    result_str = execute_tool(tool_name, tool_input)
                    tool_results.append({
                        "toolResult": {
                            "toolUseId": tool_use_id,
                            "content":   [{"text": result_str}],
                            "status":    "success",
                        }
                    })
                except Exception as e:
                    tool_results.append({
                        "toolResult": {
                            "toolUseId": tool_use_id,
                            "content":   [{"text": str(e)}],
                            "status":    "error",
                        }
                    })

            # Feed tool results back to the model
            messages.append({"role": "user", "content": tool_results})

        # ── Case 2: Model is done — extract the final text ────────────────────
        else:
            final_text = " ".join(
                block.get("text", "")
                for block in output_message["content"]
                if "text" in block
            ).strip()
            return final_text


# ── Interactive CLI ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Bedrock Agent (Gemma 3 27B) — Addition Tool Demo")
    print("Type 'quit' to exit.\n")

    test_queries = [
        "What is 42 + 58?",
        "Can you add 123.5 and 876.5 for me?",
        "Hello! How are you?",   # No tool needed — just a chat
    ]

    for query in test_queries:
        print(f"User : {query}")
        answer = run_agent(query)
        print(f"Agent: {answer}\n")

    # Interactive mode
    print("─" * 50)
    print("Interactive mode (Ctrl-C or type 'quit' to stop)")
    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break
        if not user_input:
            continue
        response = run_agent(user_input)
        print(f"Agent: {response}")