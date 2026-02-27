"""
AWS Bedrock Agent with Tool Use — Gemma 3 27B
Instrumented with Langfuse v3
"""

import json
import os
import boto3
from botocore.exceptions import ClientError
from langfuse import get_client, observe, Langfuse
from dotenv import load_dotenv
import os

load_dotenv()

# ── Langfuse setup ────────────────────────────────────────────────────────────
# Set via env vars:
#   LANGFUSE_PUBLIC_KEY=pk-lf-...
#   LANGFUSE_SECRET_KEY=sk-lf-...
#   LANGFUSE_BASE_URL=https://cloud.langfuse.com  (or US region)

Langfuse(
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
)

langfuse = get_client()

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_ID = "google.gemma-3-27b-it"
REGION   = "ap-south-1"

# Use this exact string in the Langfuse UI under Project Settings > Models
# so it can match and calculate cost automatically.
LANGFUSE_MODEL_NAME = "google.gemma-3-27b-it"

TOOL_CONFIG = {
    "tools": [
        {
            "toolSpec": {
                "name": "add_numbers",
                "description": "Adds two numbers together and returns the result.",
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "a": {"type": "number", "description": "First number."},
                            "b": {"type": "number", "description": "Second number."},
                        },
                        "required": ["a", "b"],
                    }
                },
            }
        }
    ]
}


def add_numbers(a: float, b: float) -> float:
    return a + b


def execute_tool(tool_name: str, tool_input: dict) -> str:
    if tool_name == "add_numbers":
        result = add_numbers(tool_input["a"], tool_input["b"])
        return json.dumps({"result": result})
    raise ValueError(f"Unknown tool: {tool_name}")


@observe(name="bedrock-agent-run")  # Root span — wraps entire run as one trace
def run_agent(user_message: str) -> str:
    client   = boto3.client("bedrock-runtime", region_name=REGION)
    messages = [{"role": "user", "content": [{"text": user_message}]}]

    # Track total tokens across the agentic loop for cost rollup on the trace
    total_input_tokens  = 0
    total_output_tokens = 0
    turn = 0

    while True:
        turn += 1
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

        # ── Extract token usage from Bedrock response ──────────────────────
        usage = response.get("usage", {})
        input_tokens  = usage.get("inputTokens", 0)
        output_tokens = usage.get("outputTokens", 0)
        total_input_tokens  += input_tokens
        total_output_tokens += output_tokens

        # ── Log this LLM call as a Langfuse generation ────────────────────
        with langfuse.start_as_current_observation(
            as_type="generation",
            name=f"bedrock-converse-turn-{turn}",
            model=LANGFUSE_MODEL_NAME,
            input=messages,                          # full message history sent
            output=output_message,
        ) as generation:
            generation.update(
                usage_details={
                    "input":  input_tokens,
                    "output": output_tokens,
                },
                metadata={
                    "stop_reason": stop_reason,
                    "turn": turn,
                },
            )

        messages.append(output_message)

        # ── Tool use turn ──────────────────────────────────────────────────
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

                # ── Log the tool call as a Langfuse span ──────────────────
                with langfuse.start_as_current_observation(
                    as_type="tool",
                    name=tool_name,
                    input=tool_input,
                ) as tool_span:
                    try:
                        result_str = execute_tool(tool_name, tool_input)
                        tool_span.update(output=json.loads(result_str))
                        tool_results.append({
                            "toolResult": {
                                "toolUseId": tool_use_id,
                                "content":   [{"text": result_str}],
                                "status":    "success",
                            }
                        })
                    except Exception as e:
                        tool_span.update(
                            output={"error": str(e)},
                            level="ERROR",
                        )
                        tool_results.append({
                            "toolResult": {
                                "toolUseId": tool_use_id,
                                "content":   [{"text": str(e)}],
                                "status":    "error",
                            }
                        })

            messages.append({"role": "user", "content": tool_results})

        # ── Final answer ───────────────────────────────────────────────────
        else:
            final_text = " ".join(
                block.get("text", "")
                for block in output_message["content"]
                if "text" in block
            ).strip()

            # Update the root trace with aggregate token counts
            langfuse.update_current_trace(
                output=final_text,
                metadata={
                    "total_input_tokens":  total_input_tokens,
                    "total_output_tokens": total_output_tokens,
                    "total_turns": turn,
                },
            )

            langfuse.flush()  # Ensure everything ships before returning
            return final_text


if __name__ == "__main__":
    test_queries = [
        "What is 42 + 58?",
        "Can you add 123.5 and 876.5 for me?",
        "Hello! How are you?",
    ]

    for query in test_queries:
        print(f"User : {query}")
        answer = run_agent(query)
        print(f"Agent: {answer}\n")