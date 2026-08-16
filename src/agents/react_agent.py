"""
AutoDrive RAG v2.0 — ReAct Agent
Implements a Reason-Act-Observe loop that enables the LLM to
autonomously decide which tools to call (local DB, web search,
image search) based on the user's query.

Architecture:
  1. LLM receives the user query + system prompt
  2. LLM outputs a structured JSON tool call (or a final answer)
  3. Agent executes the tool and feeds the result back to the LLM
  4. LLM reasons over the tool output and decides: call another tool or answer
  5. Repeat until the LLM produces a final answer (max 5 iterations)
"""

from __future__ import annotations

import json
import logging
import re
from typing import AsyncGenerator, Optional

logger = logging.getLogger("chatbot.agents.react")

# Maximum number of tool-calling iterations before forcing a final answer
MAX_ITERATIONS = 5

REACT_SYSTEM_PROMPT = """\
You are **AutoDrive AI**, a professional automotive sales consultant for AutoDrive — \
India's trusted pre-owned car marketplace.

You have access to the following tools:

{tool_descriptions}

## How to Use Tools
When you need information, respond with a JSON tool call in this EXACT format:
```tool_call
{{"tool": "<tool_name>", "args": {{"<param>": "<value>"}}}}
```

You may call MULTIPLE tools one at a time. After each tool call, you will receive the \
tool's output. Reason over it, then either call another tool or provide your final answer.

## Rules
1. **Always check local availability FIRST** before searching the web.
2. If the user asks to SEE a car or wants a picture, use `fetch_car_image`.
3. If the user asks for specs/features NOT in our database (BHP, safety rating, etc.), \
use `web_search_car_specs`.
4. **Currency:** Always use Indian Rupees (₹). Express prices as "₹X lakh".
5. **Be honest:** If a car isn't available locally, say so clearly.
6. When mentioning a car, include its ID: `[CAR_ID:X]`.
7. When your answer is complete, just write it normally without any tool_call block.
8. If you include images, use markdown: `![description](url)`

## Current Inventory Context
{context}
"""


class ReActAgent:
    """
    ReAct (Reason + Act) Agent that orchestrates tool calling
    for the AutoDrive chatbot.

    The agent takes a user query, decides which tools to call,
    executes them, and synthesizes a final response.
    """

    def __init__(self, llm, tools, inventory_context: str = "") -> None:
        """
        Args:
            llm: LangChain LLM instance (ChatGroq, ChatOpenAI, etc.)
            tools: AgentTools instance with all tool methods.
            inventory_context: Pre-built inventory context string.
        """
        self.llm = llm
        self.tools = tools
        self.inventory_context = inventory_context

    def _build_system_prompt(self) -> str:
        """Build the system prompt with tool descriptions."""
        tool_defs = self.tools.get_tool_definitions()
        desc_lines = []
        for t in tool_defs:
            params = ", ".join(
                f"{k}: {v}" for k, v in t.get("parameters", {}).items()
            )
            desc_lines.append(
                f"- **{t['name']}**({params}): {t['description']}"
            )
        tool_descriptions = "\n".join(desc_lines)

        return REACT_SYSTEM_PROMPT.format(
            tool_descriptions=tool_descriptions,
            context=self.inventory_context[:3000],  # Cap context size
        )

    def _extract_tool_call(self, text: str) -> Optional[dict]:
        """
        Extract a tool call JSON from the LLM's response.

        Looks for the pattern:
        ```tool_call
        {"tool": "...", "args": {...}}
        ```

        Returns:
            Parsed dict with 'tool' and 'args', or None if no tool call.
        """
        # Try to find ```tool_call ... ``` blocks
        pattern = r"```tool_call\s*\n?(.*?)\n?```"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1).strip())
            except json.JSONDecodeError:
                logger.warning("Failed to parse tool call JSON: %s", match.group(1))
                return None

        # Fallback: try to find raw JSON with "tool" key
        json_pattern = r'\{"tool"\s*:\s*"[^"]+"\s*,\s*"args"\s*:\s*\{[^}]*\}\}'
        match = re.search(json_pattern, text)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                return None

        return None

    def _execute_tool(self, tool_call: dict) -> str:
        """Execute a tool call and return the result."""
        tool_name = tool_call.get("tool", "")
        args = tool_call.get("args", {})

        logger.info("Agent calling tool: %s(%s)", tool_name, args)
        result = self.tools.execute_tool(tool_name, **args)
        logger.info("Tool %s returned %d chars", tool_name, len(result))
        return result

    async def run(
        self,
        user_msg: str,
        chat_history: list,
    ) -> AsyncGenerator[str, None]:
        """
        Run the ReAct loop and stream the final response.

        Yields:
            Tokens of the final response as they are generated.
        """
        system_prompt = self._build_system_prompt()

        # Build the conversation for the LLM
        messages = [
            {"role": "system", "content": system_prompt},
        ]
        # Add chat history
        for msg in chat_history[-10:]:  # Last 10 messages
            if hasattr(msg, "type"):
                role = "user" if msg.type == "human" else "assistant"
                messages.append({"role": role, "content": msg.content})

        messages.append({"role": "user", "content": user_msg})

        # ReAct loop
        tool_results = []
        for iteration in range(MAX_ITERATIONS):
            # Call the LLM (non-streaming for tool detection)
            response = await self.llm.ainvoke(messages)
            response_text = response.content if hasattr(response, "content") else str(response)

            # Check for tool calls
            tool_call = self._extract_tool_call(response_text)

            if tool_call is None:
                # No tool call — this is the final answer, stream it
                # Since we already have the full text, yield it in chunks
                # to simulate streaming
                for i in range(0, len(response_text), 4):
                    yield response_text[i:i + 4]
                return

            # Extract any reasoning text before the tool call
            tool_call_start = response_text.find("```tool_call")
            if tool_call_start == -1:
                tool_call_start = response_text.find('{"tool"')
            reasoning = response_text[:tool_call_start].strip()

            if reasoning:
                # Yield the reasoning as streaming tokens (the "thinking" part)
                yield f"🔍 *{reasoning}*\n\n"

            # Execute the tool
            tool_name = tool_call.get("tool", "unknown")
            yield f"⚙️ *Calling `{tool_name}`...*\n\n"

            result = self._execute_tool(tool_call)
            tool_results.append({
                "tool": tool_name,
                "result": result[:2000],  # Cap tool output
            })

            # Feed tool result back to LLM
            messages.append({
                "role": "assistant",
                "content": response_text,
            })
            messages.append({
                "role": "user",
                "content": (
                    f"Tool `{tool_name}` returned:\n"
                    f"```\n{result[:2000]}\n```\n\n"
                    "Now reason over this result. Either call another tool "
                    "or provide your final answer to the user."
                ),
            })

            logger.info(
                "ReAct iteration %d: called %s, got %d chars",
                iteration + 1, tool_name, len(result),
            )

        # If we hit max iterations, force a final answer
        messages.append({
            "role": "user",
            "content": (
                "You have used all available tool calls. "
                "Please provide your final answer NOW based on "
                "all the information gathered so far."
            ),
        })

        response = await self.llm.ainvoke(messages)
        final = response.content if hasattr(response, "content") else str(response)
        for i in range(0, len(final), 4):
            yield final[i:i + 4]


def should_use_agent(query: str) -> bool:
    """
    Determine if the query should be routed to the ReAct agent
    instead of the standard RAG pipeline.

    Returns True if the query likely needs tool calling (images,
    web specs, availability checks, etc.)
    """
    query_lower = query.lower()

    # Strong triggers — always route to agent
    strong_triggers = [
        # Image requests
        "show me", "picture", "photo", "image", "look like",
        "looks like", "what does", "how does it look",
        # Web spec requests
        "top speed", "bhp", "horsepower", "safety rating",
        "ncap", "mileage on highway", "ground clearance",
        "boot space", "specifications", "spec",
        # Comparison with external data
        "compare with", "versus", "better than",
    ]

    if any(trigger in query_lower for trigger in strong_triggers):
        return True

    # Weak triggers — only route if query seems specific (not generic)
    # "Do you have a Creta?" → agent (specific car)
    # "What SUVs do you have?" → standard RAG (generic browsing)
    weak_triggers = ["do you have", "is available", "in stock", "availability"]
    generic_words = ["suv", "sedan", "hatchback", "car", "cars", "under", "below", "cheap", "budget"]

    if any(trigger in query_lower for trigger in weak_triggers):
        # If the query also has generic browsing words, skip the agent
        if any(g in query_lower for g in generic_words):
            return False
        return True

    return False
