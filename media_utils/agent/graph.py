"""LangGraph agent for media generation and OCR."""

from typing import Annotated, Optional, Literal, TypedDict, Sequence
import operator

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from media_utils.agent.tools import create_tools


class AgentState(TypedDict):
    """State for the media agent."""
    messages: Annotated[Sequence[BaseMessage], operator.add]


class MediaAgent:
    """LangGraph agent for media generation and OCR.

    This agent can:
    - Generate images from text prompts
    - Extract text from images using OCR

    Example:
        >>> from media_utils.agent import MediaAgent
        >>> agent = MediaAgent()
        >>> result = agent.run("Generate an image of a sunset over mountains")
        >>> print(result)

        >>> # With OCR
        >>> result = agent.run("Extract text from document.png")
        >>> print(result)
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        output_dir: str = "output",
        ocr_quantization: str = "4bit",
        device: str = "cuda",
    ):
        """Initialize the media agent.

        Args:
            model_name: LLM model to use for reasoning.
            output_dir: Directory for generated images.
            ocr_quantization: Quantization for OCR ("4bit", "8bit", None).
            device: Device for LLM ("cuda" or "cpu").
        """
        self.model_name = model_name
        self.device = device
        self.output_dir = output_dir
        self.ocr_quantization = ocr_quantization

        # Tools
        self.tools = create_tools(
            output_dir=output_dir,
            ocr_quantization=ocr_quantization,
        )
        self.tool_map = {tool.name: tool for tool in self.tools}

        # Build the graph
        self._llm = None
        self._graph = None

    def _get_llm(self):
        """Lazy-load the LLM with tool binding."""
        if self._llm is None:
            from media_utils.llm.qwen import QwenLLM

            # Create a wrapper that formats for tool calling
            self._qwen = QwenLLM(
                model_name=self.model_name,
                device=self.device,
            )
        return self._qwen

    def _build_graph(self):
        """Build the LangGraph workflow."""
        if self._graph is not None:
            return self._graph

        # Create the graph
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("agent", self._agent_node)
        workflow.add_node("tools", self._tool_node)

        # Set entry point
        workflow.set_entry_point("agent")

        # Add edges
        workflow.add_conditional_edges(
            "agent",
            self._should_continue,
            {
                "continue": "tools",
                "end": END,
            }
        )
        workflow.add_edge("tools", "agent")

        self._graph = workflow.compile()
        return self._graph

    def _agent_node(self, state: AgentState) -> dict:
        """Agent node that decides what to do."""
        messages = state["messages"]
        llm = self._get_llm()

        # Format messages for the LLM
        chat_messages = []
        system_prompt = self._get_system_prompt()
        chat_messages.append({"role": "system", "content": system_prompt})

        for msg in messages:
            if isinstance(msg, HumanMessage):
                chat_messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                chat_messages.append({"role": "assistant", "content": msg.content})
            elif isinstance(msg, ToolMessage):
                chat_messages.append({"role": "user", "content": f"Tool result: {msg.content}"})

        # Get LLM response
        response = llm.chat(chat_messages)

        # Parse for tool calls
        tool_call = self._parse_tool_call(response)

        if tool_call:
            # Return AI message with tool call info
            return {
                "messages": [AIMessage(content=response, additional_kwargs={"tool_call": tool_call})]
            }
        else:
            return {"messages": [AIMessage(content=response)]}

    def _tool_node(self, state: AgentState) -> dict:
        """Execute the tool and return results."""
        messages = state["messages"]
        last_message = messages[-1]

        tool_call = last_message.additional_kwargs.get("tool_call")
        if not tool_call:
            return {"messages": [ToolMessage(content="No tool call found", tool_call_id="error")]}

        tool_name = tool_call.get("name")
        tool_args = tool_call.get("arguments", {})

        if tool_name not in self.tool_map:
            return {"messages": [ToolMessage(
                content=f"Unknown tool: {tool_name}",
                tool_call_id=tool_name
            )]}

        # Execute the tool
        tool = self.tool_map[tool_name]
        try:
            result = tool._run(**tool_args)
        except Exception as e:
            result = f"Error executing tool: {str(e)}"

        return {"messages": [ToolMessage(content=result, tool_call_id=tool_name)]}

    def _should_continue(self, state: AgentState) -> Literal["continue", "end"]:
        """Determine if we should continue to tools or end."""
        messages = state["messages"]
        last_message = messages[-1]

        # Check if there's a tool call
        if hasattr(last_message, "additional_kwargs"):
            tool_call = last_message.additional_kwargs.get("tool_call")
            if tool_call:
                return "continue"

        return "end"

    def _get_system_prompt(self) -> str:
        """Get the system prompt for the agent."""
        return """You are a helpful AI assistant with access to media tools.

Available tools:
1. generate_image - Generate images from text descriptions
   Arguments:
   - prompt (required): Text description of the image
   - negative_prompt (optional): What to avoid
   - resolution (optional): "1024x1024", "1344x768", "768x1344"
   - seed (optional): Random seed for reproducibility

2. extract_text - Extract text from images using OCR
   Arguments:
   - image_path (required): Path to the image file
   - mode (optional): "markdown", "free", "figure", "describe"

When you need to use a tool, respond with EXACTLY this format:
TOOL_CALL: tool_name
ARGUMENTS:
  argument_name: value
  another_arg: value

When you don't need a tool, just respond normally.

Examples:
- User: "Create an image of a cat on a beach"
  Response: TOOL_CALL: generate_image
  ARGUMENTS:
    prompt: A cat relaxing on a sunny beach with waves in the background
    resolution: 1024x1024

- User: "What does this document say?" (with image path)
  Response: TOOL_CALL: extract_text
  ARGUMENTS:
    image_path: document.png
    mode: markdown
"""

    def _parse_tool_call(self, response: str) -> Optional[dict]:
        """Parse a tool call from the LLM response."""
        if "TOOL_CALL:" not in response:
            return None

        try:
            lines = response.strip().split("\n")
            tool_name = None
            arguments = {}
            in_arguments = False

            for line in lines:
                line = line.strip()
                if line.startswith("TOOL_CALL:"):
                    tool_name = line.replace("TOOL_CALL:", "").strip()
                elif line == "ARGUMENTS:":
                    in_arguments = True
                elif in_arguments and ":" in line:
                    key, value = line.split(":", 1)
                    key = key.strip()
                    value = value.strip()
                    # Handle optional values
                    if value.lower() not in ["none", "null", ""]:
                        # Try to parse as int
                        try:
                            value = int(value)
                        except ValueError:
                            pass
                        arguments[key] = value

            if tool_name:
                return {"name": tool_name, "arguments": arguments}

        except Exception:
            pass

        return None

    def run(self, query: str) -> str:
        """Run the agent with a query.

        Args:
            query: User query (e.g., "Generate an image of a sunset")

        Returns:
            Agent response string.
        """
        graph = self._build_graph()

        # Run the graph
        initial_state = {"messages": [HumanMessage(content=query)]}
        result = graph.invoke(initial_state)

        # Get the final response
        messages = result["messages"]
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and not msg.additional_kwargs.get("tool_call"):
                return msg.content
            elif isinstance(msg, ToolMessage):
                return msg.content

        return "No response generated"

    def chat(self, messages: list[dict]) -> str:
        """Run the agent with a chat history.

        Args:
            messages: List of {"role": str, "content": str} dicts.

        Returns:
            Agent response string.
        """
        # Convert to LangChain messages
        lc_messages = []
        for msg in messages:
            if msg["role"] == "user":
                lc_messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                lc_messages.append(AIMessage(content=msg["content"]))

        graph = self._build_graph()
        result = graph.invoke({"messages": lc_messages})

        # Get the final response
        messages = result["messages"]
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and not msg.additional_kwargs.get("tool_call"):
                return msg.content
            elif isinstance(msg, ToolMessage):
                return msg.content

        return "No response generated"

    def unload(self):
        """Unload all models to free GPU memory."""
        for tool in self.tools:
            if hasattr(tool, "unload"):
                tool.unload()

        if self._qwen is not None:
            self._qwen.unload()
            self._qwen = None

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - unload all models."""
        self.unload()
        return False
