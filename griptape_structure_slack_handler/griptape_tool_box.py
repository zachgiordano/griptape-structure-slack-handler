import logging
import os

from griptape.tools import BaseTool, GriptapeCloudToolTool, RagTool
from griptape.engines.rag import RagEngine
from griptape.engines.rag.modules import (
    VectorStoreRetrievalRagModule,
    TextChunksResponseRagModule,
)
from griptape.engines.rag.stages import RetrievalRagStage, ResponseRagStage
from griptape.drivers import (
    GriptapeCloudVectorStoreDriver,
)
from griptape.structures import Agent
from griptape.tasks import PromptTask
from griptape.rules import Rule

from .griptape.read_only_conversation_memory import ReadOnlyConversationMemory

logger = logging.getLogger()


def get_tools(message: str, *, dynamic: bool = False) -> list[BaseTool]:
    """
    Gets tools for the Agent to use. if dynamic=True, the LLM will decide what tools to use
    based on the user input and the conversation history.
    """
    tools_dict = _init_tools_dict()
    if not dynamic:
        return [tool for tool, _ in tools_dict.values()]

    tools_descriptions = {k: description for k, (_, description) in tools_dict.items()}

    agent = Agent(
        tasks=[
            PromptTask(
                input="Given the input, what tools are needed to give an accurate response? Input: '{{ args[0] }}' Tools: {{ args[1] }}",
                rules=[
                    Rule(
                        "The tool name is the key in the tools dictionary, and the description is the value."
                    ),
                    Rule("Only respond with a comma-separated list of tool names."),
                    Rule("Do not include any other information."),
                    Rule("If no tools are needed, respond with 'None'."),
                ],
            ),
        ],
        conversation_memory=ReadOnlyConversationMemory(),
    )
    output = agent.run(message, tools_descriptions).output.value
    tool_names = output.split(",") if output != "None" else []
    logger.info(f"Tools needed: {tool_names}")
    return [tools_dict[tool_name.strip()][0] for tool_name in tool_names]


def _get_knowledge_base_tool(name: str, env_var: str) -> RagTool:
    vector_store_driver = GriptapeCloudVectorStoreDriver(
        knowledge_base_id=os.getenv(env_var, ""),
    )

    vector_store_retrieval_rag_module = VectorStoreRetrievalRagModule(
        vector_store_driver=vector_store_driver
    )

    rag_engine = RagEngine(
        retrieval_stage=RetrievalRagStage(
            retrieval_modules=[vector_store_retrieval_rag_module]
        ),
        response_stage=ResponseRagStage(
            response_modules=[TextChunksResponseRagModule()]
        ),
    )

    return RagTool(
        name=name,
        rag_engine=rag_engine,
        description="Knowledge Base with information about Griptape Operational Processes",
    )


def _init_tools_dict() -> dict[str, tuple[BaseTool, str]]:
    """
    Initializes the tools dictionary.
    The return value is a dictionary where the key is the tool name
    and the value is a tuple containing the Tool object and a description
    of what the tool can do
    """
    # TODO: Add other tools here
    rv_knowledge_base_tool = _get_knowledge_base_tool("rvKB", "RV_KNOWLEDGE_BASE_ID")
    truck_knowledge_base_tool = _get_knowledge_base_tool(
        "truckKB", "TRUCK_KNOWLEDGE_BASE_ID"
    )
    quotes_knowledge_base_tool = _get_knowledge_base_tool(
        "quotes", "QUOTE_KNOWLEDGE_BASE_ID"
    )
    return {
        "rv_knowledge_base_tool": (
            rv_knowledge_base_tool,
            rv_knowledge_base_tool.description,
        ),
        "truck_knowledge_base_tool": (
            truck_knowledge_base_tool,
            truck_knowledge_base_tool.description,
        ),
        "quote_knowledge_base_tool": (
            quotes_knowledge_base_tool,
            quotes_knowledge_base_tool.description,
        ),
        "github_tool": (
            GriptapeCloudToolTool(
                tool_id=os.getenv(
                    "GT_CLOUD_GITHUB_TOOL_ID", "fb56b523-ec53-4490-937e-013ef0f16299"
                ),
            ),
            "Intelligent GitHub agent with access to the Griptape and Griptape Cloud repository. Use when asked about the Griptape Framework or Griptape Cloud repository, or GitHub related questions.",
        ),
        "slack_tool": (
            GriptapeCloudToolTool(
                tool_id=os.getenv(
                    "GT_CLOUD_SLACK_TOOL_ID", "17f0ef8c-a5e2-4c2a-8f15-533691225195"
                ),
            ),
            "Tool with access to Slack APIs.",
        ),
    }
