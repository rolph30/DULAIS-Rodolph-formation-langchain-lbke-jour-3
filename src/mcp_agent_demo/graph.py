"""Minimal LangChain agent graph for deployment."""

from __future__ import annotations
import contextlib
import logging
from typing import TypedDict
from langgraph_sdk.runtime import ServerRuntime


import os
from datetime import datetime, timezone

from langchain.agents import AgentState, create_agent
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool

from langchain.mcp import MCPAdapter
from pathlib import Path
from os import path

# 1. Helper function to fetch MCP tools synchronously using asyncio.run()

# FIXME: proposed by chat.langchain to address langchain being async
# But it doesn't work, we'll wait for official support


# For OpenRouter, we won't use this default setup at all
# DEFAULT_MODEL = os.getenv("SIMPLE_AGENT_MODEL", "anthropic:claude-sonnet-4-6")

# https://openrouter.ai/docs/guides/community/langchain
model = init_chat_model(
    # model="mistralai/mistral-small-2603",
    model="google/gemini-3.1-flash-lite",
    model_provider="openai",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
)


class Context(TypedDict):
    """
    # Contexte = valeurs globales (souvent read-only)
    """
    pass


class State(AgentState):
    """
    # State = état de l'agent
    # Important : doit hériter de "AgentState" pour le champ messages
    """
    pass

# Autres exempltes d'outils


@tool
def utc_now() -> str:
    """Return the current UTC timestamp in ISO format."""
    return datetime.now(tz=timezone.utc).isoformat()


logger = logging.getLogger(__name__)


def mcp_servers() -> dict:
    """Serveurs MCP auxquels l'agent se connecte.

    - Une URL http(s) = serveur MCP distant
    - Un Path = script local lancé en stdio
    """
    return {
        # Documentation LangChain (guides, tutoriels)
        "docs-langchain": "https://docs.langchain.com/mcp",
        # Référence d'API (classes, méthodes, paramètres)
        "reference-langchain": "https://reference.langchain.com/mcp",
        # Script local FastMCP
        "calculator": Path(__file__).resolve().with_name("mcp_calculator.py"),
    }


def make_agent(tools: list):
    return create_agent(
        model=model,  # DEFAULT_MODEL,
        tools=tools,
        state_schema=State,
        context_schema=Context,
        system_prompt=(
            "You are a concise assistant. "
            "Use tools when they add factual precision, then return a direct answer."
        ),
        name="mcp_demo_agent",
    )


@contextlib.asynccontextmanager
async def make_graph(runtime: ServerRuntime):
    """
    Demonstration d'un graphe généré en asynchrone,
    via un le pattern "context manager factory"

    Cela permet par exemple :
    - De récupérer une liste d'outils via le protocole MCP
    - De récupérer le prompt système dans une base de données

    Il faut ensuite indiquer la bonne fonction dans langgraph.json :
    "mcp_agent_demo": "mcp_agent_demo.graph:make_graph"

    Documentation :
    - https://docs.langchain.com/langsmith/graph-rebuild
    - https://docs.langchain.com/oss/python/langchain/mcp

    NOTE: L'option "--allow-blocking" peut s'avérer nécessaire:
    uv run langgraph dev --allow-blocking

    """
    # Chargement d'un script local en mode MCP stdio (avec FastMCP)
    # On utilise "Path" de pathlib car le MCPAdapter
    # le reconnaît automatiquement comme un fichier local
    # https://docs.langchain.com/oss/python/langchain/mcp#transports
    ert = runtime.execution_runtime
    if ert is None:
        # Appel d'introspection, on ne charge pas les outils MCP
        yield make_agent(tools=[])
    else:
        # Exécution du graphe : on se connecte à plusieurs serveurs MCP
        # en même temps et on regroupe tous leurs outils.
        tools = [utc_now]
        async with contextlib.AsyncExitStack() as stack:
            for name, target in mcp_servers().items():
                try:
                    adapter = await stack.enter_async_context(MCPAdapter(target))
                    server_tools = await adapter.list_tools()
                except Exception as exc:  # un serveur HS ne bloque pas les autres
                    logger.warning("Serveur MCP '%s' ignoré : %s", name, exc)
                    continue
                logger.info(
                    "Serveur MCP '%s' : %d outils", name, len(server_tools))
                tools.extend(server_tools)
            yield make_agent(tools=tools)
