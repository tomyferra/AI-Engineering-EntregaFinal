"""Herramientas acotadas por especialista: el investigador solo busca y el analista solo calcula."""
from typing import List, Optional

from langchain_core.tools import BaseTool

from app.rag.knowledge_base import KnowledgeBase
from app.tools.knowledge import build_knowledge_tools
from app.tools.math_tools import build_math_tools
from app.tools.report_sender import InMemoryOutbox, Outbox, RedisOutbox, build_report_sender_tool
from app.tools.web_search import build_web_search_tool


def build_research_tools(kb: KnowledgeBase, tavily_api_key: Optional[str] = None) -> List[BaseTool]:
    tools = build_knowledge_tools(kb)
    web = build_web_search_tool(tavily_api_key)
    if web is not None:
        tools.append(web)
    return tools


__all__ = ["InMemoryOutbox", "Outbox", "RedisOutbox", "build_math_tools", "build_report_sender_tool",
           "build_research_tools"]
