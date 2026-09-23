"""Financial Advisor Strands agent (orchestrator).

Capabilities:
- AgentCore Memory for client profile across sessions (risk tolerance, prior plans)
- A2A delegation to portfolio-analyst, risk-assessment, market-data via Agent Gateway

Exposed at POST / for Agent Gateway /agents/financial-advisor.
"""
from __future__ import annotations

import os
from typing import Any

from rich.console import Console
from strands import Agent, tool

from _shared import a2a_client
from _shared.model import build_model

AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
MEMORY_ID = os.getenv("MEMORY_ID", "")
CLIENT_ACTOR_ID = os.getenv("CLIENT_ACTOR_ID") or "default-client"
SESSION_ID = os.getenv("SESSION_ID") or "default-session"

# Namespace of the Memory's semantic strategy, published by the kro RGD into
# the same Secret that carries MEMORY_ID. AgentCore substitutes {actorId} when
# it writes extracted facts; RetrieveMemoryRecords rejects wildcards and needs
# a fully-resolved path, so we substitute it here for reads.
MEMORY_NAMESPACE_TEMPLATE = os.getenv("MEMORY_NAMESPACE") or "/actors/{actorId}/facts/"

console = Console()


def _memory_namespace() -> str:
    """Resolve the strategy namespace template for this actor."""
    return MEMORY_NAMESPACE_TEMPLATE.replace("{actorId}", CLIENT_ACTOR_ID)


def _memory_client():
    if not MEMORY_ID:
        return None
    from bedrock_agentcore.memory import MemoryClient

    return MemoryClient(region_name=AWS_REGION)


@tool
def get_client_profile() -> dict[str, Any]:
    """Retrieve stored profile (risk tolerance, goals, prior recommendations)."""
    client = _memory_client()
    if client is None:
        return {"status": "skipped", "reason": "Memory not configured"}
    namespace = _memory_namespace()
    try:
        # Semantic search over the facts the strategy extracted from prior
        # sessions' events. `namespace` is required and must be resolved;
        # `top_k` caps the result count (there is no max_results parameter).
        response = client.retrieve_memories(
            memory_id=MEMORY_ID,
            namespace=namespace,
            query="What is this client's risk tolerance, goals, and prior recommendations?",
            top_k=5,
        )
        if response:
            facts = [
                str((record.get("content") or {}).get("text") or record)
                for record in response
            ]
            return {"status": "success", "profile": facts}
        return {"status": "empty", "profile": [], "namespace": namespace}
    except Exception as exc:
        # Surface the failure instead of letting the model read it as "new
        # client" — a silent empty result is indistinguishable from a broken
        # call, which is how this stayed broken.
        console.print(f"[red]get_client_profile failed[/red] ns={namespace}: {exc}")
        return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}


@tool
def save_client_profile(profile: str) -> dict[str, Any]:
    """Persist a client profile snapshot (tolerance, goals, constraints)."""
    client = _memory_client()
    if client is None:
        return {"status": "skipped", "reason": "Memory not configured"}
    try:
        # create_event appends a turn to short-term storage; the semantic
        # strategy then extracts durable facts from it asynchronously.
        # `messages` is a list of (text, role) tuples. save_turn() no longer
        # exists on MemoryClient.
        response = client.create_event(
            memory_id=MEMORY_ID,
            actor_id=CLIENT_ACTOR_ID,
            session_id=SESSION_ID,
            messages=[
                (f"Client profile: {profile}", "USER"),
                ("Profile stored", "ASSISTANT"),
            ],
        )
        return {
            "status": "success",
            "eventId": response.get("eventId"),
            # Extraction is async (~2-3 min), so this fact will not come back
            # from get_client_profile() until the strategy has processed it.
            "note": "Stored; searchable after asynchronous extraction completes",
        }
    except Exception as exc:
        console.print(f"[red]save_client_profile failed[/red]: {exc}")
        return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}


def _delegate(agent_name: str, task: str) -> dict[str, Any]:
    try:
        response = a2a_client.call_agent(agent_name, task)
        return {"status": "success", "agent": agent_name, **response}
    except Exception as exc:
        return {"status": "error", "agent": agent_name, "error": str(exc)}


@tool
def delegate_portfolio(task: str) -> dict[str, Any]:
    """Delegate a portfolio valuation / composition task to portfolio-analyst."""
    return _delegate("portfolio-analyst", task)


@tool
def delegate_risk(task: str) -> dict[str, Any]:
    """Delegate risk evaluation to risk-assessment."""
    return _delegate("risk-assessment", task)


@tool
def delegate_market(task: str) -> dict[str, Any]:
    """Delegate price / market trend questions to market-data."""
    return _delegate("market-data", task)


SYSTEM_PROMPT = """You are a Financial Advisor coordinating a team of specialists.

You can delegate to three agents via these tools:
- delegate_portfolio(task): Portfolio Analyst — valuations and composition
- delegate_risk(task): Risk Assessment — risk scoring vs. tolerance
- delegate_market(task): Market Data — live prices and sector trends

Workflow for any client question:
1. Call get_client_profile() FIRST to recall stored tolerance, goals, prior advice.
2. Delegate each piece of the question to the appropriate specialist. Prefer
   parallelizable information: usually market-data for quotes, portfolio-analyst
   for valuation, risk-assessment for risk scoring.
3. Synthesize a cohesive answer covering: total value, per-holding weights,
   risk verdict vs. tolerance, and a clear recommendation (hold / rebalance /
   diversify).
4. When the client reveals new information (tolerance, goals), call
   save_client_profile(profile) to persist it before ending.

Always consider the complete financial picture and explain tradeoffs in plain
language. Cite specialist outputs when relevant.
"""


def build_agent() -> Agent:
    kwargs: dict[str, Any] = {
        "tools": [
            get_client_profile,
            save_client_profile,
            delegate_portfolio,
            delegate_risk,
            delegate_market,
        ],
        "system_prompt": SYSTEM_PROMPT,
        "name": "FinancialAdvisor",
    }
    model = build_model()
    if model is not None:
        kwargs["model"] = model
    return Agent(**kwargs)
