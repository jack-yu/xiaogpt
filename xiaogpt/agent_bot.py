"""带联网搜索工具的 Agent Loop，供 xiaogpt 调用。

设计：裸调 OpenAI 兼容 /v1/chat/completions，实现标准 function calling 循环：
  1. 把用户问题 + 工具定义发给模型
  2. 若模型返回 tool_calls，执行 web_search（AnySearch API）
  3. 把工具结果作为 tool message 回灌，再次调用模型
  4. 最多 MAX_ROUNDS 轮，拿到最终文本回答

依赖：仅标准库 + httpx（xiaogpt 环境已有）。
配置通过环境变量注入：
  OPENAI_BASE_URL / OPENAI_API_KEY / LLM_MODEL
  ANYSEARCH_BASE（可选，默认 https://api.anysearch.com）
"""
from __future__ import annotations

import asyncio
import json
import os
from typing import Any

import httpx

# 所有网关配置从环境变量读取，敏感信息（key）不写入代码。
# 部署时通过 -e OPENAI_BASE_URL=... / -e OPENAI_API_KEY=... 注入。
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
ANYSEARCH_BASE = os.getenv("ANYSEARCH_BASE", "https://api.anysearch.com")
SEARCH_TIMEOUT = float(os.getenv("SEARCH_TIMEOUT", "15"))
MAX_ROUNDS = 3

# 工具定义（OpenAI function calling 格式）
TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "联网搜索最新信息。当用户问题涉及实时数据（油价、天气、新闻、价格、"
                "赛事、股价等会变化的信息）或你不确定的事实时，必须调用此工具查询，"
                "不要凭记忆回答。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索关键词，要具体，例如『2026年11月17日 北京 92号汽油 价格』",
                    }
                },
                "required": ["query"],
            },
        },
    }
]


async def _any_search(client: httpx.AsyncClient, query: str) -> str:
    """调 AnySearch 匿名搜索，返回精简的文本结果（供模型阅读）。"""
    try:
        resp = await client.post(
            f"{ANYSEARCH_BASE}/v1/search",
            json={"query": query, "max_results": 5},
            timeout=SEARCH_TIMEOUT,
        )
        resp.raise_for_status()
        payload = resp.json()
        data = payload.get("data", {}) if isinstance(payload, dict) else {}
        results = data.get("results", []) if isinstance(data, dict) else []
        if not results:
            return "搜索未返回任何结果。"
        lines = []
        for r in results[:5]:
            title = str(r.get("title", "")).strip()
            content = str(r.get("content") or r.get("snippet", "")).strip()
            lines.append(f"- {title}: {content[:400]}")
        return "\n".join(lines)
    except Exception as e:
        return f"搜索失败：{type(e).__name__}: {e}"


async def ask_with_tools(
    messages: list[dict[str, Any]],
    *,
    verbose: bool = False,
) -> str:
    """带工具循环的问答。messages 应已含 system（含日期+规则）和 user 消息。"""
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    conv: list[dict[str, Any]] = list(messages)
    async with httpx.AsyncClient(timeout=60) as client:
        for round_no in range(1, MAX_ROUNDS + 1):
            body = {
                "model": LLM_MODEL,
                "messages": conv,
                "tools": TOOLS,
                "tool_choice": "auto",
                "max_tokens": 500,
                "reasoning_effort": "none",
            }
            resp = await client.post(
                f"{OPENAI_BASE_URL}/chat/completions", headers=headers, json=body
            )
            resp.raise_for_status()
            choice = resp.json()["choices"][0]
            msg = choice["message"]
            tool_calls = msg.get("tool_calls") or []

            if not tool_calls:
                # 没有工具调用，返回最终文本
                return (msg.get("content") or "").strip()

            if verbose:
                print(f"[agent] round {round_no} tool_calls: {len(tool_calls)}")

            # 把 assistant 的 tool_calls 消息原样加入对话
            conv.append(msg)
            # 执行每个工具调用，回灌结果
            for tc in tool_calls:
                fn = tc.get("function", {})
                name = fn.get("name")
                args = json.loads(fn.get("arguments") or "{}")
                if verbose:
                    print(f"[agent] call {name}({args})")
                if name == "web_search":
                    result = await _any_search(client, args.get("query", ""))
                else:
                    result = f"未知工具：{name}"
                conv.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.get("id"),
                        "content": result,
                    }
                )
        # 超过轮数，强制拿一次不带工具的回答
        body["tools"] = []
        body.pop("tool_choice", None)
        body["messages"] = conv + [
            {
                "role": "user",
                "content": "请基于上面搜索到的信息，直接给出简短回答；如果信息不足就说查不到。",
            }
        ]
        resp = await client.post(
            f"{OPENAI_BASE_URL}/chat/completions", headers=headers, json=body
        )
        resp.raise_for_status()
        return (resp.json()["choices"][0]["message"].get("content") or "").strip()
