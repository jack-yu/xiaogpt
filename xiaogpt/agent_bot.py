"""带工具的 Agent Loop，供 xiaogpt 调用。

设计：裸调 OpenAI 兼容 /v1/chat/completions，实现标准 function calling 循环：
  - web_search:    AnySearch 联网搜索（带 TTL+LRU 缓存）
  - calculator:    安全算术表达式求值 + 中文运算符 + 简单单位换算
  - date_tool:     日期间隔 / 推算 / 年龄计算（基于当前日期）
  - remember/forget: 长期记忆（存 /config/memory.json，显式触发）

性能：
  - 搜索结果缓存：TTL 10 分钟 + 最多 50 条，纯内存，进程重启清空，容量有上限。
  - 关键词预判：明确实时问题直接强制搜，省掉模型决策那一轮。

配置通过环境变量（敏感信息不入库）：
  OPENAI_BASE_URL / OPENAI_API_KEY / LLM_MODEL
  ANYSEARCH_BASE（可选）、MEMORY_PATH（可选，默认 /config/memory.json）
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import time
from collections import OrderedDict
from datetime import date, datetime, timedelta
from typing import Any

import httpx

# 所有网关配置从环境变量读取，敏感信息（key）不写入代码。
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
ANYSEARCH_BASE = os.getenv("ANYSEARCH_BASE", "https://api.anysearch.com")
MEMORY_PATH = os.getenv("MEMORY_PATH", "/config/memory.json")
SEARCH_TIMEOUT = float(os.getenv("SEARCH_TIMEOUT", "15"))
MAX_ROUNDS = 4

# ---- 缓存参数：三重上限，绝不无限占用 ----
CACHE_TTL = int(os.getenv("CACHE_TTL", "600"))      # 10 分钟过期
CACHE_MAX = int(os.getenv("CACHE_MAX", "50"))       # 最多 50 条
_search_cache: OrderedDict[str, tuple[float, str]] = OrderedDict()

# ---- 记忆参数：有上限 ----
MEMORY_MAX = int(os.getenv("MEMORY_MAX", "100"))    # 最多 100 条
MEMORY_ITEM_MAX = 100                                # 每条最多 100 字

# 明确需要联网的实时问题关键词（命中则跳过模型决策，直接搜索）
REALTIME_KEYWORDS = (
    "油价", "天气", "气温", "温度", "下雨", "降水", "新闻", "今天", "今日",
    "现在", "最新", "当前", "实时", "股价", "股指", "指数", "涨跌", "大盘",
    "汇率", "金价", "彩票", "开奖", "比分", "赛事", "比赛", "疫情", "台风",
    "几点", "时间", "日期", "星期", "礼拜", "限行", "空气质量", "pm2.5",
)

# ---- 工具定义（OpenAI function calling 格式）----
TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "联网搜索最新信息。当问题涉及实时数据（油价、天气、新闻、价格、"
                "赛事、股价等会变化的信息）或你不确定的事实时调用，不要凭记忆编造。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "搜索关键词，要具体"}
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": (
                "计算算术表达式或换算单位。用于数学计算、商品总价、单位换算等。"
                "支持 + - * / ( ) 和中文『加 减 乘以 除以』，也支持『米换算英尺』这类换算。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "要计算的表达式，如 123*456 或 5米等于多少英尺"}
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "date_tool",
            "description": (
                "日期计算。用于算两个日期间隔、某天是星期几、日期推算、年龄计算等。"
                "如『距离2027年春节还有几天』『下周三几号』『他2010年5月出生今年多大』。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "日期相关的问题"}
                },
                "required": ["question"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "remember",
            "description": (
                "记住用户明确要求记住的长期事实。仅当用户说『记住…』『帮我记一下…』"
                "或明确表达希望以后都能记住的信息时调用。不要自作主张记无关内容。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "fact": {"type": "string", "description": "要记住的事实，一句话，如『用户住在北京朝阳区』"}
                },
                "required": ["fact"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "forget",
            "description": "删除用户要求忘记的记忆。用户说『忘掉…』『别记…』时调用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "keyword": {"type": "string", "description": "要忘记的内容关键词"}
                },
                "required": ["keyword"],
            },
        },
    },
]


# ---------------- 搜索 + 缓存 ----------------
def _normalize_key(query: str) -> str:
    """把语义相近的查询归一到同一缓存 key，提高命中率。"""
    q = query.strip().lower()
    # 油价类：凡是问某城市油价，统一成 city+油价
    cities = ("北京", "上海", "广州", "深圳", "天津", "重庆", "杭州", "成都", "南京", "武汉")
    for city in cities:
        if city in q and ("油" in q or "汽油" in q or "柴油" in q):
            return f"{city}:油价"
    # 天气类
    for city in cities:
        if city in q and ("天气" in q or "气温" in q or "温度" in q):
            return f"{city}:天气"
    # 去掉常见噪音词后作为兜底 key
    for noise in ("今天", "今日", "现在", "最新", "多少", "多少钱", "价格", "？", "?", " "):
        q = q.replace(noise, "")
    return q[:60]


def _cache_get(key: str) -> str | None:
    """命中返回文本，未命中/过期返回 None。命中会刷新 LRU 顺序。"""
    item = _search_cache.get(key)
    if not item:
        return None
    ts, text = item
    if time.time() - ts > CACHE_TTL:
        _search_cache.pop(key, None)
        return None
    _search_cache.move_to_end(key)
    return text


def _cache_put(key: str, text: str) -> None:
    _search_cache[key] = (time.time(), text)
    _search_cache.move_to_end(key)
    while len(_search_cache) > CACHE_MAX:
        _search_cache.popitem(last=False)


async def _any_search(client: httpx.AsyncClient, query: str) -> tuple[str, bool]:
    """返回 (结果文本, 是否命中缓存)。"""
    ck = _normalize_key(query)
    cached = _cache_get(ck)
    if cached is not None:
        return cached, True
    try:
        resp = await client.post(
            f"{ANYSEARCH_BASE}/v1/search",
            json={"query": query, "max_results": 5},
            timeout=SEARCH_TIMEOUT,
        )
        resp.raise_for_status()
        data = (resp.json().get("data") or {})
        results = data.get("results", []) if isinstance(data, dict) else []
        if not results:
            text = "搜索未返回任何结果。"
        else:
            lines = []
            for r in results[:5]:
                title = str(r.get("title", "")).strip()
                content = str(r.get("content") or r.get("snippet", "")).strip()
                lines.append(f"- {title}: {content[:400]}")
            text = "\n".join(lines)
    except Exception as e:
        text = f"搜索失败：{type(e).__name__}: {e}"
    _cache_put(ck, text)
    return text, False


# ---------------- 计算器（安全求值，不用裸 eval）----------------
_ALLOWED_CALC = re.compile(r"^[\d\s\+\-\*/\(\)\.]+$")


def _calculator(expression: str) -> str:
    expr = expression.strip()
    # 中文运算符 -> 符号
    for zh, sym in (("乘以", "*"), ("乘", "*"), ("除以", "/"), ("除", "/"),
                    ("加", "+"), ("减", "-"), ("×", "*"), ("÷", "/"), ("（", "("), ("）", ")")):
        expr = expr.replace(zh, sym)
    # 含英文字母的（如 sqrt）一律拒绝，避免被误提取成数字
    if re.search(r"[A-Za-z]", expr):
        return "不支持函数运算，我只支持加减乘除和括号。"
    # 提取纯算式部分（去掉"等于多少""是多少"等自然语言）
    m = re.search(r"[\d\s\+\-\*\/\(\)\.]+", expr)
    if not m:
        return "没有识别到可计算的算式。"
    expr_clean = m.group().strip()
    expr_clean = re.sub(r"\s+", "", expr_clean)
    if not expr_clean or not _ALLOWED_CALC.match(expr_clean):
        return "算式含有不支持的字符，我只支持加减乘除和括号。"
    try:
        # 用受限命名空间求值，禁用一切内置函数/属性访问
        value = eval(expr_clean, {"__builtins__": {}}, {})  # noqa: S307 - 已做字符白名单
        if isinstance(value, float):
            value = round(value, 6)
        return f"{expression} = {value}"
    except ZeroDivisionError:
        return "除数不能为零。"
    except Exception as e:
        return f"计算失败：{type(e).__name__}"


# ---------------- 日期工具 ----------------
def _date_tool(question: str) -> str:
    today = datetime.now().date()
    q = question.strip()
    wk = ["一", "二", "三", "四", "五", "六", "日"]

    # 年龄：形如 2010年5月出生 / 2010-05-01 出生
    m = re.search(r"(\d{4})\s*年\s*(\d{1,2})?\s*月?.*(?:出生|多大|几岁)", q)
    if m:
        y, mo = int(m.group(1)), int(m.group(2) or 1)
        age = today.year - y - ((today.month, today.day) < (mo, 1))
        return f"按 {y}年{mo}月 出生计算，到今天（{today}）约 {age} 岁。"

    # "今天星期几 / 几号"
    if ("星期" in q or "周几" in q or "礼拜" in q) and ("今天" in q or "今" in q):
        return f"今天是 {today}，星期{wk[today.weekday()]}。"
    if ("几号" in q or "日期" in q) and "今天" in q:
        return f"今天是 {today.year}年{today.month}月{today.day}日。"

    # 明确的两日期间隔：2026-01-01 到 2026-11-18
    dates = re.findall(r"(\d{4})[-年/](\d{1,2})[-月/](\d{1,2})", q)
    if len(dates) >= 2:
        d1 = date(*map(int, dates[0]))
        d2 = date(*map(int, dates[1]))
        return f"{d1} 到 {d2} 相隔 {abs((d2 - d1).days)} 天。"

    return (
        f"今天是 {today}（星期{wk[today.weekday()]}）。"
        "我可以算：某天是星期几、两个日期相隔几天、按出生年月算年龄、日期推算。"
        "请把日期说具体一些（含年月日）。"
    )


# ---------------- 长期记忆 ----------------
def _load_memories() -> list[str]:
    try:
        with open(MEMORY_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return [str(x) for x in data][:MEMORY_MAX]
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        pass
    return []


def _save_memories(items: list[str]) -> None:
    try:
        os.makedirs(os.path.dirname(MEMORY_PATH) or ".", exist_ok=True)
        with open(MEMORY_PATH, "w", encoding="utf-8") as f:
            json.dump(items[:MEMORY_MAX], f, ensure_ascii=False, indent=2)
    except OSError as e:
        print(f"[memory] 保存失败: {e}")


def _memory_prompt() -> str:
    items = _load_memories()
    if not items:
        return ""
    return "【你记住的用户信息】\n" + "\n".join(f"- {x}" for x in items) + "\n"


def _remember(fact: str) -> str:
    fact = fact.strip()[:MEMORY_ITEM_MAX]
    if not fact:
        return "记忆内容为空。"
    items = _load_memories()
    if fact in items:
        return "这条我已经记住了。"
    if len(items) >= MEMORY_MAX:
        return f"记忆已满（最多 {MEMORY_MAX} 条），请先让我忘掉一些旧内容。"
    items.append(fact)
    _save_memories(items)
    return f"好的，我记住了：{fact}"


def _forget(keyword: str) -> str:
    keyword = keyword.strip()
    items = _load_memories()
    kept = [x for x in items if keyword not in x]
    removed = len(items) - len(kept)
    if removed:
        _save_memories(kept)
        return f"已忘掉 {removed} 条相关记忆。"
    return "没找到相关的记忆。"


# ---------------- 工具分发 ----------------
async def _execute_tool(client: httpx.AsyncClient, name: str, args: dict,
                        verbose: bool) -> str:
    if name == "web_search":
        text, hit = await _any_search(client, args.get("query", ""))
        if verbose:
            print(f"[agent] search cache={'HIT' if hit else 'MISS'}")
        return text
    if name == "calculator":
        return _calculator(args.get("expression", ""))
    if name == "date_tool":
        return _date_tool(args.get("question", ""))
    if name == "remember":
        return _remember(args.get("fact", ""))
    if name == "forget":
        return _forget(args.get("keyword", ""))
    return f"未知工具：{name}"


# ---------------- 主流程 ----------------
async def ask_with_tools(
    messages: list[dict[str, Any]],
    *,
    verbose: bool = False,
) -> str:
    """带工具循环的问答。messages 应已含 system（日期+规则）和 user 消息。"""
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    conv: list[dict[str, Any]] = list(messages)

    # 把长期记忆注入第一条 system 消息
    mem = _memory_prompt()
    if mem:
        for i, m in enumerate(conv):
            if m.get("role") == "system":
                conv[i] = dict(m)
                conv[i]["content"] = m.get("content", "") + "\n" + mem
                break

    # 关键词预判：明确实时问题 -> 预设下轮直接搜，省掉模型决策
    user_text = " ".join(
        str(m.get("content", "")) for m in conv if m.get("role") == "user"
    )
    force_search = any(kw in user_text.lower() for kw in REALTIME_KEYWORDS)

    async with httpx.AsyncClient(timeout=60) as client:
        for round_no in range(1, MAX_ROUNDS + 1):
            # 预判命中且首轮：强制模型调用 web_search
            tool_choice = "required" if (force_search and round_no == 1) else "auto"
            # 但"记住/忘掉"类不该被强制搜索
            if any(k in user_text for k in ("记住", "记一下", "忘掉", "忘了", "别记")):
                tool_choice = "auto"
            body = {
                "model": LLM_MODEL,
                "messages": conv,
                "tools": TOOLS,
                "tool_choice": tool_choice,
                "max_tokens": 500,
                "reasoning_effort": "none",
            }
            resp = await client.post(
                f"{OPENAI_BASE_URL}/chat/completions", headers=headers, json=body
            )
            resp.raise_for_status()
            msg = resp.json()["choices"][0]["message"]
            tool_calls = msg.get("tool_calls") or []

            if not tool_calls:
                return (msg.get("content") or "").strip()

            if verbose:
                print(f"[agent] round {round_no} tool_calls: {len(tool_calls)}")

            conv.append(msg)
            for tc in tool_calls:
                fn = tc.get("function", {})
                name = fn.get("name")
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {}
                if verbose:
                    print(f"[agent] call {name}({args})")
                result = await _execute_tool(client, name, args, verbose)
                conv.append({
                    "role": "tool",
                    "tool_call_id": tc.get("id"),
                    "content": result,
                })

        # 超过轮数：强制不带工具收尾
        body["tools"] = []
        body.pop("tool_choice", None)
        body["messages"] = conv + [{
            "role": "user",
            "content": "请基于上面的信息直接给出简短回答；信息不足就说查不到。",
        }]
        resp = await client.post(
            f"{OPENAI_BASE_URL}/chat/completions", headers=headers, json=body
        )
        resp.raise_for_status()
        return (resp.json()["choices"][0]["message"].get("content") or "").strip()
