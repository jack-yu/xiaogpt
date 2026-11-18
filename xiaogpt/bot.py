from __future__ import annotations

from typing import Any, AsyncIterator

import httpx2
import republic
from republic.events import TextDelta
from rich import print

import xiaogpt.providers  # noqa: F401  register extra providers
from xiaogpt.config import Config
from xiaogpt.utils import split_sentences

# Keep the latest 5 rounds of conversation
MAX_HISTORY_MESSAGES = 10


class Chatbot:
    """A chat bot backed by Republic, which talks to any supported LLM provider."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.name = config.model.split(":", 1)[1]
        self.history: list[republic.Message] = []
        self._http_client = (
            httpx2.AsyncClient(proxy=config.proxy, timeout=600)
            if config.proxy
            else None
        )
        self.model = republic.get_model(
            config.model,
            api_key=config.api_key or None,
            api_base=config.api_base or None,
            api_format=config.api_format or None,
            http_client=self._http_client,
        )

    def _messages(self, query: str) -> list[republic.Message]:
        messages = list(self.history)
        # 动态注入当前日期时间，解决模型不知道"今天"是哪天的问题
        from datetime import datetime
        now = datetime.now().strftime("%Y年%m月%d日 %A %H:%M")
        week = {"Monday":"星期一","Tuesday":"星期二","Wednesday":"星期三","Thursday":"星期四","Friday":"星期五","Saturday":"星期六","Sunday":"星期日"}
        for en, zh in week.items():
            now = now.replace(en, zh)
        datetime_hint = f"当前日期时间是 {now}（北京时间）。涉及今天/昨天/明天等相对日期时以此为准。"
        if self.config.prompt:
            messages.insert(0, republic.system(datetime_hint + "\n" + self.config.prompt))
        else:
            messages.insert(0, republic.system(datetime_hint))
        messages.append(republic.user(query))
        return messages

    def _add_history(self, query: str, answer: republic.Message) -> None:
        self.history.extend([republic.user(query), answer])
        del self.history[:-MAX_HISTORY_MESSAGES]

    async def ask(self, query: str, **options: Any) -> str:
        # 使用带联网搜索工具的 Agent Loop（agent_bot 模块）
        import republic
        from agent_bot import ask_with_tools
        msgs = self._messages(query)
        # Republic Message -> 裸 OpenAI dict
        conv = []
        for m in msgs:
            role = getattr(m, "role", "user")
            text = ""
            parts = getattr(m, "parts", None)
            if parts:
                text = "".join(getattr(pt, "text", "") or "" for pt in parts)
            else:
                text = getattr(m, "content", "") or ""
            conv.append({"role": role, "content": text})
        verbose = bool(getattr(self.config, "verbose", 0))
        answer = await ask_with_tools(conv, verbose=verbose)
        if answer:
            self.history.append(republic.assistant(answer))
            del self.history[:-MAX_HISTORY_MESSAGES]
        print(answer)
        return answer

    async def ask_stream(self, query: str, **options: Any) -> AsyncIterator[str]:
        async with self.model.stream(self._messages(query), **options) as stream:

            async def text_gen() -> AsyncIterator[str]:
                async for event in stream:
                    if isinstance(event, TextDelta):
                        print(event.chunk, end="")
                        yield event.chunk

            try:
                async for sentence in split_sentences(text_gen()):
                    yield sentence
            finally:
                print()
        self._add_history(query, stream.response.message)

    async def close(self) -> None:
        await self.model.provider.close()
        if self._http_client is not None:
            await self._http_client.aclose()
