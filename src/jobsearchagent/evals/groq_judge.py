"""Custom Groq Judge Model for DeepEval.

Allows DeepEval to use Groq LLMs (like qwen/qwen3.8-27b or openai/gpt-oss-120b)
as evaluation judges, staying entirely within free tier limits as specified
in Project Spec Section 6.3.
"""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv

ENV_PATH = Path(__file__).resolve().parents[3] / ".env"
load_dotenv(dotenv_path=ENV_PATH)
load_dotenv()

from deepeval.models.base_model import DeepEvalBaseLLM
from groq import AsyncGroq, Groq


class GroqJudge(DeepEvalBaseLLM):
    """Custom LLM Judge powered by Groq for DeepEval metrics."""

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or os.getenv("GROQ_JUDGE_MODEL", "qwen/qwen3.8-27b")
        self.client: Optional[Groq] = None
        self.async_client: Optional[AsyncGroq] = None
        self.load_model()
        super().__init__(self.model_name)

    def load_model(self) -> Groq:
        api_key = os.getenv("GROQ_API_KEY")
        self.client = Groq(api_key=api_key)
        self.async_client = AsyncGroq(api_key=api_key)
        return self.client

    def get_model_name(self) -> str:
        return self.model_name

    def generate(self, prompt: str, schema: Optional[Any] = None) -> str:
        if self.client is None:
            self.load_model()

        kwargs: dict[str, Any] = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
        }

        for attempt in range(4):
            try:
                res = self.client.chat.completions.create(**kwargs)
                return res.choices[0].message.content or ""
            except Exception as e:
                if "rate_limit" in str(e).lower() and attempt < 3:
                    time.sleep(3.0 * (attempt + 1))
                    continue
                raise

        return ""

    async def a_generate(self, prompt: str, schema: Optional[Any] = None) -> str:
        if self.async_client is None:
            self.load_model()

        kwargs: dict[str, Any] = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
        }

        for attempt in range(4):
            try:
                res = await self.async_client.chat.completions.create(**kwargs)
                return res.choices[0].message.content or ""
            except Exception as e:
                if "rate_limit" in str(e).lower() and attempt < 3:
                    await asyncio.sleep(3.0 * (attempt + 1))
                    continue
                raise

        return ""
