import os
import traceback
from typing import List, Dict, Optional

from openai import OpenAI


class OpenAIClient:
    """
    Thin wrapper around OpenAI Chat Completions API.
    """

    def __init__(
        self,
        *,
        model: str,
        temperature: float,
        max_tokens: int,
        timeout_sec: int = 30,
    ):
        api_key = os.getenv("openai_learners_api_key")
        if not api_key:
            raise ValueError("'openai_learners_api_key' env variable not set")

        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout_sec

    def generate(
        self,
        *,
        prompt: str,
        history: Optional[List[Dict]] = None,
    ) -> tuple[str, dict]:
        """
        Generate LLM response given a prompt and optional conversation history.
        """
        messages = []

        if history:
            messages.extend(history)

        messages.append(
            {
                "role": "user",
                "content": prompt,
            }
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_completion_tokens=self.max_tokens,
                timeout=self.timeout,
            )
            text = response.choices[0].message.content.strip()
            usage = response.usage or {}

            return text, {
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "total_tokens": usage.total_tokens,
            }

        except Exception as e:
            # Fail gracefully – LLM should never break trading pipeline
            print(f"LLM error: {traceback.format_exc()}")
            return f"[LLM ERROR] {str(e)}", {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            }