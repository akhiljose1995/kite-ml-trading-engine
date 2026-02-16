from datetime import date
from typing import Optional


class CostGuard:
    """
    Guards LLM usage based on token limits.
    """

    def __init__(
        self,
        *,
        max_daily_tokens: int,
        max_prompt_tokens: int,
        enabled: bool = True,
    ):
        self.enabled = enabled
        self.max_daily_tokens = max_daily_tokens
        self.max_prompt_tokens = max_prompt_tokens

        self._used_tokens_today = 0
        self._current_day = date.today()
        self._disabled = False

    def is_allowed(self, prompt: str) -> bool:
        if not self.enabled:
            return True

        self._rollover_if_needed()

        prompt_tokens = self._estimate_tokens(prompt)
        print(f"Estimated prompt tokens: {prompt_tokens}, Max allowed: {self.max_prompt_tokens}")
        if prompt_tokens > self.max_prompt_tokens:
            self._disabled = True
            return False

        print(f"Tokens used today: {self._used_tokens_today}/{self.max_daily_tokens}")
        if self._used_tokens_today + prompt_tokens > self.max_daily_tokens:
            self._disabled = True
            return False

        return True

    def record_usage(self, usage: dict) -> None:
        if not self.enabled:
            return

        total_tokens = usage.get("total_tokens", 0)
        print(f"LLM usage - Prompt tokens: {usage.get('prompt_tokens', 0)},"
                f" Completion tokens: {usage.get('completion_tokens', 0)},"
                f" Total tokens: {total_tokens}")
        self._used_tokens_today += total_tokens

    def is_disabled(self) -> bool:
        return self._disabled

    def _rollover_if_needed(self) -> None:
        today = date.today()
        if today != self._current_day:
            self._current_day = today
            self._used_tokens_today = 0
            self._disabled = False

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """
        Rough heuristic: ~4 chars per token.
        Good enough for guarding.
        """
        if not text:
            return 0
        return max(1, len(text) // 4)