from typing import List, Dict


class ConversationManager:
    """
    Manages LLM conversation history with turn limits.
    """

    def __init__(self, max_turns: int = 10):
        self.max_turns = max_turns
        self._history: List[Dict] = []

    def add_user(self, content: str) -> None:
        self._history.append({"role": "user", "content": content})
        self._trim()

    def add_assistant(self, content: str) -> None:
        self._history.append({"role": "assistant", "content": content})
        self._trim()

    def get_history(self) -> List[Dict]:
        return list(self._history)

    def reset(self) -> None:
        self._history.clear()

    def _trim(self) -> None:
        """
        Keep only the most recent turns.
        One turn = user + assistant
        """
        max_messages = self.max_turns * 2
        if len(self._history) > max_messages:
            self._history = self._history[-max_messages:]