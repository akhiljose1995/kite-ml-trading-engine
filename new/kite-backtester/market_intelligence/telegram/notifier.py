import time
import requests
from typing import List, Optional
from telegram_text_splitter import split_markdown_into_chunks


class TelegramNotifier:
    """
    Sends messages to Telegram safely with chunking and rate limits.
    """

    TELEGRAM_API_URL = "https://api.telegram.org/bot{token}/sendMessage"
    MAX_MESSAGE_LENGTH = 4000  # keep buffer below Telegram hard limit

    def __init__(
        self,
        *,
        bot_token: str,
        chat_id: str,
        enabled: bool = True,
        parse_mode: Optional[str] = "Markdown",
        rate_limit_sec: float = 2.0,
        timeout_sec: int = 10,
    ):
        self.enabled = enabled
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.parse_mode = parse_mode
        self.rate_limit = rate_limit_sec
        self.timeout = timeout_sec

    def send(self, message: str, **kwargs) -> None:
        """
        Public API: send message safely.
            - Respects enabled flag
            - Handles empty messages
            - Supports optional 'notify' flag for important messages
            - Splits long messages into chunks
        """
        if not self.enabled:
            return

        if not message:
            return
        
        # If a kwargs 'notify' is set
        if kwargs.get("notify"):
            self._send_chunk(f"🔔 {message}")

        else:
            chunks = self._split_message(message)
            #chunks = split_markdown_into_chunks(message)   

            for i, chunk in enumerate(chunks):
                chunk = f"```\n{chunk}\n```"
                #print(f"[TelegramNotifier] Sending message chunk {i+1} (length {len(chunk)}):\n{chunk}\n{'-'*40}")
                self._send_chunk(chunk)
                time.sleep(self.rate_limit)

    # ------------------------
    # Internal helpers
    # ------------------------

    def _send_chunk(self, text: str) -> None:
        url = self.TELEGRAM_API_URL.format(token=self.bot_token)

        payload = {
            "chat_id": self.chat_id,
            "text": text,
        }

        if self.parse_mode:
            payload["parse_mode"] = self.parse_mode

        try:
            response = requests.post(
                url,
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
        except Exception as e:
            # Never crash the pipeline because of Telegram
            print(f"[TelegramNotifier] Error sending message: {e}")

    def _split_message(self, text: str) -> List[str]:
        """
        Split message into Telegram-safe chunks.
        """
        chunks = []
        while len(text) > self.MAX_MESSAGE_LENGTH:
            split_at = text.rfind("\n", 0, self.MAX_MESSAGE_LENGTH)
            if split_at == -1:
                split_at = self.MAX_MESSAGE_LENGTH

            chunks.append(text[:split_at])  # +1 to include the newline character
            text = text[split_at:]

        if text:
            chunks.append(text)

        return chunks