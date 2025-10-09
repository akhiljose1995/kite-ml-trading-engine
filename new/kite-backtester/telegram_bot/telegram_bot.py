import requests
import config

class TelegramBot:
    """
    TelegramBot class for sending messages using Telegram Bot API.
    """

    def __init__(self, bot_token=None):
        """
        Initializes the TelegramBot instance.

        Args:
            bot_token (str, optional): Telegram bot token. If not provided, it uses config.TELEGRAM_BOT_TOKEN.
        """
        self.bot_token = bot_token or config.TELEGRAM_BOT_TOKEN

    def send_message(self, message, chat_id):
        """
        Sends a message to a Telegram user or group.

        Args:
            message (str): The message text to send.
            chat_id (int or str): The chat ID (user or group). For groups, usually starts with -100.

        Returns:
            dict: Telegram API response as JSON.
        """
        response_list = []
        for cid in chat_id:
            url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
            payload = {
                'chat_id': cid,
                'text': message
            }

            try:
                response = requests.post(url, data=payload)
                response.raise_for_status()
                response_list.append(response.json())
            except requests.exceptions.RequestException as e:
                response_list.append({"ok": False, "error": str(e)})
        return response_list
