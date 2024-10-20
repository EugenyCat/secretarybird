from telegram import Bot
import os


class Observer:
    """
        Interface Observer
    """
    def notify(self, message):
        pass


class TelegramNotifier(Observer):
    """
        Concrete observer
    """
    # Telegram credential
    __BOT_TOKEN = os.getenv('TOKEN_TELEGRAM_ALERT_BOT')
    __CHAT_ID = os.getenv('CHAT_ID_TELEGRAM_ALERT_BOT')

    def __init__(self):
        self.bot = Bot(token=self.__BOT_TOKEN)

    def notify(self, message):
        self.bot.send_message(chat_id=self.__CHAT_ID, text=message, parse_mode='HTML')
