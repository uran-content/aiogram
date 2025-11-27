import asyncio
import time
from enum import Enum
from collections import deque
from typing import Deque, Dict


class ChatType(Enum):
    PRIVATE = "private"
    GROUP = "group"
    CHANNEL = "channel"


class TelegramRateLimiter:
    """
    Асинхронный лимитер для Telegram-бота.

    Ограничения:
      1) Глобально: не более `global_per_second` сообщений в секунду
         (по всем чатам суммарно).
      2) В одном чате: не более 1 сообщения в секунду.
      3) По всем группам суммарно: не более 20 сообщений в минуту.

    Использование:
        limiter = TelegramRateLimiter(global_per_second=30)

        await limiter.wait(str(chat_id), ChatType.GROUP)
        await bot.send_message(chat_id, "Hello")
    """

    # 1 сообщение в секунду на чат
    _CHAT_CAPACITY = 1
    _CHAT_PERIOD = 1.0  # сек

    # 20 сообщений в минуту по всем группам СУММАРНО
    _GROUP_CAPACITY = 20
    _GROUP_PERIOD = 60.0  # сек

    def __init__(self, global_per_second: int) -> None:
        if global_per_second is None:
            global_per_second = 29

        if global_per_second <= 0:
            raise ValueError("global_per_second должен быть > 0")

        # Глобальный лимит (N сообщений в секунду по всем чатам)
        self.global_per_second = global_per_second
        self._global_window: Deque[float] = deque()

        # Лимит "1 сообщение в секунду на чат"
        self._per_chat_window: Dict[str, Deque[float]] = {}

        # ЕДИНЫЙ лимит для ВСЕХ групп: 20 сообщений в минуту
        self._groups_global_window: Deque[float] = deque()

        # Один общий lock для атомарных обновлений окон
        self._lock = asyncio.Lock()

    async def wait(self, chat_id: str, chat_type: ChatType = ChatType.PRIVATE) -> None:
        """
        Ожидает, пока все лимиты (глобальный, по чату и групповой)
        позволят отправить следующее сообщение.

        Не блокирует другие корутины (использует только asyncio.sleep).
        """
        # Можно передавать chat_id как int, но внутри приводим к str для единообразия
        chat_id = str(chat_id)

        while True:
            async with self._lock:
                now = time.monotonic()

                # 1. Глобальный лимит: N сообщений в 1 секунду
                delay_global = self._calc_delay(
                    window=self._global_window,
                    capacity=self.global_per_second,
                    period=1.0,
                    now=now,
                )

                # 2. Лимит 1 сообщение в секунду на конкретный чат
                chat_window = self._per_chat_window.setdefault(chat_id, deque())
                delay_chat = self._calc_delay(
                    window=chat_window,
                    capacity=self._CHAT_CAPACITY,
                    period=self._CHAT_PERIOD,
                    now=now,
                )

                # 3. Лимит 20 сообщений в минуту по ВСЕМ группам
                delay_groups = 0.0
                if chat_type == ChatType.GROUP:
                    delay_groups = self._calc_delay(
                        window=self._groups_global_window,
                        capacity=self._GROUP_CAPACITY,
                        period=self._GROUP_PERIOD,
                        now=now,
                    )

                # Итоговая задержка — максимум из всех
                delay = max(delay_global, delay_chat, delay_groups)

                if delay <= 0:
                    # Можно отправлять: фиксируем факт "отправки" в нужных окнах
                    self._global_window.append(now)
                    chat_window.append(now)
                    if chat_type == ChatType.GROUP:
                        self._groups_global_window.append(now)
                    return  # выходим из wait()

            # Спим ВНЕ lock, чтобы не держать его и не блокировать другие корутины
            await asyncio.sleep(delay)

    @staticmethod
    def _calc_delay(
        window: Deque[float],
        capacity: int,
        period: float,
        now: float,
    ) -> float:
        """
        Скользящее окно:
        - выкидываем старые события (старше period),
        - если событий меньше capacity — ограничение не активно (delay = 0),
        - иначе считаем, сколько надо подождать, чтобы "освободился" слот.
        """
        # Очищаем от старых таймстемпов
        while window and (now - window[0]) >= period:
            window.popleft()

        if len(window) < capacity:
            return 0.0

        oldest = window[0]
        return period - (now - oldest)
