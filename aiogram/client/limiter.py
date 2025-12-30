import asyncio
import time
from enum import Enum
from collections import deque
from typing import Deque, Dict
from aiogram.exceptions import TelegramRetryAfter


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

    Приоритеты:
      - priority=1 (по умолчанию): "первый" / высокий приоритет.
      - priority=2: "второй" приоритет, который срабатывает только тогда,
        когда нет активных ожиданий с priority=1.

    Использование:
        limiter = TelegramRateLimiter(global_per_second=30)

        # Высокий приоритет (как раньше, по умолчанию)
        await limiter.wait(str(chat_id), ChatType.GROUP)

        # Явно высокий приоритет
        await limiter.wait(str(chat_id), ChatType.GROUP, priority=1)

        # Второй приоритет
        await limiter.wait(str(chat_id), ChatType.GROUP, priority=2)
    """

    # 1 сообщение в секунду на чат
    _CHAT_CAPACITY = 1
    _CHAT_PERIOD = 1.0  # сек

    # 20 сообщений в минуту по всем группам СУММАРНО
    _GROUP_CAPACITY = 20
    _GROUP_PERIOD = 60.0  # сек

    def __init__(self, global_per_second: int, broadcast_share: float = 2.0 / 3.0) -> None:
        if global_per_second is None:
            global_per_second = 29

        if global_per_second <= 0:
            raise ValueError("global_per_second должен быть > 0")
        
        if not (0 < broadcast_share <= 1):
            raise ValueError("broadcast_share должен быть в (0, 1]")

        # Глобальный лимит (N сообщений в секунду по всем чатам)
        self.global_per_second = global_per_second
        self._global_window: Deque[float] = deque()

        # Лимит "1 сообщение в секунду на чат"
        self._per_chat_window: Dict[str, Deque[float]] = {}

        # ЕДИНЫЙ лимит для ВСЕХ групп: 20 сообщений в минуту
        self._groups_global_window: Deque[float] = deque()

        # Доп. окно для broadcast-сообщений (priority=2)
        self.broadcast_capacity = max(1, int(self.global_per_second * broadcast_share))
        self._broadcast_window: Deque[float] = deque()

        # Один общий lock для атомарных обновлений окон
        self._lock = asyncio.Lock()

        # Сколько сейчас активных ожиданий высокого приоритета (priority=1)
        # Любое ожидание со вторым приоритетом (priority=2) будет пускаться
        # только если это значение равно 0.
        self._high_priority_waiters: int = 0

    async def wait(
        self,
        chat_id: str,
        chat_type: ChatType = ChatType.PRIVATE,
        priority: int = 1,
    ) -> None:
        """
        Ожидает, пока все лимиты (глобальный, по чату, групповой и при необходимости broadcast)
        позволят отправить следующее сообщение.

        priority:
          1 — высокий приоритет (обычные сообщения)
          2 — broadcast: ограничен отдельным лимитом (2/3 от global_per_second)
              и запускается только когда нет активных high priority.
        """
        chat_id = str(chat_id)
        is_high_priority = (priority == 1)
        is_broadcast = (priority == 2)

        # Регистрируем "я — ожидающий с высоким приоритетом"
        if is_high_priority:
            async with self._lock:
                self._high_priority_waiters += 1

        try:
            while True:
                async with self._lock:
                    now = time.monotonic()

                    # Если это второй приоритет, а есть хотя бы один
                    # активный высокий приоритет — даём небольшую задержку
                    # и даже не пробуем "залезать" в лимиты.
                    if not is_high_priority and self._high_priority_waiters > 0:
                        delay = 0.05  # 50 мс — просто чтобы не крутить цикл
                    else:
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

                        # 4. Доп. лимит для broadcast-сообщений
                        delay_broadcast = 0.0
                        if is_broadcast:
                            delay_broadcast = self._calc_delay(
                                window=self._broadcast_window,
                                capacity=self.broadcast_capacity,
                                period=1.0,
                                now=now,
                            )

                        # Итоговая задержка — максимум из всех
                        delay = max(delay_global, delay_chat, delay_groups, delay_broadcast)

                        if delay <= 0:
                            # Можно отправлять: фиксируем факт "отправки" в нужных окнах
                            self._global_window.append(now)
                            chat_window.append(now)
                            if chat_type == ChatType.GROUP:
                                self._groups_global_window.append(now)
                            if is_broadcast:
                                self._broadcast_window.append(now)
                            # Выходим из wait()
                            return

                # Спим ВНЕ lock, чтобы не держать его и не блокировать другие корутины
                await asyncio.sleep(delay)
        finally:
            # Как только корутина с высоким приоритетом полностью закончила wait(),
            # уменьшаем счётчик таких ожиданий.
            if is_high_priority:
                async with self._lock:
                    self._high_priority_waiters -= 1

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
