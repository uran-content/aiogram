import asyncio
import time
import logging
from enum import Enum
from collections import deque
from typing import Deque, Dict, Any, Optional, Callable, Awaitable, TypeVar

from aiogram.exceptions import TelegramRetryAfter


class ChatType(Enum):
    PRIVATE = "private"
    GROUP = "group"
    CHANNEL = "channel"


T = TypeVar("T")


class TelegramRateLimiter:
    """
    Асинхронный лимитер для Telegram-бота.

    Базовые ограничения:
      1) В одном чате: не более 1 сообщения/запроса в секунду.
      2) В группах/супергруппах/каналах: не более 20 сообщений/запросов в минуту НА КАЖДЫЙ чат.
      3) "Global" ограничение (global_per_second) применяется:
         - всегда для broadcast (priority=2),
         - для НЕ-private чатов (группы/каналы) — как и раньше,
         - НО для private чатов при priority != 2 — НЕ применяется (по вашей просьбе).

    Приоритеты:
      - priority=1: высокий.
      - priority!=1: низкий; не стартует, пока есть активные ожидания priority=1.
      - priority=2: broadcast; дополнительно ограничен broadcast_capacity.

    Важно:
      Telegram может вернуть 429 RetryAfter — его нужно уважать.
      Для этого есть notify_retry_after(...) и run(...).
    """

    # 1 сообщение/сек на чат
    _CHAT_CAPACITY = 1
    _CHAT_PERIOD = 1.0  # сек

    # 20 сообщений/мин на КАЖДЫЙ group/supergroup/channel чат
    _GROUP_CAPACITY = 20
    _GROUP_PERIOD = 60.0  # сек

    def __init__(self, global_per_second: int, broadcast_share: float = 2.0 / 3.0) -> None:
        if global_per_second is None:
            global_per_second = 29

        if global_per_second <= 0:
            raise ValueError("global_per_second должен быть > 0")

        if not (0 < broadcast_share <= 1):
            raise ValueError("broadcast_share должен быть в (0, 1]")

        self.global_per_second = int(global_per_second)

        # Глобальное окно (N сообщений/сек) — теперь используется НЕ всегда
        self._global_window: Deque[float] = deque()

        # Пер-чат окно (1 сообщение/сек на chat_id)
        self._per_chat_window: Dict[str, Deque[float]] = {}

        # Пер-групповое окно (20/60сек) на каждый group/supergroup/channel чат
        self._per_group_window: Dict[str, Deque[float]] = {}

        # Broadcast-окно (priority=2) — доля от global_per_second
        self.broadcast_capacity = max(1, int(self.global_per_second * broadcast_share))
        self._broadcast_window: Deque[float] = deque()

        # RetryAfter backoff: отдельно глобальный и на конкретный чат
        self._global_block_until: float = 0.0
        self._chat_block_until: Dict[str, float] = {}

        # Активные ожидания высокого приоритета
        self._high_priority_waiters: int = 0

        # Condition под одним lock
        self._cond = asyncio.Condition()

        self.logger = logging.getLogger("aiogramLimiter")

    async def wait(
        self,
        chat_id: str,
        chat_type: Any = ChatType.PRIVATE,
        priority: int = 1,
    ) -> None:
        """
        Ждёт, пока можно отправлять.

        Совместимость:
          - chat_id: приводится к str
          - chat_type: ChatType / строка ('private'/'group'/'supergroup'/'channel') / enum aiogram
          - priority: 1 (high), 2 (broadcast), другое (low)
        """
        chat_id = str(chat_id)
        is_high_priority = (priority == 1)
        is_broadcast = (priority == 2)

        if chat_id == "453786465":
            self.logger.warning(f"Пришел 453786465: is_high_priority={is_high_priority}, is_broadcast={is_broadcast}")

        # Регистрируем high-priority waiter
        if is_high_priority:
            async with self._cond:
                self._high_priority_waiters += 1

        try:
            while True:
                async with self._cond:
                    # Low priority ждёт, пока есть high-priority ожидания
                    while (not is_high_priority) and self._high_priority_waiters > 0:
                        await self._cond.wait()

                    now = time.monotonic()

                    # 1) Сначала — принудительные блоки от RetryAfter
                    delay_block = self._calc_block_delay(chat_id=chat_id, now=now)
                    if delay_block > 0:
                        delay = delay_block
                    else:
                        # 2) Решаем: применять ли глобальный лимит для этого кейса
                        apply_global = self._should_apply_global_limit(chat_type=chat_type, priority=priority)

                        # 3) Глобальный лимит (только если apply_global=True)
                        delay_global = 0.0
                        if apply_global:
                            delay_global = self._calc_delay(
                                window=self._global_window,
                                capacity=self.global_per_second,
                                period=1.0,
                                now=now,
                            )

                        # 4) Пер-чат лимит: 1/сек ВСЕГДА (в т.ч. в личке)
                        chat_window = self._per_chat_window.setdefault(chat_id, deque())
                        delay_chat = self._calc_delay(
                            window=chat_window,
                            capacity=self._CHAT_CAPACITY,
                            period=self._CHAT_PERIOD,
                            now=now,
                        )

                        # 5) Пер-групповой лимит 20/мин (только для group/supergroup/channel)
                        delay_group = 0.0
                        group_window: Optional[Deque[float]] = None
                        if self._is_group_limited(chat_type):
                            group_window = self._per_group_window.setdefault(chat_id, deque())
                            delay_group = self._calc_delay(
                                window=group_window,
                                capacity=self._GROUP_CAPACITY,
                                period=self._GROUP_PERIOD,
                                now=now,
                            )

                        # 6) Broadcast-лимит (priority=2)
                        delay_broadcast = 0.0
                        if is_broadcast:
                            delay_broadcast = self._calc_delay(
                                window=self._broadcast_window,
                                capacity=self.broadcast_capacity,
                                period=1.0,
                                now=now,
                            )

                        delay = max(delay_global, delay_chat, delay_group, delay_broadcast)

                        if delay <= 0:
                            # Можно отправлять: фиксируем событие в нужных окнах
                            if apply_global:
                                self._global_window.append(now)

                            chat_window.append(now)

                            if group_window is not None:
                                group_window.append(now)

                            if is_broadcast:
                                self._broadcast_window.append(now)

                            return

                if chat_id == "453786465":
                    self.logger.warning(f"Задержка для 453786465: delay={delay}")

                # Спим вне condition
                await asyncio.sleep(delay)
        finally:
            if is_high_priority:
                async with self._cond:
                    self._high_priority_waiters -= 1
                    if self._high_priority_waiters <= 0:
                        self._high_priority_waiters = 0
                        self._cond.notify_all()

    async def notify_retry_after(self, retry_after: float, chat_id: Optional[str] = None) -> None:
        """
        Сообщить лимитеру, что Telegram вернул RetryAfter.
        Если chat_id задан — блокируем только этот чат.
        Иначе — блокируем глобально.
        """
        retry_after = float(retry_after)
        if retry_after <= 0:
            return

        async with self._cond:
            now = time.monotonic()
            until = now + retry_after
            if chat_id is None:
                self._global_block_until = max(self._global_block_until, until)
            else:
                chat_id = str(chat_id)
                self._chat_block_until[chat_id] = max(self._chat_block_until.get(chat_id, 0.0), until)

            self._cond.notify_all()

    async def run(
        self,
        coro_factory: Callable[[], Awaitable[T]],
        chat_id: str,
        chat_type: Any = ChatType.PRIVATE,
        priority: int = 1,
        *,
        max_retries: int = 3,
    ) -> T:
        """
        Helper: wait() -> выполнить -> при TelegramRetryAfter подождать и ретраить.
        """
        attempts = 0
        while True:
            await self.wait(chat_id=chat_id, chat_type=chat_type, priority=priority)
            try:
                return await coro_factory()
            except TelegramRetryAfter as exc:
                # В большинстве случаев правильнее блокировать конкретный чат.
                self.logger.warning(f"------453786465----- TelegramRetryAfter: {exc.retry_after}s. ChatID: {chat_id}")
                await self.notify_retry_after(exc.retry_after, chat_id=chat_id)

                attempts += 1
                if attempts > max_retries:
                    raise

    def _calc_block_delay(self, chat_id: str, now: float) -> float:
        block_until = max(self._global_block_until, self._chat_block_until.get(chat_id, 0.0))
        if now >= block_until:
            return 0.0
        return block_until - now

    @staticmethod
    def _normalize_chat_type(chat_type: Any) -> str:
        if isinstance(chat_type, ChatType):
            return chat_type.value

        if isinstance(chat_type, Enum):
            value = getattr(chat_type, "value", None)
            if isinstance(value, str):
                return value

        if isinstance(chat_type, str):
            return chat_type

        return str(chat_type)

    @classmethod
    def _is_private(cls, chat_type: Any) -> bool:
        t = cls._normalize_chat_type(chat_type).strip().lower()
        return t == "private"

    @classmethod
    def _is_group_limited(cls, chat_type: Any) -> bool:
        t = cls._normalize_chat_type(chat_type).strip().lower()
        return t in {"group", "supergroup", "channel"}

    @classmethod
    def _should_apply_global_limit(cls, chat_type: Any, priority: int) -> bool:
        """
        Новое правило (как вы просили):
          - broadcast (priority=2): глобальный лимит применяется всегда
          - private + НЕ broadcast: глобальный лимит НЕ применяется
          - всё остальное: глобальный лимит применяется
        """
        if priority == 2:
            return True
        if cls._is_private(chat_type):
            return False
        return True

    @staticmethod
    def _calc_delay(window: Deque[float], capacity: int, period: float, now: float) -> float:
        while window and (now - window[0]) >= period:
            window.popleft()

        if len(window) < capacity:
            return 0.0

        oldest = window[0]
        return period - (now - oldest)
