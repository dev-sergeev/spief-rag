"""Cloud.ru Embeddings с локальным SQLite-кэшем и журналом каждой попытки."""

from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from hashlib import sha256
import json
import math
from pathlib import Path
import sqlite3
import time
from typing import Literal

from dotenv import dotenv_values
import httpx
import numpy as np


MODEL = "ai-sage/Giga-Embeddings-instruct-3B"


def create_client(project_dir: Path) -> httpx.Client:
    """Прочитать ключ Cloud.ru из окружения/.env в корне проекта или запросить ввод.

    Ключ не сохраняется в исходниках и не читается из соседних проектов.
    При интерактивном вводе хранится только в окружении текущего процесса.
    """
    import os
    from getpass import getpass

    config = {**dotenv_values(project_dir / ".env"), **os.environ}
    credentials = config.get("CLOUD_RU_API_KEY")
    if not credentials:
        credentials = getpass("API-ключ Cloud.ru (скрытый ввод): ").strip()
        if not credentials:
            raise ValueError("Нужен CLOUD_RU_API_KEY для Cloud.ru")
        os.environ["CLOUD_RU_API_KEY"] = credentials
    return httpx.Client(
        base_url="https://foundation-models.api.cloud.ru/v1",
        headers={"Authorization": f"Bearer {credentials}"},
        timeout=httpx.Timeout(300, connect=30),
        follow_redirects=False,
    )


class EmbeddingStore:
    """Кэш отдельных текстов по модели и SHA-256; журнал без секретов.

    Записываются успешные и неуспешные вызовы. Отсутствующий usage при
    ошибке хранится как NULL, а не как нулевая стоимость. Временные ошибки повторяются.
    """

    def __init__(
        self, path: Path, client: httpx.Client, model: str = MODEL,
        *, max_attempts: int = 30,
    ) -> None:
        """Открыть кэш; max_attempts включает исходную попытку и все повторы."""
        if max_attempts < 1:
            raise ValueError("max_attempts должен быть положительным")
        self.max_attempts = max_attempts
        path.parent.mkdir(parents=True, exist_ok=True)
        self.client = client
        self.model = model
        self._last_request = 0.0
        self.db = sqlite3.connect(path)
        self.db.executescript(
            "CREATE TABLE IF NOT EXISTS vectors ("
            "model TEXT, digest TEXT, vector TEXT NOT NULL, "
            "PRIMARY KEY(model, digest));"
            "CREATE TABLE IF NOT EXISTS calls ("
            "id INTEGER PRIMARY KEY, model TEXT, items INTEGER, "
            "seconds REAL, tokens INTEGER, error TEXT);"
            "CREATE TABLE IF NOT EXISTS call_usage ("
            "call_id INTEGER PRIMARY KEY, usage TEXT);"
        )

    def close(self) -> None:
        """Закрыть соединение с локальным кэшем."""
        self.db.close()

    def embed(self, texts: list[str], input_type: Literal["passage", "query"] = "passage") -> np.ndarray:
        """Получить векторы, отправляя отсутствующие тексты пачками до 32.

        Кэш сохраняется после каждой успешной пачки. Порядок API-ответа
        восстанавливается по index, а не предполагается совпадающим с запросом.
        """
        if input_type not in ("passage", "query"):
            raise ValueError("input_type должен быть passage или query")
        keys = [sha256((input_type + "\0" + text).encode("utf-8")).hexdigest() for text in texts]
        vectors = {}
        missing = {}
        for key, text in zip(keys, texts):
            if key in vectors or key in missing:
                continue
            row = self.db.execute(
                "SELECT vector FROM vectors WHERE model=? AND digest=?",
                (self.model, key),
            ).fetchone()
            if row:
                vectors[key] = json.loads(row[0])
            else:
                missing[key] = text
        items = list(missing.items())
        for offset in range(0, len(items), 32):
            batch = items[offset:offset + 32]
            embeddings = self._fetch_embeddings([text for _, text in batch])
            for (key, _), vector in zip(batch, embeddings):
                vectors[key] = vector
                self.db.execute(
                    "INSERT OR REPLACE INTO vectors VALUES (?, ?, ?)",
                    (self.model, key, json.dumps(vector)),
                )
            self.db.commit()
        return np.asarray([vectors[key] for key in keys], dtype=np.float32)

    def _fetch_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Повторять временные сбои, сохраняя usage и ошибку каждой попытки."""
        for attempt in range(1, self.max_attempts + 1):
            delay = 3.1 - (time.monotonic() - self._last_request)
            if delay > 0:
                time.sleep(delay)
            self._last_request = time.monotonic()
            started = time.perf_counter()
            usage = None
            tokens = None
            error = None
            retry_delay = None
            try:
                response = self.client.post(
                    "/embeddings",
                    json={"input": texts, "model": self.model, "encoding_format": "float"},
                )
                try:
                    payload = response.json()
                except ValueError:
                    response.raise_for_status()
                    raise
                usage = payload.get("usage")
                tokens = usage.get("prompt_tokens") if usage else None
                response.raise_for_status()
                data = payload["data"]
                by_index = {item["index"]: item["embedding"] for item in data}
                if len(data) != len(texts) or set(by_index) != set(range(len(texts))):
                    raise ValueError("API вернул неполный набор эмбеддингов")
                matrix = np.asarray([by_index[i] for i in range(len(texts))])
                if matrix.ndim != 2 or not np.isfinite(matrix).all() or np.any(np.linalg.norm(matrix, axis=1) == 0):
                    raise ValueError("API вернул некорректные векторы")
                return [by_index[i] for i in range(len(texts))]
            except Exception as exc:
                status = exc.response.status_code if isinstance(exc, httpx.HTTPStatusError) else None
                error = type(exc).__name__
                if status is not None:
                    error = f"{error} HTTP {status}"
                transient = (
                    isinstance(exc, (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError))
                    or status in (408, 429, 500, 502, 503, 504)
                )
                if transient and attempt < self.max_attempts:
                    retry_delay = min(60.0, 5.0 * 2 ** min(attempt - 1, 4))
                    if isinstance(exc, httpx.HTTPStatusError):
                        retry_delay = max(retry_delay, _retry_after_seconds(exc.response.headers.get("Retry-After")))
                elif isinstance(exc, (httpx.HTTPError,)):
                    raise RuntimeError(
                        f"Ошибка эмбеддингов: {error}; попытка {attempt}/{self.max_attempts}. "
                        "Кэш сохранён. Проверьте доступность сервиса, ключ и квоту."
                    ) from None
                else:
                    raise
            finally:
                cursor = self.db.execute(
                    "INSERT INTO calls(model, items, seconds, tokens, error) VALUES (?, ?, ?, ?, ?)",
                    (self.model, len(texts), time.perf_counter() - started, tokens, error),
                )
                self.db.execute(
                    "INSERT INTO call_usage VALUES (?, ?)",
                    (cursor.lastrowid, json.dumps(usage)),
                )
                self.db.commit()
            print(
                f"Эмбеддинги: {error}, попытка {attempt}/{self.max_attempts}; "
                f"повтор через {retry_delay:.1f} с.", flush=True,
            )
            time.sleep(retry_delay)

    def usage(self) -> list[dict]:
        """Вернуть журнал этой БД, включая прошлые запуски; это не цены в рублях."""
        cursor = self.db.execute(
            "SELECT calls.id, model, items, seconds, tokens, error, call_usage.usage "
            "FROM calls LEFT JOIN call_usage ON calls.id=call_usage.call_id ORDER BY calls.id"
        )
        names = [column[0] for column in cursor.description]
        records = [dict(zip(names, row)) for row in cursor]
        for record in records:
            record["usage"] = json.loads(record["usage"]) if record["usage"] else None
        return records


def _retry_after_seconds(value: str | None) -> float:
    """Разобрать Retry-After в секундах или формате HTTP-даты."""
    if value is None:
        return 0.0
    try:
        seconds = float(value)
        return max(0.0, seconds) if math.isfinite(seconds) else 0.0
    except ValueError:
        try:
            date = parsedate_to_datetime(value)
            if date.tzinfo is None:
                date = date.replace(tzinfo=timezone.utc)
            return max(0.0, (date - datetime.now(timezone.utc)).total_seconds())
        except (ValueError, TypeError, OverflowError):
            return 0.0
