import json
import os
import sqlite3
from collections import deque
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from herd.core.config import HERD_HOME

DB_PATH = os.path.join(HERD_HOME, "history.db")


class MetricsCollector:
    def __init__(self, max_history: int = 50, db_path: str = DB_PATH):
        self.stats: Dict[str, Dict[str, Any]] = {}
        self.history: deque = deque(maxlen=max_history)
        self.in_flight_map: Dict[str, int] = {}
        self.pool_rr_index: int = 0
        self.db_path = db_path
        self._init_db()

    def inc_in_flight(self, model_name: str):
        """Increments in-flight request counter for a model."""
        self.in_flight_map[model_name] = self.in_flight_map.get(model_name, 0) + 1

    def dec_in_flight(self, model_name: str):
        """Decrements in-flight request counter for a model."""
        self.in_flight_map[model_name] = max(0, self.in_flight_map.get(model_name, 1) - 1)

    def get_in_flight(self, model_name: str) -> int:
        """Returns active in-flight requests for a model."""
        return self.in_flight_map.get(model_name, 0)

    def select_least_busy_pool_model(self, pool_models: list) -> Optional[str]:
        """Selects least-busy pool model, using round-robin rotation for tie-breaking."""
        if not pool_models:
            return None

        model_counts = []
        for idx, m in enumerate(pool_models):
            count = self.get_in_flight(m)
            model_counts.append((count, idx, m))

        min_count = min(item[0] for item in model_counts)
        candidates = [item for item in model_counts if item[0] == min_count]

        if len(candidates) == 1:
            return candidates[0][2]

        selected_item = candidates[self.pool_rr_index % len(candidates)]
        self.pool_rr_index = (self.pool_rr_index + 1) % 1000000
        return selected_item[2]

    def _init_db(self):
        """Initializes the SQLite database table for request logs."""
        try:
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS request_logs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT,
                        model_name TEXT,
                        endpoint TEXT,
                        prompt_tokens INTEGER,
                        completion_tokens INTEGER,
                        duration_sec REAL,
                        is_error INTEGER,
                        prompt_snippet TEXT,
                        response_snippet TEXT,
                        full_prompt TEXT,
                        full_response TEXT
                    )
                """)
                conn.commit()
        except Exception:
            pass

    def record_request(
        self,
        model_name: str,
        endpoint: str,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        duration_sec: float = 0.0,
        is_error: bool = False,
        prompt_snippet: str = "",
        response_snippet: str = "",
        full_prompt: Any = None,
        full_response: Any = None,
    ):
        if model_name not in self.stats:
            self.stats[model_name] = {
                "requests_total": 0,
                "errors_total": 0,
                "prompt_tokens_total": 0,
                "completion_tokens_total": 0,
                "total_duration_sec": 0.0,
                "endpoints": {},
            }

        model_stats = self.stats[model_name]
        model_stats["requests_total"] += 1
        if is_error:
            model_stats["errors_total"] += 1

        model_stats["prompt_tokens_total"] += prompt_tokens
        model_stats["completion_tokens_total"] += completion_tokens
        model_stats["total_duration_sec"] += duration_sec

        # Track endpoint specific request counts
        model_stats["endpoints"][endpoint] = model_stats["endpoints"].get(endpoint, 0) + 1

        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

        prompt_str = (
            json.dumps(full_prompt)
            if isinstance(full_prompt, (dict, list))
            else str(full_prompt or "")
        )
        resp_str = (
            json.dumps(full_response)
            if isinstance(full_response, (dict, list))
            else str(full_response or "")
        )

        req_id = None
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO request_logs (
                        timestamp, model_name, endpoint, prompt_tokens, completion_tokens,
                        duration_sec, is_error, prompt_snippet, response_snippet,
                        full_prompt, full_response
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        timestamp,
                        model_name,
                        endpoint,
                        prompt_tokens,
                        completion_tokens,
                        round(duration_sec, 3),
                        1 if is_error else 0,
                        prompt_snippet,
                        response_snippet,
                        prompt_str,
                        resp_str,
                    ),
                )
                conn.commit()
                req_id = cursor.lastrowid
        except Exception:
            pass

        # Record entry in in-memory history buffer
        entry = {
            "id": req_id or (len(self.history) + 1),
            "timestamp": timestamp,
            "model_name": model_name,
            "endpoint": endpoint,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "duration_sec": round(duration_sec, 3),
            "is_error": is_error,
            "prompt_snippet": prompt_snippet,
            "response_snippet": response_snippet,
        }
        self.history.appendleft(entry)

    def get_history(self, limit: Optional[int] = 50) -> List[Dict[str, Any]]:
        """Returns the list of recent requests up to the specified limit from SQLite."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                query = "SELECT id, timestamp, model_name, endpoint, prompt_tokens, completion_tokens, duration_sec, is_error, prompt_snippet, response_snippet FROM request_logs ORDER BY id DESC"
                if limit and limit > 0:
                    query += f" LIMIT {limit}"
                cursor.execute(query)
                rows = cursor.fetchall()
                if rows:
                    return [dict(row) for row in rows]
        except Exception:
            pass

        history_list = list(self.history)
        if limit is not None and limit > 0:
            return history_list[:limit]
        return history_list

    def get_request_by_id(self, request_id: int) -> Optional[Dict[str, Any]]:
        """Retrieves full request & response payload by request ID."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM request_logs WHERE id = ?", (request_id,))
                row = cursor.fetchone()
                if row:
                    res = dict(row)
                    try:
                        res["full_prompt"] = json.loads(res["full_prompt"])
                    except Exception:
                        pass
                    try:
                        res["full_response"] = json.loads(res["full_response"])
                    except Exception:
                        pass
                    return res
        except Exception:
            pass
        return None

    def get_stats(self) -> Dict[str, Any]:
        """Returns computed statistics for all models."""
        result = {}
        for model, data in self.stats.items():
            reqs = data["requests_total"]
            total_tokens = data["prompt_tokens_total"] + data["completion_tokens_total"]
            avg_latency = data["total_duration_sec"] / reqs if reqs > 0 else 0.0

            # Completion speed (tokens per second generated)
            c_tokens = data["completion_tokens_total"]
            avg_speed = (
                c_tokens / data["total_duration_sec"]
                if data["total_duration_sec"] > 0 and c_tokens > 0
                else 0.0
            )

            result[model] = {
                "request_count": reqs,
                "error_count": data["errors_total"],
                "prompt_tokens": data["prompt_tokens_total"],
                "completion_tokens": data["completion_tokens_total"],
                "total_tokens": total_tokens,
                "avg_latency_ms": avg_latency * 1000.0,
                "avg_speed_tps": avg_speed,
                "endpoints": data["endpoints"],
            }
        return result

    def get_prometheus_metrics(self, active_models: list) -> str:
        """Generates Prometheus-formatted metrics string."""
        lines = []

        # 1. Active models count
        lines.append("# HELP herd_active_models Number of active running models.")
        lines.append("# TYPE herd_active_models gauge")
        lines.append(f"herd_active_models {len(active_models)}")

        # Helper to format Prometheus labels
        def fmt_labels(labels: dict) -> str:
            if not labels:
                return ""
            return "{" + ",".join(f'{k}="{v}"' for k, v in labels.items()) + "}"

        # Gauges for active model resource usage (RAM/CPU)
        lines.append(
            "# HELP herd_model_cpu_percent CPU usage percentage of the active model process."
        )
        lines.append("# TYPE herd_model_cpu_percent gauge")
        lines.append(
            "# HELP herd_model_memory_bytes RAM usage in bytes of the active model process."
        )
        lines.append("# TYPE herd_model_memory_bytes gauge")

        for m in active_models:
            labels = {"model": m["model"], "port": str(m["port"])}
            lines.append(f"herd_model_cpu_percent{fmt_labels(labels)} {m.get('cpu_percent', 0.0)}")
            lines.append(f"herd_model_memory_bytes{fmt_labels(labels)} {m.get('memory_bytes', 0)}")

        # Cumulative statistics (Counters & Histograms)
        lines.append("# HELP herd_requests_total Total number of API requests sent to Herd.")
        lines.append("# TYPE herd_requests_total counter")
        lines.append("# HELP herd_request_errors_total Total number of failed requests.")
        lines.append("# TYPE herd_request_errors_total counter")
        lines.append("# HELP herd_tokens_total Total tokens processed (prompt or completion).")
        lines.append("# TYPE herd_tokens_total counter")
        lines.append(
            "# HELP herd_request_duration_seconds_total Total latency spent processing requests in seconds."
        )
        lines.append("# TYPE herd_request_duration_seconds_total counter")

        for model, data in self.stats.items():
            # Requests
            for ep, count in data["endpoints"].items():
                lines.append(f'herd_requests_total{{model="{model}",endpoint="{ep}"}} {count}')
            lines.append(f'herd_request_errors_total{{model="{model}"}} {data["errors_total"]}')

            # Tokens
            lines.append(
                f'herd_tokens_total{{model="{model}",type="prompt"}} {data["prompt_tokens_total"]}'
            )
            lines.append(
                f'herd_tokens_total{{model="{model}",type="completion"}} {data["completion_tokens_total"]}'
            )

            # Duration
            lines.append(
                f'herd_request_duration_seconds_total{{model="{model}"}} {data["total_duration_sec"]:.6f}'
            )

        return "\n".join(lines) + "\n"


# Global collector instance
collector = MetricsCollector()
