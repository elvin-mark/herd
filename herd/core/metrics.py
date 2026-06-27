from typing import Dict, Any


class MetricsCollector:
    def __init__(self):
        # Maps model_name -> stats dict
        self.stats: Dict[str, Dict[str, Any]] = {}

    def record_request(
        self,
        model_name: str,
        endpoint: str,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        duration_sec: float = 0.0,
        is_error: bool = False,
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
        model_stats["endpoints"][endpoint] = (
            model_stats["endpoints"].get(endpoint, 0) + 1
        )

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
                "requests": reqs,
                "errors": data["errors_total"],
                "prompt_tokens": data["prompt_tokens_total"],
                "completion_tokens": data["completion_tokens_total"],
                "total_tokens": total_tokens,
                "avg_latency_sec": avg_latency,
                "avg_speed_tok_sec": avg_speed,
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
            lines.append(
                f"herd_model_cpu_percent{fmt_labels(labels)} {m.get('cpu_percent', 0.0)}"
            )
            lines.append(
                f"herd_model_memory_bytes{fmt_labels(labels)} {m.get('memory_bytes', 0)}"
            )

        # Cumulative statistics (Counters & Histograms)
        lines.append(
            "# HELP herd_requests_total Total number of API requests sent to Herd."
        )
        lines.append("# TYPE herd_requests_total counter")
        lines.append(
            "# HELP herd_request_errors_total Total number of failed requests."
        )
        lines.append("# TYPE herd_request_errors_total counter")
        lines.append(
            "# HELP herd_tokens_total Total tokens processed (prompt or completion)."
        )
        lines.append("# TYPE herd_tokens_total counter")
        lines.append(
            "# HELP herd_request_duration_seconds_total Total latency spent processing requests in seconds."
        )
        lines.append("# TYPE herd_request_duration_seconds_total counter")

        for model, data in self.stats.items():
            # Requests
            for ep, count in data["endpoints"].items():
                lines.append(
                    f'herd_requests_total{{model="{model}",endpoint="{ep}"}} {count}'
                )
            lines.append(
                f'herd_request_errors_total{{model="{model}"}} {data["errors_total"]}'
            )

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
