import json
from datetime import datetime
from pathlib import Path
from typing import Dict


class ApplicationLogs:
    """
    Captures app-level metrics: response time and API calls.
    """

    def __init__(self, file_path: str = "logging/application_metrics.jsonl") -> None:
        self.file_path = Path(file_path)
        self.file_path.parent.mkdir(parents=True, exist_ok=True)

    def log_request(self, endpoint: str, response_time_ms: int, api_calls: int, status: str) -> None:
        payload: Dict = {
            "timestamp": datetime.utcnow().isoformat(),
            "endpoint": endpoint,
            "response_time_ms": response_time_ms,
            "api_calls": api_calls,
            "status": status,
        }
        with self.file_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(payload) + "\n")
