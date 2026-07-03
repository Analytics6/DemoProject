import json
from datetime import datetime
from pathlib import Path
from typing import Dict


class SystemLogs:
    """
    Captures system-level metrics (Azure/cloud/tools telemetry).
    """

    def __init__(self, file_path: str = "logging/system_metrics.jsonl") -> None:
        self.file_path = Path(file_path)
        self.file_path.parent.mkdir(parents=True, exist_ok=True)

    def log_azure_metric(
        self, service_name: str, cpu_percent: float, memory_mb: float, tool_status: str
    ) -> None:
        payload: Dict = {
            "timestamp": datetime.utcnow().isoformat(),
            "service": service_name,
            "cpu_percent": cpu_percent,
            "memory_mb": memory_mb,
            "tool_status": tool_status,
            "source": "azure_cloud_tools",
        }
        with self.file_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(payload) + "\n")
