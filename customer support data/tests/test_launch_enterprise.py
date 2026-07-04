import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from launch_enterprise import build_streamlit_command


class LaunchEnterpriseTests(unittest.TestCase):
    def test_build_streamlit_command_uses_expected_app(self) -> None:
        command = build_streamlit_command(port=8501)

        self.assertEqual(command[0], sys.executable)
        self.assertIn("streamlit", command)
        self.assertTrue(any(item.endswith("frontend/main.py") for item in command))
        self.assertIn("--server.port", command)
        self.assertIn("8501", command)


if __name__ == "__main__":
    unittest.main()
