import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from frontend.app_backend import authenticate_user, get_dashboard_stats, initialize_database


class EnterpriseBackendTests(unittest.TestCase):
    def test_database_initialization_and_authentication(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "support.db"
            initialize_database(db_path=str(db_path))

            self.assertTrue(db_path.exists())
            self.assertTrue(authenticate_user("admin", "Admin@123!", db_path=str(db_path)))
            self.assertFalse(authenticate_user("admin", "wrong-pass", db_path=str(db_path)))

            stats = get_dashboard_stats(db_path=str(db_path))
            self.assertGreaterEqual(stats["active_users"], 1)
            self.assertGreaterEqual(stats["open_tickets"], 1)
            self.assertGreaterEqual(stats["audit_events"], 1)


if __name__ == "__main__":
    unittest.main()
