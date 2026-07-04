import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB = ROOT / "databases" / "support_app.sqlite"


def initialize_database(db_path: Optional[str] = None) -> str:
    db_file = Path(db_path or DEFAULT_DB)
    db_file.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_file)
    cur = conn.cursor()

    cur.executescript(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'agent',
            full_name TEXT NOT NULL,
            department TEXT NOT NULL DEFAULT 'support'
        );

        CREATE TABLE IF NOT EXISTS tickets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            customer_name TEXT NOT NULL,
            subject TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'open',
            priority TEXT NOT NULL DEFAULT 'medium',
            assigned_to TEXT NOT NULL DEFAULT 'agent',
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS audit_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            action TEXT NOT NULL,
            details TEXT NOT NULL,
            created_at TEXT NOT NULL
        );

        INSERT OR IGNORE INTO users (username, password, role, full_name, department)
        VALUES
            ('admin', 'Admin@123!', 'admin', 'Alicia Chen', 'operations'),
            ('agent', 'Agent@123!', 'agent', 'Marcus Lee', 'support'),
            ('analyst', 'Analyst@123!', 'analyst', 'Priya Shah', 'insights');

        INSERT OR IGNORE INTO tickets (customer_name, subject, status, priority, assigned_to, created_at)
        VALUES
            ('Jordan Patel', 'Refund request for order #1042', 'open', 'high', 'agent', '2026-07-04 09:00:00'),
            ('Nina Brooks', 'Billing discrepancy on invoice', 'pending', 'medium', 'admin', '2026-07-04 10:15:00'),
            ('Leo Gomez', 'Delivery delay inquiry', 'resolved', 'low', 'analyst', '2026-07-04 11:00:00');

        INSERT OR IGNORE INTO audit_log (username, action, details, created_at)
        VALUES
            ('admin', 'login', 'Initial admin login', '2026-07-04 09:05:00'),
            ('agent', 'ticket_update', 'Updated refund request', '2026-07-04 09:30:00'),
            ('analyst', 'report_view', 'Viewed dashboard summary', '2026-07-04 10:00:00');
        """
    )
    conn.commit()
    conn.close()
    return str(db_file)


def get_connection(db_path: Optional[str] = None) -> sqlite3.Connection:
    db_file = Path(db_path or DEFAULT_DB)
    db_file.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(db_file)


def authenticate_user(username: str, password: str, db_path: Optional[str] = None) -> bool:
    conn = get_connection(db_path)
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM users WHERE username = ? AND password = ?", (username, password))
    result = cur.fetchone()
    conn.close()
    return bool(result)


def get_user_profile(username: str, db_path: Optional[str] = None) -> Optional[Dict[str, str]]:
    conn = get_connection(db_path)
    cur = conn.cursor()
    cur.execute(
        "SELECT username, role, full_name, department FROM users WHERE username = ?",
        (username,),
    )
    row = cur.fetchone()
    conn.close()
    if not row:
        return None
    return {
        "username": row[0],
        "role": row[1],
        "full_name": row[2],
        "department": row[3],
    }


def get_dashboard_stats(db_path: Optional[str] = None) -> Dict[str, int]:
    conn = get_connection(db_path)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM users")
    active_users = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM tickets WHERE status = 'open'")
    open_tickets = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM audit_log")
    audit_events = cur.fetchone()[0]
    conn.close()
    return {
        "active_users": active_users,
        "open_tickets": open_tickets,
        "audit_events": audit_events,
    }


def list_tickets(db_path: Optional[str] = None) -> List[Dict[str, str]]:
    conn = get_connection(db_path)
    cur = conn.cursor()
    cur.execute(
        "SELECT id, customer_name, subject, status, priority, assigned_to, created_at FROM tickets ORDER BY id"
    )
    rows = cur.fetchall()
    conn.close()
    return [
        {
            "id": row[0],
            "customer_name": row[1],
            "subject": row[2],
            "status": row[3],
            "priority": row[4],
            "assigned_to": row[5],
            "created_at": row[6],
        }
        for row in rows
    ]


def log_audit_event(username: str, action: str, details: str, db_path: Optional[str] = None) -> None:
    conn = get_connection(db_path)
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO audit_log (username, action, details, created_at) VALUES (?, ?, ?, ?)",
        (username, action, details, datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")),
    )
    conn.commit()
    conn.close()
