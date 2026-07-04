"""Capture full-page screenshots of all Retail Support React UI routes."""
from __future__ import annotations

import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "ui-screenshots"
BASE_URL = "http://127.0.0.1:3000"
USERNAME = "admin"
PASSWORD = "Admin@123!"

PAGES = [
    ("01-login.png", "/login", False),
    ("02-dashboard.png", "/dashboard", True),
    ("03-tickets.png", "/tickets", True),
    ("04-customers.png", "/customers", True),
    ("05-ai.png", "/ai", True),
    ("06-rag.png", "/rag", True),
    ("07-analytics.png", "/analytics", True),
    ("08-operations.png", "/operations", True),
    ("09-integrations.png", "/integrations", True),
    ("10-admin.png", "/admin", True),
    ("11-help.png", "/help", True),
]


def wait_for_page_ready(page, path: str) -> None:
    page.wait_for_load_state("networkidle", timeout=30000)
    if path != "/login":
        page.wait_for_selector(".loading-screen", state="detached", timeout=15000)
    page.wait_for_timeout(1500)


def login(page) -> None:
    page.goto(f"{BASE_URL}/login", wait_until="domcontentloaded")
    wait_for_page_ready(page, "/login")
    page.fill("#username", USERNAME)
    page.fill("#password", PASSWORD)
    page.click('button[type="submit"]')
    page.wait_for_url("**/dashboard**", timeout=30000)
    wait_for_page_ready(page, "/dashboard")


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    created: list[str] = []
    issues: list[str] = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1440, "height": 900})
        page = context.new_page()

        for filename, path, requires_auth in PAGES:
            out_path = OUTPUT_DIR / filename
            try:
                if requires_auth and "/dashboard" not in page.url:
                    login(page)

                page.goto(f"{BASE_URL}{path}", wait_until="domcontentloaded")
                wait_for_page_ready(page, path)

                if path != "/login" and "/login" in page.url:
                    raise RuntimeError("Redirected to login; authentication failed")

                page.screenshot(path=str(out_path), full_page=True)
                created.append(str(out_path))
                print(f"[ok] {filename} -> {path}")
            except Exception as exc:
                msg = f"{filename} ({path}): {exc}"
                issues.append(msg)
                print(f"[error] {msg}", file=sys.stderr)

        browser.close()

    print(f"\nOutput folder: {OUTPUT_DIR}")
    print(f"Screenshots created: {len(created)}/{len(PAGES)}")
    for item in created:
        print(f"  - {Path(item).name}")

    if issues:
        print("\nIssues:")
        for issue in issues:
            print(f"  - {issue}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
