"""Capture full-page screenshots of all UI routes."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "ui-screenshots"
OUT.mkdir(exist_ok=True)

PAGES = [
    ("01-login", "/login", False),
    ("02-dashboard", "/dashboard", True),
    ("03-tickets", "/tickets", True),
    ("04-customers", "/customers", True),
    ("05-ai-assistant", "/ai", True),
    ("06-rag-center", "/rag", True),
    ("07-analytics", "/analytics", True),
    ("08-operations", "/operations", True),
    ("09-integrations", "/integrations", True),
    ("10-admin", "/admin", True),
    ("11-help", "/help", True),
]


def main() -> None:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "playwright", "-q"])
        subprocess.check_call([sys.executable, "-m", "playwright", "install", "chromium"])
        from playwright.sync_api import sync_playwright

    base = "http://127.0.0.1:3000"
    saved = []

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1440, "height": 900})

        # Login first for protected routes
        page.goto(f"{base}/login", wait_until="networkidle", timeout=120000)
        page.fill("#username", "admin")
        page.fill("#password", "Admin@123!")
        page.click('button[type="submit"]')
        page.wait_for_url("**/dashboard**", timeout=60000)

        for name, route, _ in PAGES:
            if route == "/login":
                # logout and capture login
                page.evaluate("localStorage.removeItem('support_token')")
                page.goto(f"{base}/login", wait_until="networkidle", timeout=60000)
            else:
                page.goto(f"{base}{route}", wait_until="networkidle", timeout=60000)
            path = OUT / f"{name}.png"
            page.screenshot(path=str(path), full_page=True)
            saved.append(path.name)
            print(f"Saved {path}")

        browser.close()

    print(f"\nDone: {len(saved)} screenshots in {OUT}")


if __name__ == "__main__":
    main()
