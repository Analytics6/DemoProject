"""
Generate 500 retail support tickets for RAG indexing and SQLite storage.

Usage:
  python scripts/generate_tickets.py
  python scripts/generate_tickets.py --count 500 --seed 42
"""
import argparse
import json
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CUSTOMERS = [
    "Jordan Patel", "Nina Brooks", "Leo Gomez", "Sam Rivera", "Emma Wilson",
    "Aarav Sharma", "Priya Nair", "Rohan Verma", "Mia Chen", "Oliver Grant",
    "Sofia Alvarez", "Ethan Moore", "Isabella Rossi", "Noah Kim", "Ava Thompson",
    "Lucas Singh", "Chloe Martin", "Ben Carter", "Zara Khan", "Daniel Wright",
]
AGENTS = ["agent", "admin", "analyst"]
STATUSES = ["open", "pending", "resolved"]
PRIORITIES = ["low", "medium", "high"]
CATEGORIES = [
    "refund", "billing", "shipping", "product", "account", "warranty", "promotion", "inventory",
]

SUBJECT_TEMPLATES = {
    "refund": [
        "Refund request for order #{oid}",
        "Partial refund needed — damaged item #{oid}",
        "Return not processed for order #{oid}",
        "Refund status inquiry order #{oid}",
    ],
    "billing": [
        "Billing discrepancy on invoice #{oid}",
        "Double charge on order #{oid}",
        "Promo code not applied to order #{oid}",
        "Subscription billing error #{oid}",
    ],
    "shipping": [
        "Delivery delay for order #{oid}",
        "Package marked delivered but not received #{oid}",
        "Wrong address on shipment #{oid}",
        "Express shipping not fulfilled #{oid}",
    ],
    "product": [
        "Defective {product} from order #{oid}",
        "Product not as described — {product}",
        "Missing accessory in order #{oid}",
        "Size exchange request for {product}",
    ],
    "account": [
        "Unable to login to account",
        "Password reset not working",
        "Account locked after failed attempts",
        "Update email address on account",
    ],
    "warranty": [
        "Warranty claim for {product}",
        "Extended warranty registration issue",
        "Warranty repair status #{oid}",
    ],
    "promotion": [
        "Summer sale discount not reflected",
        "Loyalty points not credited",
        "Gold tier benefits inquiry",
        "Promotion eligibility question",
    ],
    "inventory": [
        "Is {product} back in stock?",
        "Pre-order availability for {product}",
        "Store pickup stock check — {product}",
    ],
}

PRODUCTS = [
    "Wireless Earbuds", "Smart Watch S2", "Denim Jeans", "Classic Cotton T-Shirt",
    "Coffee Maker 1.5L", "Air Fryer 4L", "Running Shoes", "Laptop Sleeve",
]

RESOLUTIONS = [
    "Issued full refund and sent confirmation email.",
    "Replacement item shipped with expedited delivery.",
    "Applied promotional credit to customer account.",
    "Escalated to billing team for manual review.",
    "Provided tracking update and $10 courtesy credit.",
    "Walked customer through account recovery steps.",
    "Registered warranty and scheduled repair pickup.",
    "Confirmed inventory restock date and offered rain check.",
]


def _random_date(start_days_ago: int = 180) -> str:
    base = datetime.utcnow() - timedelta(days=random.randint(1, start_days_ago))
    return base.strftime("%Y-%m-%d %H:%M:%S")


def generate_ticket(ticket_id: int, rng: random.Random) -> dict:
    category = rng.choice(CATEGORIES)
    oid = rng.randint(1000, 9999)
    product = rng.choice(PRODUCTS)
    subject_tpl = rng.choice(SUBJECT_TEMPLATES[category])
    subject = subject_tpl.format(oid=oid, product=product)
    status = rng.choices(STATUSES, weights=[30, 25, 45])[0]
    priority = rng.choices(PRIORITIES, weights=[25, 50, 25])[0]
    if category in ("refund", "billing") and status == "open":
        priority = rng.choice(["high", "medium"])

    description = (
        f"Customer reports: {subject}. "
        f"Category: {category}. Order reference #{oid}. "
        f"Product involved: {product}. "
        f"Customer contacted support via {'email' if rng.random() > 0.3 else 'chat'}. "
        f"Priority level: {priority}."
    )
    resolution = rng.choice(RESOLUTIONS) if status == "resolved" else ""

    return {
        "id": ticket_id,
        "customer_name": rng.choice(CUSTOMERS),
        "subject": subject,
        "description": description,
        "category": category,
        "status": status,
        "priority": priority,
        "assigned_to": rng.choice(AGENTS),
        "order_id": f"ORD-{oid}",
        "product": product,
        "resolution": resolution,
        "created_at": _random_date(),
    }


def generate_tickets(count: int = 500, seed: int = 42) -> list:
    rng = random.Random(seed)
    return [generate_ticket(i + 1, rng) for i in range(count)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate support ticket dataset")
    parser.add_argument("--count", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=str(ROOT / "data" / "tickets.json"))
    args = parser.parse_args()

    tickets = generate_tickets(count=args.count, seed=args.seed)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(tickets, indent=2), encoding="utf-8")

    by_cat = {}
    for t in tickets:
        by_cat[t["category"]] = by_cat.get(t["category"], 0) + 1

    print(f"Generated {len(tickets)} tickets -> {output}")
    print("By category:", by_cat)


if __name__ == "__main__":
    main()
