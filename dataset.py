"""
Part 1 — Task 1: Order Dataset Generator and Validation for Nykaa Domain Support Agent.
Track: E-commerce & Retail (Nykaa).

Requirements:
- Seeded, deterministic generator producing ORDERS list of >= 40 records (we generate 50).
- Categories: Apparel, Electronics, Home, Footwear, Beauty (each >= 3 records).
- Statuses: Placed, Shipped, Delivered, Returned, Refunded (each >= 1 record).
- order_value_inr: realistic price range (INR 299 to INR 14,999).
  Reasoning: Reflects Nykaa's retail mix spanning accessible personal care/beauty items to premium cosmetics, fashion apparel, and electronic styling appliances.
- days_since_created: integer in [0, 30].
- delayed_shipment: boolean, strictly between 10% and 30%.
"""

import random
from typing import List, Dict, Any

# Dataset design choices stated for deterministic grading reproducibility:
DATASET_SEED = 42
TOTAL_ORDERS = 50

CATEGORIES = ["Apparel", "Electronics", "Home", "Footwear", "Beauty"]
# Weighted distribution reflecting Nykaa's beauty-first catalog:
CATEGORY_WEIGHTS = [0.20, 0.15, 0.15, 0.15, 0.35]

STATUSES = ["Placed", "Shipped", "Delivered", "Returned", "Refunded"]
STATUS_WEIGHTS = [0.15, 0.25, 0.40, 0.10, 0.10]

PRICE_RANGES = {
    "Beauty": (299, 4999),
    "Apparel": (599, 6999),
    "Footwear": (799, 7999),
    "Home": (499, 5499),
    "Electronics": (1299, 14999),
}


def generate_orders(seed: int = DATASET_SEED, count: int = TOTAL_ORDERS) -> List[Dict[str, Any]]:
    """Generates a deterministic order dataset for the Nykaa e-commerce domain."""
    rng = random.Random(seed)
    orders: List[Dict[str, Any]] = []

    for i in range(1, count + 1):
        record_id = f"NYK-{1000 + i}"
        category = rng.choices(CATEGORIES, weights=CATEGORY_WEIGHTS, k=1)[0]
        status = rng.choices(STATUSES, weights=STATUS_WEIGHTS, k=1)[0]

        min_price, max_price = PRICE_RANGES[category]
        order_value_inr = rng.randint(min_price // 10, max_price // 10) * 10 - 1  # Realistic e-commerce pricing (.e.g 499, 1299)
        if order_value_inr < min_price:
            order_value_inr = min_price

        days_since_created = rng.randint(0, 30)

        # Delayed shipment logic:
        # Delivered/Returned/Refunded are historical; Placed/Shipped with age > 5 days have higher delay probability.
        if status in ["Placed", "Shipped"]:
            delay_prob = 0.50 if days_since_created > 5 else 0.15
        elif status == "Delivered":
            delay_prob = 0.12 if days_since_created > 7 else 0.05
        else:
            delay_prob = 0.10

        delayed_shipment = rng.random() < delay_prob

        orders.append({
            "record_id": record_id,
            "category": category,
            "status": status,
            "order_value_inr": int(order_value_inr),
            "days_since_created": int(days_since_created),
            "delayed_shipment": bool(delayed_shipment),
        })

    return orders


# Singleton generated dataset
ORDERS: List[Dict[str, Any]] = generate_orders(DATASET_SEED, TOTAL_ORDERS)


def validate_dataset(orders_list: List[Dict[str, Any]] = ORDERS) -> Dict[str, Any]:
    """Validates structural criteria and returns summary metrics."""
    total = len(orders_list)
    assert total >= 40, f"Expected >= 40 orders, got {total}"

    category_counts = {cat: 0 for cat in CATEGORIES}
    status_counts = {stat: 0 for stat in STATUSES}
    delayed_count = 0

    for order in orders_list:
        category_counts[order["category"]] += 1
        status_counts[order["status"]] += 1
        if order["delayed_shipment"]:
            delayed_count += 1

    # Structural assertions
    for cat, count in category_counts.items():
        assert count >= 3, f"Category '{cat}' must have >= 3 records, got {count}"

    for stat, count in status_counts.items():
        assert count >= 1, f"Status '{stat}' must have >= 1 record, got {count}"

    delay_pct = (delayed_count / total) * 100.0
    assert 10.0 <= delay_pct <= 30.0, f"Delayed shipment percentage must be between 10% and 30%, got {delay_pct:.2f}%"

    return {
        "total_records": total,
        "category_counts": category_counts,
        "status_counts": status_counts,
        "delayed_count": delayed_count,
        "delayed_percentage": delay_pct,
        "status_coverage_ok": all(c >= 1 for c in status_counts.values()),
        "category_coverage_ok": all(c >= 3 for c in category_counts.values()),
        "delayed_band_ok": 10.0 <= delay_pct <= 30.0,
    }


def print_dataset_report(orders_list: List[Dict[str, Any]] = ORDERS) -> None:
    """Prints a formatted report of dataset statistics for verification."""
    metrics = validate_dataset(orders_list)
    print("=" * 60)
    print("NYKAA ORDER DATASET REPORT (Part 1 - Task 1)")
    print("=" * 60)
    print(f"Total Orders Generated: {metrics['total_records']}")
    print(f"Random Seed: {DATASET_SEED}")
    print("\nCategory Counts (Requirement: each >= 3):")
    for cat, count in metrics['category_counts'].items():
        print(f"  - {cat:12s}: {count:2d} records")
    print("\nStatus Counts (Requirement: each >= 1):")
    for stat, count in metrics['status_counts'].items():
        print(f"  - {stat:12s}: {count:2d} records")
    print(f"\nDelayed Shipments: {metrics['delayed_count']} / {metrics['total_records']} "
          f"({metrics['delayed_percentage']:.2f}%)")
    print(f"Delayed Band [10% - 30%] check: {'PASSED' if metrics['delayed_band_ok'] else 'FAILED'}")
    print("=" * 60)


if __name__ == "__main__":
    print_dataset_report()
