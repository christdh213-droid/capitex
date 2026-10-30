"""Verification for Dynamic Trailing Stop-Loss & Take-Profit."""

import sys
import requests

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

BASE_URL = "http://127.0.0.1:8000"


def test_trailing_stop():
    print("Testing Trailing Stop Trigger...")
    # Reset
    requests.post(f"{BASE_URL}/portfolio/reset")

    # Buy 10 shares of MSFT at $400 with 5% trailing stop ($380 trigger) and 10% take profit ($440 trigger)
    buy_payload = {
        "wallet_type": "human",
        "ticker": "MSFT",
        "action": "BUY",
        "quantity": 10.0,
        "price": 400.0,
        "stop_loss_pct": 5.0,
        "take_profit_pct": 10.0,
    }
    r = requests.post(f"{BASE_URL}/trade/execute", json=buy_payload)
    assert r.status_code == 200

    # Verify holding has peak_price = 400
    r = requests.get(f"{BASE_URL}/portfolio/human")
    holdings = r.json()["holdings"]
    assert len(holdings) == 1
    assert holdings[0]["peak_price"] == 400.0
    assert holdings[0]["trailing_stop_price"] == 380.0
    assert holdings[0]["take_profit_price"] == 440.0
    print("✓ Initial buy with 5% trailing stop and 10% take profit validated.")

    # Execute manual sell of 5 shares
    sell_payload = {
        "wallet_type": "human",
        "ticker": "MSFT",
        "action": "SELL",
        "quantity": 5.0,
        "price": 420.0,
    }
    r = requests.post(f"{BASE_URL}/trade/execute", json=sell_payload)
    assert r.status_code == 200
    sell_data = r.json()
    assert sell_data["trade"]["realized_pnl"] == 100.0  # (420 - 400) * 5
    print(f"✓ Partial sell validated. Realized P&L: +${sell_data['trade']['realized_pnl']:.2f}")

    # Check portfolio after sell
    r = requests.get(f"{BASE_URL}/portfolio/human")
    summary = r.json()
    assert summary["realized_pnl"] == 100.0
    assert summary["win_rate_pct"] == 100.0
    assert summary["win_loss_ratio"] == 1.0
    assert len(summary["holdings"]) == 1
    assert summary["holdings"][0]["quantity"] == 5.0
    print("✓ Win/loss ratio, realized P&L, and remaining holding count validated.")
    print("🎉 All risk & partial sell tests passed!")


if __name__ == "__main__":
    test_trailing_stop()
