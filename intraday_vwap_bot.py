import os
import requests

# Get webhook from Railway environment variable
WEBHOOK_URL = os.getenv("WEBHOOK_URL")

def send_test_trade():

    if not WEBHOOK_URL:
        print("WEBHOOK_URL not found.")
        return

    # Simulated trade data
    current_price = 512.43
    target_price = 514.02
    direction = "CALL"
    strike = 515
    expiration = "03/01"
    premium = 2.18
    delta = 0.61

    contracts = 4
    capital_used = premium * 100 * contracts
    estimated_profit = 348

    message = f"""
🚨 VWAP OPTIONS SIGNAL 🚨

Underlying: SPY
Direction: {direction}
Spot: ${current_price}

Contract: SPY {expiration} {strike}C
Delta: {delta}
Premium: ${premium}

Contracts: {contracts}
Capital Used: ${capital_used}
Max Risk: ${capital_used}

Target Underlying: ${target_price}
Estimated Profit: ${estimated_profit}
"""

    response = requests.post(WEBHOOK_URL, json={"content": message})

    print("Status Code:", response.status_code)
    print("Test trade alert sent.")

if __name__ == "__main__":
    send_test_trade()
