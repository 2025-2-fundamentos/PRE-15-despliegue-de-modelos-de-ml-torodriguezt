import argparse
import json
import sys
from typing import Dict
import requests

DEFAULT_PAYLOAD: Dict[str, float] = {
    "bedrooms": 3,
    "bathrooms": 2.0,
    "sqft_living": 1800,
    "sqft_lot": 5000,
    "floors": 1.0,
    "waterfront": 0,
    "condition": 3,
}


def main():
    parser = argparse.ArgumentParser(description="Client for the house price prediction API")
    parser.add_argument("--url", default="http://127.0.0.1:5000/predict", help="Prediction endpoint URL")
    parser.add_argument("--data", help="JSON payload string with house features; if omitted, a default example is used")
    args = parser.parse_args()

    if args.data:
        try:
            payload = json.loads(args.data)
        except json.JSONDecodeError:
            print("--data must be a valid JSON string", file=sys.stderr)
            sys.exit(1)
    else:
        payload = DEFAULT_PAYLOAD

    resp = requests.post(args.url, json=payload, timeout=10)
    if resp.status_code != 200:
        print(f"Error {resp.status_code}: {resp.text}", file=sys.stderr)
        sys.exit(1)

    print(resp.json())


if __name__ == "__main__":
    main()
