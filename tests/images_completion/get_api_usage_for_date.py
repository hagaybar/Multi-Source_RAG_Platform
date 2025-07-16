import os
import requests
import argparse
from datetime import datetime, timedelta

def get_daily_usage(api_key, target_date):
    """
    Fetches and displays daily OpenAI API usage for a specific date.

    Args:
        api_key (str): Your OpenAI API key.
        target_date (str): The date to fetch usage for, in 'YYYY-MM-DD' format.
    """
    api_url = "https://api.openai.com/v1/usage"
    headers = {
        "Authorization": f"Bearer {api_key}"
    }
    params = {
        "date": target_date
    }

    print(f"Fetching usage data for: {target_date}...\n")

    try:
        response = requests.get(api_url, headers=headers, params=params)
        response.raise_for_status()  # Raises an HTTPError for bad responses (4xx or 5xx)

        data = response.json()
        print("Raw response:")
        print(data)

        if not data.get('daily_costs'):
            print("No usage data found for this date.")
            return

        # --- Display Daily Cost Breakdown ---
        print("--- Daily Cost Breakdown ---")
        total_daily_cost = 0
        for daily_cost in data.get('daily_costs', []):
            for item in daily_cost.get('line_items', []):
                model_name = item.get('name')
                cost = item.get('cost') / 100 # Cost is returned in cents
                total_daily_cost += cost
                print(f"  - Model: {model_name:<20} | Cost: ${cost:.4f}")
        
        print("---------------------------------")
        print(f"Total cost for {target_date}: ${total_daily_cost:.4f}")
        print("---------------------------------\n")


        # --- Display Total Usage for Current Billing Period ---
        if 'total_usage' in data:
            # Total usage is returned in cents
            total_billing_usage = data['total_usage'] / 100
            print("--- Billing Period Summary ---")
            print(f"Total usage for the current billing period: ${total_billing_usage:.2f}")
            print("----------------------------")

    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
        print(f"Response Body: {response.text}")
    except requests.exceptions.RequestException as req_err:
        print(f"A request error occurred: {req_err}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    # --- Set up argument parser ---
    parser = argparse.ArgumentParser(description="Fetch daily OpenAI API usage.")
    parser.add_argument(
        "--date",
        type=str,
        help="The date to fetch usage for in YYYY-MM-DD format. Defaults to yesterday."
    )
    args = parser.parse_args()

    # --- Determine the target date ---
    if args.date:
        try:
            # Validate date format
            datetime.strptime(args.date, '%Y-%m-%d')
            date_to_check = args.date
        except ValueError:
            print("Error: Date must be in YYYY-MM-DD format.")
            exit(1)
    else:
        # Default to yesterday's date
        yesterday = datetime.now() - timedelta(days=1)
        date_to_check = yesterday.strftime('%Y-%m-%d')

    # --- Get API key from environment variable ---
    # It's more secure to load the key from an environment variable
    # than to hardcode it in the script.
    openai_api_key = os.getenv("OPENAI_API_KEY")

    if not openai_api_key:
        print("Error: The OPENAI_API_KEY environment variable is not set.")
        print("Please set it before running the script.")
        print("Example (Linux/macOS): export OPENAI_API_KEY='your_key_here'")
        print("Example (Windows): set OPENAI_API_KEY=your_key_here")
        exit(1)

    get_daily_usage(openai_api_key, date_to_check)
