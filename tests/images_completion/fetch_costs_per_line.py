import os
import time
import requests
import csv
from datetime import datetime, timedelta, timezone


# ---- CONFIGURATION ----
# Set this in your environment or replace directly
OPENAI_API_KEY = os.getenv("OPENAI_ADMIN_KEY")
BASE_URL = "https://api.openai.com/v1/organization/costs"
DAYS_BACK = 1  # Adjust as needed
OUTPUT_FILE = "daily_costs_by_line_item.csv"

# ---- TIME WINDOW ----
# now = int(time.time())
# start_time = int(
#     (datetime.now(timezone.utc) - timedelta(days=DAYS_BACK))
#     .replace(hour=0, minute=0, second=0, microsecond=0)
#     .timestamp()
# )

# Start of today (UTC)
start_time = int(
    datetime.now(timezone.utc)
    .replace(hour=0, minute=0, second=0, microsecond=0)
    .timestamp()
)

# Now (UTC)
# end_time = int(datetime.now(timezone.utc).timestamp())
end_time = int(
    (
        datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        + timedelta(days=1)
    ).timestamp()
)

# ---- QUERY PARAMS ----
params = {
    "start_time": start_time,
    "end_time": end_time,
    "bucket_width": "1d",
    "group_by": ["line_item"],
    "limit": DAYS_BACK,
}

headers = {
    "Authorization": f"Bearer {OPENAI_API_KEY}",
    "Content-Type": "application/json"
}

# ---- API CALL ----
response = requests.get(BASE_URL, headers=headers, params=params)
response.raise_for_status()
data = response.json()

# ---- SAVE TO CSV ----
with open(OUTPUT_FILE, mode="w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["date", "line_item", "amount", "currency"])

    for bucket in data.get("data", []):
        date_str = datetime.utcfromtimestamp(bucket["start_time"]).strftime("%Y-%m-%d")
        for result in bucket.get("results", []):
            line_item = result.get("line_item") or "unspecified"
            amount = result["amount"]["value"]
            currency = result["amount"]["currency"]
            writer.writerow([date_str, line_item, amount, currency])

print(f"[✓] Saved daily costs to '{OUTPUT_FILE}'")
