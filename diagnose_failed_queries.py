"""
Diagnostic script to investigate why queries returned 0 chunks.

Checks:
1. Email date range in database
2. Emails from "last week" relative to test date
3. Emails from sender "Sarah"
4. Why "recent" worked but "last week" didn't
"""

import json
import sys
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict

# Paths
project_root = Path(__file__).parent
metadata_file = project_root / "data/projects/Primo_List_2/output/metadata/outlook_eml_metadata.jsonl"

print("=" * 80)
print("DIAGNOSTIC: Failed Query Investigation")
print("=" * 80)

# Test run date (from test results)
TEST_DATE = datetime(2025, 11, 25)
LAST_WEEK_START = TEST_DATE - timedelta(days=7)

print(f"\nTest Run Date: {TEST_DATE.strftime('%Y-%m-%d')}")
print(f"Last Week Range: {LAST_WEEK_START.strftime('%Y-%m-%d')} to {TEST_DATE.strftime('%Y-%m-%d')}")

# Read metadata
print(f"\nReading metadata from: {metadata_file}")
emails = []
with open(metadata_file, 'r', encoding='utf-8') as f:
    for line in f:
        emails.append(json.loads(line))

print(f"Total email chunks: {len(emails):,}")

# Analysis 1: Date range
print("\n" + "=" * 80)
print("ANALYSIS 1: Email Date Range")
print("=" * 80)

dates = []
for email in emails:
    if 'date' in email and email['date']:
        try:
            # Parse date (format: "YYYY-MM-DD HH:MM:SS" or ISO format)
            date_str = email['date']
            if 'T' in date_str:
                dt = datetime.fromisoformat(date_str.replace('Z', '+00:00').split('+')[0])
            else:
                dt = datetime.strptime(date_str.split(' ')[0], '%Y-%m-%d')
            dates.append(dt)
        except:
            pass

if dates:
    min_date = min(dates)
    max_date = max(dates)
    print(f"\nEarliest email: {min_date.strftime('%Y-%m-%d')}")
    print(f"Latest email: {max_date.strftime('%Y-%m-%d')}")
    print(f"Date range span: {(max_date - min_date).days} days")

    # Check if we have recent data
    days_since_latest = (TEST_DATE - max_date).days
    print(f"\nDays between latest email and test date: {days_since_latest} days")

    if days_since_latest > 7:
        print(f"⚠️  WARNING: Latest email is {days_since_latest} days old!")
        print(f"   'Last week' filter would find NO emails (last email: {max_date.strftime('%Y-%m-%d')})")
else:
    print("⚠️  No valid dates found in metadata!")

# Analysis 2: Emails in "last week"
print("\n" + "=" * 80)
print("ANALYSIS 2: Emails from 'Last Week'")
print("=" * 80)

last_week_emails = []
for email in emails:
    if 'date' in email and email['date']:
        try:
            date_str = email['date']
            if 'T' in date_str:
                dt = datetime.fromisoformat(date_str.replace('Z', '+00:00').split('+')[0])
            else:
                dt = datetime.strptime(date_str.split(' ')[0], '%Y-%m-%d')

            if LAST_WEEK_START <= dt <= TEST_DATE:
                last_week_emails.append(email)
        except:
            pass

print(f"\nEmails in last week: {len(last_week_emails)}")

if last_week_emails:
    print("\nSample emails from last week:")
    for email in last_week_emails[:3]:
        print(f"  - Date: {email.get('date', 'N/A')}, Subject: {email.get('subject', 'N/A')[:60]}...")
else:
    print("⚠️  NO emails found in last week range!")
    print(f"   This explains why 'last week' query returned 0 chunks")

# Analysis 3: Emails from "Sarah"
print("\n" + "=" * 80)
print("ANALYSIS 3: Emails from Sender 'Sarah'")
print("=" * 80)

sarah_emails = []
for email in emails:
    sender = email.get('sender', '') or email.get('from', '')
    if sender and 'sarah' in sender.lower():
        sarah_emails.append(email)

print(f"\nEmails from 'Sarah': {len(sarah_emails)}")

if sarah_emails:
    print("\nSample emails from Sarah:")
    for email in sarah_emails[:3]:
        print(f"  - From: {email.get('sender', email.get('from', 'N/A'))}")
        print(f"    Subject: {email.get('subject', 'N/A')[:60]}...")
else:
    print("⚠️  NO emails found from sender 'Sarah'!")
    print(f"   This explains why 'Sarah' query returned 0 chunks")

# Analysis 4: "Research Assistant" mentions
print("\n" + "=" * 80)
print("ANALYSIS 4: 'Research Assistant' Mentions")
print("=" * 80)

research_assistant_emails = []
for email in emails:
    text = email.get('text', '')
    subject = email.get('subject', '')
    if text and ('research assistant' in text.lower() or 'research assistant' in subject.lower()):
        research_assistant_emails.append(email)

print(f"\nEmails mentioning 'Research Assistant': {len(research_assistant_emails)}")

if research_assistant_emails:
    print("\nSample emails:")
    for email in research_assistant_emails[:3]:
        print(f"  - From: {email.get('sender', email.get('from', 'N/A'))}")
        print(f"    Subject: {email.get('subject', 'N/A')[:60]}...")
else:
    print("⚠️  NO emails mention 'Research Assistant'!")

# Analysis 5: Successful query comparison
print("\n" + "=" * 80)
print("ANALYSIS 5: Why 'recent conversations' Worked vs 'last week' Failed")
print("=" * 80)

# Check last 30 days
recent_start = TEST_DATE - timedelta(days=30)
recent_emails = []
for email in emails:
    if 'date' in email and email['date']:
        try:
            date_str = email['date']
            if 'T' in date_str:
                dt = datetime.fromisoformat(date_str.replace('Z', '+00:00').split('+')[0])
            else:
                dt = datetime.strptime(date_str.split(' ')[0], '%Y-%m-%d')

            if recent_start <= dt <= TEST_DATE:
                recent_emails.append(email)
        except:
            pass

print(f"\nEmails in last 30 days ('recent'): {len(recent_emails)}")
print(f"Emails in last 7 days ('last week'): {len(last_week_emails)}")

if len(recent_emails) > 0 and len(last_week_emails) == 0:
    print("\n💡 INSIGHT: 'Recent' (30 days) finds emails, but 'last week' (7 days) doesn't!")
    print("   This means email data is older than 7 days from test date.")

# Summary
print("\n" + "=" * 80)
print("SUMMARY & RECOMMENDATIONS")
print("=" * 80)

issues_found = []

if len(last_week_emails) == 0:
    issues_found.append("No emails in 'last week' range")

if len(sarah_emails) == 0:
    issues_found.append("No emails from sender 'Sarah'")

if len(research_assistant_emails) == 0:
    issues_found.append("No emails mentioning 'Research Assistant'")

if issues_found:
    print("\n⚠️  Issues Found:")
    for issue in issues_found:
        print(f"   - {issue}")

    print("\n📋 Recommendations:")
    print("   1. Test queries should match actual data availability")
    print("   2. Use 'recent' (30+ days) instead of 'last week' for temporal queries")
    print("   3. Use actual sender names from the database")
    print("   4. Verify topic keywords exist in email content")
else:
    print("\n✅ All test queries should have found results!")
    print("   Need to investigate retrieval logic.")

print("\n" + "=" * 80)
