import pytest
from scripts.chunking import chunker_v3
from scripts.chunking.models import Chunk
from scripts.chunking.rules_v3 import get_rule
from scripts.utils.email_utils import clean_email_text


def count_tokens(text: str) -> int:
    return len(text.split())


def test_chunker_eml_by_email_block():
    raw_email = """
Hi team,

Just a quick reminder that all timesheets must be submitted by Friday at the 
end of business day. Please take the time to carefully review your entries to 
ensure all hours worked, project codes, and client billing information are 
accurate and complete before submission. Late or incomplete timesheets can 
delay payroll processing and affect your pay schedule, so timely submission 
is crucial for everyone.
If you encounter any technical difficulties with the timesheet system, have 
questions about proper coding procedures, or need clarification on billable 
versus non-billable hours, please don't hesitate to reach out to the HR 
department immediately. Our team is available to assist you and ensure your 
timesheet is processed correctly.
We truly appreciate your cooperation and commitment to meeting these 
important deadlines consistently.

Thanks,
Alice

On Mon, Bob wrote:
> Hi Alice,
> Thanks for the update. I'll get it done by EOD. Let me know if you need 
> anything else.
> Best regards,
> Bob

From: carol@example.com
Subject: RE: Timesheets
> Absolutely. I'm syncing with my manager now. I'll have everything ready 
> by tomorrow morning.
> Regards,
> Carol
"""

    meta = {
        "doc_type": "eml",
        "content_type": "email",
        "sender": "alice@example.com",
        "subject": "Timesheet Reminder",
    }

    rule = get_rule("eml")
    cleaned_text = clean_email_text(raw_email)
    token_count = count_tokens(cleaned_text)
    chunks: list[Chunk] = chunker_v3.split(raw_email, meta)

    # --- Basic structure ---
    assert isinstance(chunks, list)
    assert all(isinstance(c, Chunk) for c in chunks)

    # --- Behavior depends on total token count ---
    if token_count <= rule.max_tokens:
        assert len(chunks) == 1, (
            "Should produce a single chunk if under max token limit"
        )
    else:
        assert len(chunks) >= 2, (
            "Should produce multiple chunks if over max token limit"
        )

        # --- Token bounds and overlap ---
        for i, c in enumerate(chunks):
            assert c.token_count <= rule.max_tokens + 20
            if i < len(chunks) - 1:
                assert c.token_count >= rule.min_tokens or i == 0
            assert c.meta["doc_type"] == "eml"

        if rule.overlap and len(chunks) >= 2:
            words1 = chunks[0].text.split()
            words2 = chunks[1].text.split()
            overlap = rule.overlap
            if len(words1) >= overlap and len(words2) >= overlap:
                assert words1[-overlap:] == words2[:overlap], (
                    "Overlap mismatch"
                )

    # Optional: Debug output
    print(f"\nCleaned text token count: {token_count}")
    print(f"Chunks produced: {len(chunks)}")
    for i, chunk in enumerate(chunks):
        print(
            f"\n--- Chunk {i + 1} ({chunk.token_count} tokens) ---\n"
            f"{chunk.text}"
        )
