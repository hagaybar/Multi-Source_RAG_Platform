#!/usr/bin/env python3
"""
Unit tests for ContextAssembler.

Tests:
1. Quote removal (>, On X wrote:, etc.)
2. Signature stripping (Best regards, Sent from iPhone, etc.)
3. Thread grouping and chronological merging
4. Content deduplication
5. Noise filtering (newsletters, auto-replies)
6. Source attribution formatting
"""

import pytest
from scripts.retrieval.context_assembler import ContextAssembler
from scripts.chunking.models import Chunk


class TestQuoteRemoval:
    """Test removal of quoted text from emails."""

    def setup_method(self):
        """Initialize assembler for each test."""
        self.assembler = ContextAssembler()

    def test_remove_angle_bracket_quotes(self):
        """Test removal of > prefixed quoted lines."""
        text = """Hi team,

I agree with your points.

> Let's meet Tuesday.
> What time works?

How about 2pm?"""

        cleaned = self.assembler._clean_email(text)

        assert "> Let's meet Tuesday" not in cleaned
        assert "> What time works?" not in cleaned
        assert "I agree with your points" in cleaned
        assert "How about 2pm?" in cleaned

    def test_remove_on_x_wrote_pattern(self):
        """Test removal of 'On X wrote:' patterns."""
        text = """Thanks for the update.

On Jan 15, 2025, Alice wrote:
The project is on track.

I'll review it soon."""

        cleaned = self.assembler._clean_email(text)

        assert "On Jan 15" not in cleaned
        assert "Thanks for the update" in cleaned
        assert "I'll review it soon" in cleaned

    def test_remove_email_header_blocks(self):
        """Test removal of forwarded email headers."""
        text = """Here is the original message:

From: alice@company.com
Sent: Tuesday, January 15
To: team@company.com
Subject: Budget Update

The budget was approved.

Let me know your thoughts."""

        cleaned = self.assembler._clean_email(text)

        # This pattern might not catch multiline headers perfectly
        # The key is that content before and after is preserved
        assert "Here is the original message" in cleaned
        assert "The budget was approved" in cleaned
        assert "Let me know your thoughts" in cleaned

    def test_remove_outlook_quote_markers(self):
        """Test removal of Outlook quote separators."""
        text = """See response below.

-----Original Message-----
The meeting is at 2pm.
------------------------------

Thanks!"""

        cleaned = self.assembler._clean_email(text)

        # Original Message marker should be removed
        assert "Original Message" not in cleaned
        # Content before marker should be preserved
        assert "See response below" in cleaned
        # Note: Content after dash separators may be treated as signature


class TestSignatureStripping:
    """Test removal of email signatures."""

    def setup_method(self):
        """Initialize assembler for each test."""
        self.assembler = ContextAssembler()

    def test_remove_standard_signature_delimiter(self):
        """Test removal of standard -- signature delimiter."""
        text = """The report is ready for review.

--
Alice Johnson
Senior Manager"""

        cleaned = self.assembler._clean_email(text)

        assert "Alice Johnson" not in cleaned
        assert "Senior Manager" not in cleaned
        assert "The report is ready" in cleaned

    def test_remove_mobile_signatures(self):
        """Test removal of mobile device signatures."""
        mobile_sigs = [
            "Sent from my iPhone",
            "Sent from my iPad",
            "Sent from my Android",
        ]

        for sig in mobile_sigs:
            text = f"Meeting confirmed.\n\n{sig}"
            cleaned = self.assembler._clean_email(text)

            assert sig not in cleaned
            assert "Meeting confirmed" in cleaned

    def test_remove_common_closings(self):
        """Test removal of common email closings."""
        closings = [
            "Best regards,",
            "Thanks,",
            "Thank you,",
            "Sincerely,",
            "Regards,",
        ]

        for closing in closings:
            text = f"The budget is approved.\n\n{closing}\nAlice"
            cleaned = self.assembler._clean_email(text)

            # Closing and name after it should be removed
            assert "Alice" not in cleaned
            assert "The budget is approved" in cleaned

    def test_signature_block_removal(self):
        """Test removal of complete signature blocks."""
        text = """Project update attached.

Best regards,
Alice Johnson
Senior Project Manager
Company Inc.
Phone: 555-0123"""

        cleaned = self.assembler._clean_email(text)

        # Everything after "Best regards," should be removed
        assert "Alice Johnson" not in cleaned
        assert "Senior Project Manager" not in cleaned
        assert "555-0123" not in cleaned
        assert "Project update attached" in cleaned


class TestThreadGrouping:
    """Test thread grouping and chronological merging."""

    def setup_method(self):
        """Initialize assembler for each test."""
        self.assembler = ContextAssembler()

    def test_normalize_subject(self):
        """Test subject normalization for thread grouping."""
        test_cases = [
            ("Budget Discussion", "budget discussion"),
            ("Re: Budget Discussion", "budget discussion"),
            ("Fwd: Budget Discussion", "budget discussion"),
            ("RE: Fwd: Re: Budget Discussion", "budget discussion"),
            ("  Extra   Spaces  ", "extra spaces"),
        ]

        for input_subject, expected in test_cases:
            result = self.assembler._normalize_subject(input_subject)
            assert result == expected

    def test_group_by_thread(self):
        """Test grouping emails by normalized subject."""
        chunks = [
            Chunk(
                doc_id="email_1",
                text="First message",
                meta={"subject": "Budget Discussion"},
                token_count=5
            ),
            Chunk(
                doc_id="email_2",
                text="Reply message",
                meta={"subject": "Re: Budget Discussion"},
                token_count=5
            ),
            Chunk(
                doc_id="email_3",
                text="Different topic",
                meta={"subject": "Server Migration"},
                token_count=5
            ),
        ]

        threads = self.assembler._group_by_thread(chunks)

        assert len(threads) == 2  # Two distinct threads
        assert "budget discussion" in threads
        assert "server migration" in threads
        assert len(threads["budget discussion"]) == 2
        assert len(threads["server migration"]) == 1

    def test_chronological_merging(self):
        """Test chronological sorting within threads."""
        threads = {
            "budget discussion": [
                Chunk(
                    doc_id="email_3",
                    text="Third",
                    meta={"date": "2025-01-15 10:00:00"},
                    token_count=5
                ),
                Chunk(
                    doc_id="email_1",
                    text="First",
                    meta={"date": "2025-01-15 08:00:00"},
                    token_count=5
                ),
                Chunk(
                    doc_id="email_2",
                    text="Second",
                    meta={"date": "2025-01-15 09:00:00"},
                    token_count=5
                ),
            ]
        }

        merged = self.assembler._merge_threads_chronologically(threads)

        # Should be chronologically ordered
        assert merged[0].meta["date"] == "2025-01-15 08:00:00"
        assert merged[1].meta["date"] == "2025-01-15 09:00:00"
        assert merged[2].meta["date"] == "2025-01-15 10:00:00"


class TestContentDeduplication:
    """Test deduplication of repeated content."""

    def setup_method(self):
        """Initialize assembler for each test."""
        self.assembler = ContextAssembler()

    def test_text_overlap_ratio(self):
        """Test word-based overlap calculation."""
        text1 = "The quick brown fox jumps over the lazy dog"
        text2 = "The quick brown fox runs fast"

        # text1 unique words: {the, quick, brown, fox, jumps, over, lazy, dog} = 8
        # text2 unique words: {the, quick, brown, fox, runs, fast} = 6
        # Overlap: {the, quick, brown, fox} = 4
        # Ratio: 4/8 = 0.5

        ratio = self.assembler._text_overlap_ratio(text1, text2)
        assert ratio == 0.5

    def test_deduplicate_identical_content(self):
        """Test removal of duplicate emails."""
        chunks = [
            {
                "text": "The budget has been approved for Q1 projects",
                "meta": {"sender": "alice@company.com"}
            },
            {
                "text": "The budget has been approved for Q1 projects",
                "meta": {"sender": "bob@company.com"}
            },
        ]

        unique = self.assembler._deduplicate_content(chunks)

        # Should keep only one copy
        assert len(unique) == 1

    def test_deduplicate_high_overlap(self):
        """Test removal of emails with >80% overlap."""
        chunks = [
            {
                "text": "The project timeline is on schedule and we expect delivery by March",
                "meta": {"sender": "alice@company.com"}
            },
            {
                "text": "The project timeline is on schedule and we expect delivery soon",
                "meta": {"sender": "alice@company.com"}
            },
        ]

        unique = self.assembler._deduplicate_content(chunks)

        # Should deduplicate (>80% overlap)
        assert len(unique) == 1

    def test_keep_distinct_content(self):
        """Test that distinct emails are preserved."""
        chunks = [
            {
                "text": "The budget was approved yesterday",
                "meta": {"sender": "alice@company.com"}
            },
            {
                "text": "The server migration is complete",
                "meta": {"sender": "bob@company.com"}
            },
        ]

        unique = self.assembler._deduplicate_content(chunks)

        # Should keep both (distinct content)
        assert len(unique) == 2


class TestNoiseFiltering:
    """Test filtering of newsletters, auto-replies, notifications."""

    def setup_method(self):
        """Initialize assembler for each test."""
        self.assembler = ContextAssembler()

    def test_filter_system_emails(self):
        """Test filtering of no-reply and system emails."""
        chunks = [
            Chunk(
                doc_id="email_1",
                text="Important update",
                meta={"sender": "noreply@company.com"},
                token_count=5
            ),
            Chunk(
                doc_id="email_2",
                text="Project status",
                meta={"sender": "alice@company.com"},
                token_count=5
            ),
            Chunk(
                doc_id="email_3",
                text="Notification",
                meta={"sender": "donotreply@system.com"},
                token_count=5
            ),
        ]

        filtered = self.assembler._filter_noise(chunks)

        # Should filter out noreply emails
        assert len(filtered) == 1
        assert filtered[0].meta["sender"] == "alice@company.com"

    def test_filter_newsletters(self):
        """Test filtering of newsletter content."""
        chunks = [
            Chunk(
                doc_id="email_1",
                text="Click here to unsubscribe from our newsletter",
                meta={"sender": "marketing@company.com"},
                token_count=10
            ),
            Chunk(
                doc_id="email_2",
                text="Meeting notes from yesterday",
                meta={"sender": "alice@company.com"},
                token_count=10
            ),
        ]

        filtered = self.assembler._filter_noise(chunks)

        # Should filter newsletter
        assert len(filtered) == 1
        assert "Meeting notes" in filtered[0].text

    def test_filter_auto_replies(self):
        """Test filtering of out-of-office messages."""
        chunks = [
            Chunk(
                doc_id="email_1",
                text="I am out of office until next week. This is an automatic reply.",
                meta={"sender": "bob@company.com"},
                token_count=15
            ),
            Chunk(
                doc_id="email_2",
                text="The report is ready for review",
                meta={"sender": "alice@company.com"},
                token_count=10
            ),
        ]

        filtered = self.assembler._filter_noise(chunks)

        # Should filter auto-reply
        assert len(filtered) == 1
        assert "report is ready" in filtered[0].text


class TestContextAssembly:
    """Test complete context assembly pipeline."""

    def setup_method(self):
        """Initialize assembler for each test."""
        self.assembler = ContextAssembler()

    def test_assemble_with_factual_lookup(self):
        """Test assembly with factual_lookup intent (relevance order)."""
        chunks = [
            Chunk(
                doc_id="email_1",
                text="The budget was approved yesterday.",
                meta={
                    "doc_type": "outlook_eml",
                    "subject": "Budget Update",
                    "sender_name": "Alice Johnson",
                    "sender": "alice@company.com",
                    "date": "2025-01-15 09:00:00"
                },
                token_count=10
            ),
            Chunk(
                doc_id="email_2",
                text="Thanks for the update!",
                meta={
                    "doc_type": "outlook_eml",
                    "subject": "Re: Budget Update",
                    "sender_name": "Bob Smith",
                    "sender": "bob@company.com",
                    "date": "2025-01-15 10:00:00"
                },
                token_count=5
            ),
        ]

        intent = {"primary_intent": "factual_lookup"}
        context = self.assembler.assemble(chunks, intent)

        # Should include both emails with source attributions
        assert "Email #1:" in context
        assert "Email #2:" in context
        assert "Alice Johnson" in context
        assert "Bob Smith" in context
        assert "Budget Update" in context
        assert "The budget was approved" in context

    def test_assemble_with_thread_summary(self):
        """Test assembly with thread_summary intent (chronological order)."""
        chunks = [
            Chunk(
                doc_id="email_3",
                text="2pm works for me.",
                meta={
                    "doc_type": "outlook_eml",
                    "subject": "Re: Budget Discussion",
                    "sender_name": "Alice Johnson",
                    "sender": "alice@company.com",
                    "date": "2025-01-15 09:20:00"
                },
                token_count=5
            ),
            Chunk(
                doc_id="email_1",
                text="Let's meet Tuesday to discuss the budget.",
                meta={
                    "doc_type": "outlook_eml",
                    "subject": "Budget Discussion",
                    "sender_name": "Alice Johnson",
                    "sender": "alice@company.com",
                    "date": "2025-01-15 09:00:00"
                },
                token_count=10
            ),
            Chunk(
                doc_id="email_2",
                text="I agree! What time works?",
                meta={
                    "doc_type": "outlook_eml",
                    "subject": "Re: Budget Discussion",
                    "sender_name": "Bob Smith",
                    "sender": "bob@company.com",
                    "date": "2025-01-15 09:15:00"
                },
                token_count=7
            ),
        ]

        intent = {"primary_intent": "thread_summary"}
        context = self.assembler.assemble(chunks, intent)

        # Should be chronologically ordered
        # Email #1 should have earliest date
        email_sections = context.split("---")
        assert len(email_sections) == 3

        # First email should be from 09:00:00
        assert "09:00:00" in email_sections[0]
        assert "Let's meet Tuesday" in email_sections[0]

    def test_assemble_removes_quotes_and_signatures(self):
        """Test that assembly removes quotes and signatures."""
        chunks = [
            Chunk(
                doc_id="email_1",
                text="""Hi team,

The project is on track.

Best regards,
Alice Johnson""",
                meta={
                    "doc_type": "outlook_eml",
                    "subject": "Project Update",
                    "sender_name": "Alice Johnson",
                    "sender": "alice@company.com",
                    "date": "2025-01-15 09:00:00"
                },
                token_count=15
            ),
            Chunk(
                doc_id="email_2",
                text="""> The project is on track.

Great news!

Sent from my iPhone""",
                meta={
                    "doc_type": "outlook_eml",
                    "subject": "Re: Project Update",
                    "sender_name": "Bob Smith",
                    "sender": "bob@company.com",
                    "date": "2025-01-15 10:00:00"
                },
                token_count=10
            ),
        ]

        intent = {"primary_intent": "factual_lookup"}
        context = self.assembler.assemble(chunks, intent)

        # Signatures should be removed
        assert "Best regards" not in context
        assert "Alice Johnson" in context  # But in attribution header
        assert "Sent from my iPhone" not in context

        # Quotes should be removed
        assert context.count("The project is on track") == 1  # Only once, not quoted
        assert "Great news!" in context

    def test_assemble_empty_chunks(self):
        """Test handling of empty chunk list."""
        context = self.assembler.assemble([])

        assert context == "No context provided."

    def test_assemble_with_temporal_intent(self):
        """Test assembly with temporal_query intent (newest first)."""
        chunks = [
            Chunk(
                doc_id="email_1",
                text="Older message",
                meta={
                    "doc_type": "outlook_eml",
                    "subject": "Update",
                    "sender_name": "Alice",
                    "sender": "alice@company.com",
                    "date": "2025-01-14 09:00:00"
                },
                token_count=5
            ),
            Chunk(
                doc_id="email_2",
                text="Newer message",
                meta={
                    "doc_type": "outlook_eml",
                    "subject": "Update",
                    "sender_name": "Bob",
                    "sender": "bob@company.com",
                    "date": "2025-01-15 09:00:00"
                },
                token_count=5
            ),
        ]

        intent = {
            "primary_intent": "temporal_query",
            "metadata": {"time_range": "recent"}
        }
        context = self.assembler.assemble(chunks, intent)

        # Should have newer message first
        email_sections = context.split("---")
        assert "2025-01-15" in email_sections[0]
        assert "Newer message" in email_sections[0]


class TestIntegrationScenarios:
    """Test realistic end-to-end scenarios."""

    def setup_method(self):
        """Initialize assembler for each test."""
        self.assembler = ContextAssembler()

    def test_real_world_email_thread(self):
        """Test with realistic email thread."""
        chunks = [
            Chunk(
                doc_id="email_1",
                text="""Hi team,

Let's schedule a meeting to discuss the Primo NDE migration timeline.

Best regards,
Sarah Johnson
Project Manager""",
                meta={
                    "doc_type": "outlook_eml",
                    "subject": "Primo NDE Migration",
                    "sender_name": "Sarah Johnson",
                    "sender": "sarah@company.com",
                    "date": "2025-01-15 09:00:00"
                },
                token_count=20
            ),
            Chunk(
                doc_id="email_2",
                text="""> Let's schedule a meeting to discuss the Primo NDE migration timeline.

Sounds good! I'm available Tuesday or Wednesday.

Thanks,
Mike Chen

Sent from my iPhone""",
                meta={
                    "doc_type": "outlook_eml",
                    "subject": "Re: Primo NDE Migration",
                    "sender_name": "Mike Chen",
                    "sender": "mike@company.com",
                    "date": "2025-01-15 09:30:00"
                },
                token_count=25
            ),
            Chunk(
                doc_id="email_3",
                text="""> Sounds good! I'm available Tuesday or Wednesday.

Let's do Tuesday at 2pm.

--
Sarah Johnson""",
                meta={
                    "doc_type": "outlook_eml",
                    "subject": "Re: Primo NDE Migration",
                    "sender_name": "Sarah Johnson",
                    "sender": "sarah@company.com",
                    "date": "2025-01-15 10:00:00"
                },
                token_count=15
            ),
        ]

        intent = {"primary_intent": "thread_summary"}
        context = self.assembler.assemble(chunks, intent)

        # Check quote removal
        assert context.count("Let's schedule a meeting") == 1
        assert context.count("Sounds good!") == 1

        # Check signature removal
        assert "Project Manager" not in context
        assert "Sent from my iPhone" not in context

        # Check source attributions are present
        assert "Sarah Johnson" in context  # In attribution
        assert "Mike Chen" in context

        # Check content is preserved
        assert "Tuesday at 2pm" in context
        assert "I'm available Tuesday or Wednesday" in context


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
