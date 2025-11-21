# Email-Specific Prompting - Analysis & Proposal

**Date:** 2025-01-19
**Status:** Proposal
**Context:** Outlook integration milestone - preparing for retrieve/ask steps

---

## Current System Analysis

### Current Prompt Template (DEFAULT_PROMPT_TEMPLATE_V2)

**Domain:** Library systems (Alma, Primo, SAML authentication, APIs)

**Strengths:**
- Clear step-by-step structure
- Citations with source IDs
- Multilingual support (Hebrew/English)
- Focuses on "how-to" guidance

**Limitations for Emails:**
1. ❌ **Domain mismatch**: Template assumes library tech troubleshooting, not email search
2. ❌ **Missing email metadata**: Context doesn't include subject, sender, date
3. ❌ **Format assumption**: Expects procedural guides, not conversational email content
4. ❌ **No temporal awareness**: Can't handle "recent emails" or date-based queries
5. ❌ **No sender/recipient context**: Critical for email search ("emails from John about X")

### Current Context Builder (prompt_builder.py:157-162)

```python
context_item = f"Source ID: [{source_id_str}]\nContent: {text}"

# Only adds page_number for documents
page_number = chunk.meta.get('page_number')
if page_number:
    context_item += f"\nPage: {page_number}"
```

**What's Missing for Emails:**
- Subject line
- Sender name/email
- Recipient(s)
- Date/time sent
- Message index (for multi-email files)

---

## Email-Specific Requirements

### 1. **Metadata Requirements**

Email chunks have rich metadata that should be exposed in prompts:

```python
chunk.meta = {
    "doc_type": "outlook_eml",
    "source_filepath": "outlook://account/folder",
    "content_type": "email",
    "subject": "Re: Project Update",         # ← CRITICAL
    "sender": "john@company.com",            # ← CRITICAL
    "sender_name": "John Smith",             # ← CRITICAL
    "date": "2025-01-15 14:30:00",          # ← CRITICAL
    "message_id": "outlook_msg_123",
    "message_index": 5
}
```

### 2. **Common Email Search Queries**

**Temporal Queries:**
- "What were the latest updates on the Primo project?"
- "Show me emails from last week about the budget"
- "Recent conversations about API integration"

**Sender-Based Queries:**
- "What did Sarah say about the migration?"
- "Find emails from the IT department about security"
- "Has anyone mentioned the deadline?"

**Topic-Based Queries:**
- "Summarize the discussion about vendor selection"
- "What issues were raised in the team meeting emails?"
- "Find action items from project emails"

**Combined Queries:**
- "What did John say last month about authentication?"
- "Recent emails from management about budget cuts"
- "Find Sarah's response to the deployment question"

### 3. **Email-Specific Use Cases**

Unlike document search, email search needs:

1. **Conversation Reconstruction**
   - Thread awareness (Re: chains)
   - Back-and-forth context
   - Quote/reply handling

2. **Action Item Extraction**
   - "Can you summarize action items from these emails?"
   - "What deadlines were mentioned?"
   - "Who is responsible for what?"

3. **Sentiment & Tone**
   - "Were there any concerns raised?"
   - "What was the general sentiment about the proposal?"

4. **Information Tracking**
   - "When did we first discuss this topic?"
   - "How has the plan evolved over time?"
   - "What was the final decision?"

---

## Proposal: Email-Optimized Prompting

### Option 1: Email-Specific Template (Recommended)

Create a new template specifically for email content:

```python
EMAIL_PROMPT_TEMPLATE = """
You are an intelligent email assistant helping users search and understand their email communications.

Your job is to answer questions based ONLY on the provided email conversations.
If the emails do not contain the answer, clearly state that.

When answering:
1. **Identify relevant emails** by subject, sender, and date
2. **Summarize key points** from the email content
3. **Maintain context** - consider the conversation flow if multiple emails are related
4. **Extract actionable information** - deadlines, action items, decisions, concerns
5. **Cite sources** using email metadata: [Sender Name, Subject, Date]

If the user asks about:
- **"Recent" or "latest"**: Prioritize emails by date (most recent first)
- **Specific person**: Focus on emails from/to that sender
- **Action items**: Look for tasks, deadlines, "TODO", "please", "need to"
- **Decisions**: Look for conclusive statements, "decided", "agreed", "final"

---

Context (Email Excerpts):
{context_str}

---

User Question:
{query_str}

---

Answer:
"""
```

### Option 2: Enhanced Context Builder

Modify `build_prompt()` to detect email chunks and include rich metadata:

```python
def build_prompt(self, query: str, context_chunks: List[Chunk]) -> str:
    """Enhanced to include email metadata in context."""
    context_items = []

    for i, chunk in enumerate(context_chunks):
        doc_type = chunk.meta.get("doc_type", "")

        if doc_type == "outlook_eml":
            # Email-specific context formatting
            subject = chunk.meta.get("subject", "No Subject")
            sender_name = chunk.meta.get("sender_name", "Unknown")
            sender_email = chunk.meta.get("sender", "")
            date = chunk.meta.get("date", "Unknown Date")

            context_item = f"""
Email #{i+1}:
From: {sender_name} <{sender_email}>
Subject: {subject}
Date: {date}

Content:
{text}
            """.strip()
        else:
            # Document-specific context (existing logic)
            context_item = f"Source ID: [{source_id_str}]\nContent: {text}"
            page_number = chunk.meta.get('page_number')
            if page_number:
                context_item += f"\nPage: {page_number}"

        context_items.append(context_item)

    context_str = "\n\n---\n\n".join(context_items)
    # ... rest of method
```

**Example Output:**

```
Email #1:
From: Sarah Johnson <sarah.j@company.com>
Subject: Re: Budget Approval for Q1
Date: 2025-01-15 09:30:00

Content:
Thanks for sending the updated budget proposal. I reviewed it with
the finance team and we're ready to approve the $50K allocation for
the new servers. Please proceed with the vendor selection process.

---

Email #2:
From: Mike Chen <mike.chen@company.com>
Subject: Budget Approval for Q1
Date: 2025-01-14 16:45:00

Content:
Hi Sarah, attached is the updated Q1 budget proposal. We've reduced
the server costs by 10% as you suggested. Looking forward to your
approval so we can move forward.
```

### Option 3: Hybrid Approach (Best)

Combine both:
1. **Template selection** based on doc_type in retrieved chunks
2. **Dynamic context formatting** based on content type

```python
class PromptBuilder:
    def __init__(self, template: str | None = None, ...):
        self.template = template or DEFAULT_PROMPT_TEMPLATE_V2
        self.email_template = EMAIL_PROMPT_TEMPLATE  # ← New

    def build_prompt(self, query: str, context_chunks: List[Chunk]) -> str:
        # Auto-detect if all/most chunks are emails
        email_chunks = sum(1 for c in context_chunks
                          if c.meta.get("doc_type") == "outlook_eml")

        # Use email template if >50% of chunks are emails
        if email_chunks > len(context_chunks) / 2:
            selected_template = self.email_template
            context_str = self._build_email_context(context_chunks)
        else:
            selected_template = self.template
            context_str = self._build_document_context(context_chunks)

        return selected_template.format(
            context_str=context_str,
            query_str=query
        )
```

---

## Implementation Plan

### Phase 1: Quick Win (Immediate)

**Goal:** Make emails work reasonably well with minimal changes

**Changes:**
1. Add email metadata to context builder (15 lines of code)
2. Update context formatting for `doc_type == "outlook_eml"`

**Effort:** 30 minutes
**Impact:** Medium - emails will have proper context

### Phase 2: Email Template (Next)

**Goal:** Optimize LLM prompting for email-specific queries

**Changes:**
1. Create `EMAIL_PROMPT_TEMPLATE` constant
2. Add template selection logic based on doc_type
3. Test with common email queries

**Effort:** 1-2 hours
**Impact:** High - significantly better email search results

### Phase 3: Advanced Features (Future)

**Goal:** Leverage email-specific capabilities

**Possible Enhancements:**
1. **Thread Reconstruction**: Group emails by subject/conversation
2. **Temporal Filtering**: Pre-filter chunks by date before retrieval
3. **Sender Filtering**: Add sender-based retrieval strategies
4. **Action Item Extraction**: Special prompt for finding TODOs/deadlines
5. **Sentiment Analysis**: Detect concerns/blockers in email threads

**Effort:** Ongoing
**Impact:** Very High - transforms email search into intelligent assistant

---

## Example Queries & Expected Behavior

### Query 1: "What did Sarah say about the budget?"

**Current System (Without Email Metadata):**
```
Context:
Source ID: [outlook://account/Inbox]
Content: Thanks for sending the updated budget proposal. I reviewed it...

Answer: Based on the context, someone mentioned reviewing a budget proposal...
```
❌ **Problem**: No indication this was from Sarah

**With Email Metadata:**
```
Email #1:
From: Sarah Johnson <sarah.j@company.com>
Subject: Re: Budget Approval for Q1
Date: 2025-01-15 09:30:00

Content: Thanks for sending the updated budget proposal. I reviewed it...

Answer: Sarah Johnson said (on Jan 15, 2025) that she reviewed the budget
proposal with the finance team and approved the $50K allocation...
[Sarah Johnson, "Re: Budget Approval for Q1", 2025-01-15]
```
✅ **Better**: Clear attribution, date context, proper citation

### Query 2: "What were recent concerns about the migration?"

**With Email Template:**
```
Email #1:
From: IT Team <it-team@company.com>
Subject: Migration - Potential Issues
Date: 2025-01-18 11:00:00

Content: We've identified three potential blockers for the migration...

Answer: Recent concerns about the migration (as of Jan 18, 2025) include:
1. Database compatibility issues (raised by IT Team)
2. Downtime window conflicts (mentioned in follow-up)
3. Rollback plan gaps (IT Team's primary concern)

Sources: [IT Team, "Migration - Potential Issues", 2025-01-18]
```
✅ **Better**: Temporal awareness, organized by concern, clear sources

### Query 3: "Summarize action items from project emails"

**With Email Template:**
```
Answer: Based on the project emails, here are the action items:

**Immediate (This Week):**
- Finalize vendor selection (Sarah, by Friday Jan 19)
  [Sarah Johnson, "Re: Budget Approval", 2025-01-15]

- Complete security review (Mike, by Jan 20)
  [Mike Chen, "Security Checklist", 2025-01-17]

**Upcoming (Next 2 Weeks):**
- Schedule kickoff meeting (Team Lead, by end of month)
  [Jane Doe, "Project Timeline", 2025-01-16]

All action items extracted from 5 emails dated Jan 14-18, 2025.
```
✅ **Better**: Organized, owner-assigned, deadlines clear, citations

---

## Recommendation

**Implement Phase 1 + Phase 2 immediately:**

1. **Enhanced Context Builder** (Phase 1)
   - Quick win, minimal code
   - Makes emails immediately more useful
   - No breaking changes

2. **Email-Specific Template** (Phase 2)
   - Significant quality improvement
   - Tailored to email use cases
   - Auto-detection keeps backward compatibility

**Defer Phase 3** until we validate the basic email search workflow works well.

---

## Code Changes Required

### File 1: `scripts/prompting/prompt_builder.py`

**Add email template constant** (after line 75):
```python
EMAIL_PROMPT_TEMPLATE = """..."""  # See Option 1 above
```

**Modify `__init__`** (line 105):
```python
self.template = template or DEFAULT_PROMPT_TEMPLATE_V2
self.email_template = EMAIL_PROMPT_TEMPLATE  # ← ADD THIS
```

**Enhance `build_prompt`** (lines 157-162):
```python
# Replace existing metadata logic with doc_type-aware version
if chunk.meta.get("doc_type") == "outlook_eml":
    # Email-specific formatting
    subject = chunk.meta.get("subject", "No Subject")
    sender_name = chunk.meta.get("sender_name", "Unknown")
    date = chunk.meta.get("date", "Unknown Date")

    context_item = f"""Email from {sender_name}
Subject: {subject}
Date: {date}
Content: {text}"""
else:
    # Document formatting (existing logic)
    context_item = f"Source ID: [{source_id_str}]\nContent: {text}"
    page_number = chunk.meta.get('page_number')
    if page_number:
        context_item += f"\nPage: {page_number}"
```

**Add template auto-selection** (before line 174):
```python
# Auto-select template based on content type
email_chunks = sum(1 for c in context_chunks
                  if c.meta.get("doc_type") == "outlook_eml")
if email_chunks > len(context_chunks) / 2:
    template_to_use = self.email_template
else:
    template_to_use = self.template

final_prompt = template_to_use.format(...)
```

---

## Testing Plan

### Test Queries for Email Search

**Basic Retrieval:**
- "Find emails from Sarah"
- "Show me the latest email about the budget"
- "What did the IT team say?"

**Date-Based:**
- "Recent emails about the migration"
- "What was discussed last week about security?"

**Topic + Sender:**
- "What did John say about the API integration?"
- "Sarah's thoughts on the vendor proposal?"

**Action-Oriented:**
- "What are the action items from project emails?"
- "What deadlines were mentioned?"
- "Who needs to do what?"

**Analytical:**
- "Summarize the discussion about X"
- "What concerns were raised about Y?"
- "How did the plan evolve over time?"

---

## Success Metrics

**Quality Improvements:**
- ✅ Answers include sender names (not just "someone said")
- ✅ Answers reference email dates (temporal context)
- ✅ Citations use meaningful identifiers (sender, subject, date)
- ✅ Action items are properly attributed and dated

**User Experience:**
- ✅ Natural email queries work ("emails from Sarah")
- ✅ Temporal queries work ("recent", "last week")
- ✅ Results are organized and easy to scan
- ✅ Multi-email context is preserved (threads)

---

## Next Steps

1. **Review this proposal** - Gather feedback on approach
2. **Implement Phase 1** - Enhanced context builder (30 min)
3. **Implement Phase 2** - Email template + auto-selection (1-2 hours)
4. **Test with real queries** - Use test queries above
5. **Iterate based on results** - Refine template and context formatting
6. **Document best practices** - Update CLAUDE.md with email search tips

---

**Ready to implement?** The code changes are straightforward and backward-compatible.
