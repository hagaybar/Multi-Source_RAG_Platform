"""
Action Item Extractor

Extracts action items, tasks, and deadlines from emails.

Uses LLM-based extraction with pattern-matching fallback for reliability.
"""

import re
import json
from typing import List, Dict
from scripts.chunking.models import Chunk
from scripts.utils.logger import LoggerManager
from scripts.api_clients.openai.completer import OpenAICompleter

logger = LoggerManager.get_logger("action_item_extractor")


class ActionItemExtractor:
    """
    Extracts action items and deadlines from emails.

    Features:
    - LLM-based extraction (GPT-4o for best accuracy)
    - Pattern-matching fallback
    - Identifies task, deadline, assignee, and source
    """

    def __init__(self, use_llm: bool = True):
        """
        Initialize action item extractor.

        Args:
            use_llm: Use LLM for extraction (more accurate but costs ~$0.002/extraction)
        """
        self.use_llm = use_llm

        # Action item patterns for fallback
        self.action_patterns = [
            r'(?:TODO|Action item):\s*(.+)',
            r'(?:need to|should|must)\s+(.+?)(?:\.|,|;|\n)',
            r'(?:please|can you)\s+(.+?)(?:\.|,|;|\n)',
        ]

        # Deadline patterns
        self.deadline_patterns = [
            r'(?:by|before|due|deadline)\s+(\w+\s+\d+)',
            r'(?:by|before|due|deadline)\s+(today|tomorrow|next week|this week)',
        ]

        logger.info(f"ActionItemExtractor initialized (LLM: {use_llm})")

    def extract(self, emails: List[Chunk]) -> List[Dict]:
        """
        Extract action items from emails.

        Args:
            emails: List of email chunks

        Returns:
            [
                {
                    "task": "Review the budget proposal",
                    "deadline": "Friday",
                    "assigned_to": "Bob",
                    "source": "Email from Alice, Jan 15"
                },
                ...
            ]
        """
        logger.debug(f"Extracting action items from {len(emails)} emails")

        if self.use_llm:
            try:
                action_items = self._extract_with_llm(emails)
                logger.info(f"Extracted {len(action_items)} action items using LLM")
                return action_items
            except Exception as e:
                logger.warning(
                    f"LLM extraction failed, falling back to patterns: {e}",
                    extra={"error": str(e)}
                )

        # Fallback to pattern matching
        action_items = self._extract_with_patterns(emails)
        logger.info(f"Extracted {len(action_items)} action items using patterns")
        return action_items

    def _extract_with_llm(self, emails: List[Chunk]) -> List[Dict]:
        """
        Use LLM to extract action items.

        Args:
            emails: List of email chunks

        Returns:
            List of action item dictionaries
        """
        context = self._format_emails(emails)

        prompt = f"""Extract all action items, tasks, and deadlines from these emails.

Emails:
{context}

For each action item, identify:
- The task description
- Deadline (if mentioned)
- Who is responsible (if mentioned)
- Which email it came from

Return ONLY valid JSON array (no markdown):
[
  {{
    "task": "Review proposal",
    "deadline": "Friday",
    "assigned_to": "Bob",
    "source_email": "#2"
  }},
  ...
]"""

        completer = OpenAICompleter(model_name="gpt-4o")

        response = completer.get_completion(
            prompt=prompt,
            temperature=0.0,  # Deterministic for extraction
            max_tokens=500
        )

        logger.debug(f"LLM response: {response[:200]}...", extra={"llm_response": response})

        # Handle error responses
        if response.startswith("[ERROR]"):
            raise ValueError(f"LLM returned error: {response}")

        try:
            # Clean markdown if present
            response_clean = response.strip()
            if response_clean.startswith("```"):
                lines = response_clean.split("\n")
                json_lines = [line for line in lines if line and not line.startswith("```")]
                response_clean = "\n".join(json_lines)

            action_items = json.loads(response_clean)

            # Validate structure
            if not isinstance(action_items, list):
                raise ValueError(f"LLM response is not a list: {type(action_items)}")

            return action_items

        except json.JSONDecodeError as e:
            logger.error(
                f"Failed to parse LLM JSON response: {e}",
                extra={"response": response}
            )
            raise ValueError(f"Invalid JSON from LLM: {e}")

    def _extract_with_patterns(self, emails: List[Chunk]) -> List[Dict]:
        """
        Fallback pattern-based extraction.

        Args:
            emails: List of email chunks

        Returns:
            List of action item dictionaries
        """
        actions = []

        for i, email in enumerate(emails):
            text = email.text
            sender = email.meta.get("sender_name", "Unknown")
            date = email.meta.get("date", "")

            # Extract action items
            for pattern in self.action_patterns:
                matches = re.findall(pattern, text, re.I)
                for match in matches:
                    task = match.strip()

                    # Try to extract deadline from the task or surrounding text
                    deadline = None
                    for deadline_pattern in self.deadline_patterns:
                        deadline_match = re.search(deadline_pattern, text, re.I)
                        if deadline_match:
                            deadline = deadline_match.group(1)
                            break

                    actions.append({
                        "task": task,
                        "deadline": deadline,
                        "assigned_to": None,  # Hard to extract with patterns
                        "source": f"Email #{i+1} from {sender} ({date})"
                    })

        return actions

    def _format_emails(self, emails: List[Chunk]) -> str:
        """
        Format emails for LLM prompt.

        Args:
            emails: List of email chunks

        Returns:
            Formatted string with numbered emails
        """
        formatted = []

        for i, email in enumerate(emails):
            sender = email.meta.get("sender_name", "Unknown")
            date = email.meta.get("date", "")
            text = email.text[:500]  # Truncate long emails

            formatted.append(f"Email #{i+1} from {sender} ({date}):\n{text}")

        return "\n\n".join(formatted)


if __name__ == "__main__":
    # Quick test
    from datetime import datetime

    extractor = ActionItemExtractor(use_llm=False)  # Use patterns for quick test

    # Test email with action items
    test_email = Chunk(
        id="1",
        doc_id="d1",
        text="""Hi team,

We need to complete the following tasks:
- Review the budget proposal by Friday
- Please schedule a meeting with the vendor

Can you also send me the latest report?

Thanks!""",
        token_count=50,
        meta={
            "sender_name": "Alice Johnson",
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "subject": "Action Items"
        }
    )

    action_items = extractor.extract([test_email])

    print(f"\nExtracted {len(action_items)} action items:")
    for item in action_items:
        print(f"  - {item['task']}")
        if item['deadline']:
            print(f"    Deadline: {item['deadline']}")
        print(f"    Source: {item['source']}")
