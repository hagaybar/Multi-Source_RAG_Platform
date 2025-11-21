"""
Decision Extractor

Extracts decisions, conclusions, and approvals from emails.

Uses LLM-based extraction for best accuracy.
"""

import json
from typing import List, Dict
from scripts.chunking.models import Chunk
from scripts.utils.logger import LoggerManager
from scripts.api_clients.openai.completer import OpenAICompleter

logger = LoggerManager.get_logger("decision_extractor")


class DecisionExtractor:
    """
    Extracts decisions and conclusions from emails.

    Features:
    - LLM-based extraction (GPT-4o for best accuracy)
    - Identifies decision, decision maker, date, and source
    """

    def __init__(self, use_llm: bool = True):
        """
        Initialize decision extractor.

        Args:
            use_llm: Use LLM for extraction (more accurate but costs ~$0.002/extraction)
        """
        self.use_llm = use_llm

        logger.info(f"DecisionExtractor initialized (LLM: {use_llm})")

    def extract(self, emails: List[Chunk]) -> List[Dict]:
        """
        Extract decisions from emails.

        Args:
            emails: List of email chunks

        Returns:
            [
                {
                    "decision": "Approved budget of $50K",
                    "made_by": "Sarah",
                    "date": "2025-01-15",
                    "source": "Email #3"
                },
                ...
            ]
        """
        logger.debug(f"Extracting decisions from {len(emails)} emails")

        if self.use_llm:
            try:
                decisions = self._extract_with_llm(emails)
                logger.info(f"Extracted {len(decisions)} decisions using LLM")
                return decisions
            except Exception as e:
                logger.warning(
                    f"LLM extraction failed: {e}",
                    extra={"error": str(e)}
                )
                return []

        # No pattern fallback for decisions (too complex)
        logger.warning("LLM disabled, returning empty list")
        return []

    def _extract_with_llm(self, emails: List[Chunk]) -> List[Dict]:
        """
        Use LLM to extract decisions.

        Args:
            emails: List of email chunks

        Returns:
            List of decision dictionaries
        """
        context = self._format_emails(emails)

        prompt = f"""Extract all decisions, conclusions, and approvals from these emails.

Emails:
{context}

For each decision, identify:
- What was decided
- Who made the decision (if mentioned)
- When (date if mentioned)
- Which email

Return ONLY valid JSON array (no markdown):
[
  {{
    "decision": "Approved $50K budget",
    "made_by": "Sarah",
    "date": "2025-01-15",
    "source_email": "#3"
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

            decisions = json.loads(response_clean)

            # Validate structure
            if not isinstance(decisions, list):
                raise ValueError(f"LLM response is not a list: {type(decisions)}")

            return decisions

        except json.JSONDecodeError as e:
            logger.error(
                f"Failed to parse LLM JSON response: {e}",
                extra={"response": response}
            )
            raise ValueError(f"Invalid JSON from LLM: {e}")

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

    extractor = DecisionExtractor(use_llm=False)  # Disable LLM for quick test

    # Test email with decisions
    test_email = Chunk(
        id="1",
        doc_id="d1",
        text="""Team,

After reviewing all the proposals, I've decided to approve the vendor selection.
We will go with Vendor A for the migration project.

The budget of $50,000 has been approved.

Best regards,
Sarah""",
        token_count=50,
        meta={
            "sender_name": "Sarah Wilson",
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "subject": "Decision on Vendor Selection"
        }
    )

    decisions = extractor.extract([test_email])

    print(f"\nExtracted {len(decisions)} decisions:")
    for dec in decisions:
        print(f"  - {dec.get('decision', 'N/A')}")
        if dec.get('made_by'):
            print(f"    Made by: {dec['made_by']}")
        if dec.get('date'):
            print(f"    Date: {dec['date']}")
