"""
Answer Validator

Validates LLM-generated answers for quality and accuracy.

Checks:
- Required elements present (e.g., list for action items)
- Citations supported by context
- Contradictions in source material
- Completeness

Optional LLM-based contradiction detection for higher accuracy.
"""

import re
import json
from typing import Dict, List
from scripts.utils.logger import LoggerManager
from scripts.api_clients.openai.completer import OpenAICompleter

logger = LoggerManager.get_logger("answer_validator")


class AnswerValidator:
    """
    Validates LLM answers for quality and accuracy.

    Features:
    - Format checking (e.g., list for action items, sender mentioned)
    - Unsupported claims detection (heuristic)
    - Optional LLM-based contradiction detection
    """

    def __init__(self, use_llm: bool = False):
        """
        Initialize answer validator.

        Args:
            use_llm: Enable LLM-based contradiction detection (costs ~$0.001/validation)
        """
        self.use_llm = use_llm

        logger.info(
            f"AnswerValidator initialized (LLM contradiction detection: {use_llm})"
        )

    def validate(self, answer: str, context: str, intent: Dict) -> Dict:
        """
        Validate answer quality.

        Args:
            answer: LLM-generated answer to validate
            context: Email context used to generate answer
            intent: Detected intent with metadata

        Returns:
            {
                "is_valid": True/False,
                "issues": ["contradiction found", ...],
                "suggestions": ["Add conflicting info from Email #3"],
                "confidence": 0.85
            }
        """
        logger.debug(
            f"Validating answer for intent: {intent.get('primary_intent')}",
            extra={
                "intent": intent.get("primary_intent"),
                "answer_length": len(answer),
                "context_length": len(context),
            }
        )

        issues = []
        suggestions = []

        # Check 1: Required format based on intent
        format_check = self._check_format(answer, intent)
        if not format_check["valid"]:
            issues.append(format_check["issue"])
            suggestions.append(format_check["suggestion"])
            logger.warning(
                f"Format check failed: {format_check['issue']}",
                extra={"format_issue": format_check}
            )

        # Check 2: Contradiction detection (LLM-based if enabled)
        if self.use_llm:
            try:
                contradiction_check = self._check_contradictions_llm(answer, context)
                if contradiction_check["has_contradiction"]:
                    issues.append("Potential contradiction detected")
                    suggestions.append(contradiction_check["detail"])
                    logger.warning(
                        "Contradiction detected",
                        extra={"contradiction": contradiction_check["detail"]}
                    )
            except Exception as e:
                logger.error(
                    f"LLM contradiction check failed: {e}",
                    extra={"error": str(e)}
                )

        # Check 3: Unsupported claims (simple heuristic)
        unsupported = self._check_unsupported_claims(answer, context)
        if unsupported:
            issues.append(f"Found {len(unsupported)} unsupported claims")
            suggestions.extend(unsupported)
            logger.warning(
                f"Found {len(unsupported)} unsupported claims",
                extra={"unsupported_claims": unsupported}
            )

        is_valid = len(issues) == 0
        confidence = 1.0 - (len(issues) * 0.2)  # Rough confidence score

        result = {
            "is_valid": is_valid,
            "issues": issues,
            "suggestions": suggestions,
            "confidence": max(confidence, 0.0)
        }

        logger.info(
            f"Validation result: {'VALID' if is_valid else 'INVALID'} "
            f"(confidence: {result['confidence']:.2f})",
            extra={"validation_result": result}
        )

        return result

    def _check_format(self, answer: str, intent: Dict) -> Dict:
        """
        Check if answer format matches intent requirements.

        Args:
            answer: LLM-generated answer
            intent: Detected intent with metadata

        Returns:
            {"valid": True/False, "issue": "...", "suggestion": "..."}
        """
        primary_intent = intent.get("primary_intent")

        if primary_intent == "action_items":
            # Should have list or bullets
            has_list = any(marker in answer for marker in ['•', '-', '*', '1.', '2.', '3.'])
            if not has_list:
                return {
                    "valid": False,
                    "issue": "Action items query but no list format",
                    "suggestion": "Format answer as bulleted list of tasks"
                }

        elif primary_intent == "sender_query":
            # Should mention the sender's name
            sender = intent.get("metadata", {}).get("sender")
            if sender and sender.lower() not in answer.lower():
                return {
                    "valid": False,
                    "issue": f"Sender '{sender}' not mentioned in answer",
                    "suggestion": f"Ensure answer references what {sender} said"
                }

        elif primary_intent == "decision_tracking":
            # Should mention decision-related words
            decision_words = ["decided", "decision", "agreed", "approved", "conclusion"]
            has_decision_word = any(word in answer.lower() for word in decision_words)
            if not has_decision_word:
                return {
                    "valid": False,
                    "issue": "Decision query but no decision-related language",
                    "suggestion": "Answer should clearly state what was decided"
                }

        return {"valid": True}

    def _check_contradictions_llm(self, answer: str, context: str) -> Dict:
        """
        Use LLM to detect contradictions between answer and context.

        Args:
            answer: LLM-generated answer
            context: Email context

        Returns:
            {"has_contradiction": True/False, "detail": "..."}
        """
        logger.debug("Running LLM contradiction check")

        prompt = f"""Check if the answer contains contradictory information from the emails.

Emails:
{context[:1500]}

Answer:
{answer}

Are there any contradictions or conflicting information? If yes, describe them.

Return ONLY valid JSON (no markdown):
{{
  "has_contradiction": true/false,
  "detail": "Description of contradiction if found"
}}"""

        completer = OpenAICompleter(model_name="gpt-3.5-turbo")

        response = completer.get_completion(
            prompt=prompt,
            temperature=0.0,  # Deterministic for consistency
            max_tokens=150
        )

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

            result = json.loads(response_clean)
            return result
        except json.JSONDecodeError as e:
            logger.error(
                f"Failed to parse LLM JSON response: {e}",
                extra={"response": response}
            )
            # Return safe default
            return {"has_contradiction": False, "detail": ""}

    def _check_unsupported_claims(self, answer: str, context: str) -> List[str]:
        """
        Simple heuristic check for unsupported claims.

        Looks for specific numbers, dates, names in answer that don't appear in context.

        Args:
            answer: LLM-generated answer
            context: Email context

        Returns:
            List of unsupported claim descriptions
        """
        unsupported = []

        # Extract numbers from answer (dates, amounts, etc.)
        numbers_in_answer = set(re.findall(r'\b\d+\b', answer))
        numbers_in_context = set(re.findall(r'\b\d+\b', context))

        unsupported_numbers = numbers_in_answer - numbers_in_context
        if unsupported_numbers:
            unsupported.append(
                f"Numbers not in context: {', '.join(sorted(unsupported_numbers))}"
            )

        # Extract capitalized words (potential names/proper nouns)
        # But exclude common sentence starters and common words
        common_words = {
            "The", "This", "That", "These", "Those", "A", "An", "In", "On", "At",
            "Here", "There", "Where", "When", "What", "Which", "Who", "Why", "How",
            "Complete", "Schedule", "Review", "Action", "Items", "Task", "Tasks",
            "Email", "Emails", "From", "To", "Subject", "Date", "Time",
            "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
            "January", "February", "March", "April", "May", "June", "July",
            "August", "September", "October", "November", "December",
            "Project", "Discussion", "Meeting", "Budget", "Report", "Proposal",
            "Vendor", "Proposals"
        }

        proper_nouns_answer = set(
            word for word in re.findall(r'\b[A-Z][a-z]+\b', answer)
            if word not in common_words
        )
        proper_nouns_context = set(
            word for word in re.findall(r'\b[A-Z][a-z]+\b', context)
            if word not in common_words
        )

        unsupported_names = proper_nouns_answer - proper_nouns_context
        if unsupported_names and len(unsupported_names) > 2:
            # Only flag if more than 2 names missing (avoid false positives)
            unsupported.append(
                f"Potential names not in context: {', '.join(sorted(list(unsupported_names)[:3]))}"
            )

        return unsupported


if __name__ == "__main__":
    # Quick test
    validator = AnswerValidator(use_llm=False)

    # Test case 1: Action items without list format
    answer1 = "You need to complete the budget report and schedule a meeting with Alice."
    context1 = "Email: Please complete the budget report by Friday and schedule a meeting with Alice."
    intent1 = {"primary_intent": "action_items"}

    result1 = validator.validate(answer1, context1, intent1)
    print("\nTest 1 - Action items without list:")
    print(f"  Valid: {result1['is_valid']}")
    print(f"  Issues: {result1['issues']}")

    # Test case 2: Sender query with sender mentioned
    answer2 = "Alice said the budget needs to increase by 20% for Q4."
    context2 = "From: Alice\nWe need to increase the budget by 20% for Q4."
    intent2 = {"primary_intent": "sender_query", "metadata": {"sender": "Alice"}}

    result2 = validator.validate(answer2, context2, intent2)
    print("\nTest 2 - Sender query (valid):")
    print(f"  Valid: {result2['is_valid']}")
    print(f"  Confidence: {result2['confidence']:.2f}")

    # Test case 3: Unsupported numbers
    answer3 = "The project has 150 tasks remaining and needs $50,000 budget."
    context3 = "The project is ongoing with many tasks remaining."
    intent3 = {"primary_intent": "factual_lookup"}

    result3 = validator.validate(answer3, context3, intent3)
    print("\nTest 3 - Unsupported claims:")
    print(f"  Valid: {result3['is_valid']}")
    print(f"  Issues: {result3['issues']}")
    print(f"  Suggestions: {result3['suggestions']}")
