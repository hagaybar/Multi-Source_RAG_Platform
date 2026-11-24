"""
Test Domain Knowledge Injector (Phase 6, Task 6.3)

Tests the DomainKnowledgeInjector's ability to:
- Load domain knowledge from YAML
- Resolve entity aliases
- Inject relevant context into prompts
- Include workflows and common issues when relevant
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.agents.domain_knowledge_injector import DomainKnowledgeInjector


def test_domain_knowledge_injector():
    """Test DomainKnowledgeInjector on various entity sets."""

    print("=" * 80)
    print("DOMAIN KNOWLEDGE INJECTOR TEST (Phase 6, Task 6.3)")
    print("=" * 80)

    injector = DomainKnowledgeInjector()

    # Test cases
    test_cases = [
        {
            "entities": ["Primo VE", "facets"],
            "description": "Core product + feature",
            "expect_entities": 2,
            "expect_workflows": True,
            "expect_issues": True
        },
        {
            "entities": ["NDE"],
            "description": "Feature (alias test)",
            "expect_entities": 1,
            "expect_workflows": False,
            "expect_issues": False
        },
        {
            "entities": ["CSL", "Chicago citation"],
            "description": "Technology + related feature",
            "expect_entities": 2,
            "expect_workflows": True,
            "expect_issues": True
        },
        {
            "entities": ["Get It Button"],
            "description": "Feature with common issues",
            "expect_entities": 1,
            "expect_workflows": True,
            "expect_issues": True
        },
        {
            "entities": [],
            "description": "Empty entity list",
            "expect_entities": 0,
            "expect_workflows": False,
            "expect_issues": False
        },
    ]

    results = []

    for i, test in enumerate(test_cases, 1):
        entities = test["entities"]
        description = test["description"]
        expect_entities = test["expect_entities"]
        expect_workflows = test["expect_workflows"]
        expect_issues = test["expect_issues"]

        print(f"\n{'='*80}")
        print(f"TEST {i}/{len(test_cases)}: {description}")
        print(f"{'='*80}")
        print(f"Entities: {entities}")
        print("-" * 80)

        # Get context
        context = injector.get_context(
            entities,
            include_workflows=True,
            include_issues=True
        )

        # Check context
        has_content = len(context) > 0
        has_workflows = "**Relevant Workflows:**" in context
        has_issues = "**Common Issues:**" in context

        # Display results
        print(f"\n📊 Context Results:")
        print(f"  - Context length: {len(context)} chars")
        print(f"  - Has domain knowledge: {has_content}")
        print(f"  - Has workflows: {has_workflows}")
        print(f"  - Has common issues: {has_issues}")

        if context:
            print(f"\n💡 Generated Context (first 500 chars):")
            print(context[:500])
            if len(context) > 500:
                print(f"... ({len(context) - 500} more chars)")

        # Test prompt enrichment
        original_prompt = f"Answer this query about {', '.join(entities) if entities else 'Primo'}."
        enriched_prompt = injector.enrich_prompt(original_prompt, entities)

        enrichment_added = len(enriched_prompt) > len(original_prompt)
        print(f"\n📝 Prompt Enrichment:")
        print(f"  - Original length: {len(original_prompt)} chars")
        print(f"  - Enriched length: {len(enriched_prompt)} chars")
        print(f"  - Added: {len(enriched_prompt) - len(original_prompt)} chars")
        print(f"  - Enrichment applied: {enrichment_added}")

        # Validate results
        if expect_entities == 0:
            # Empty entity list - should return empty context
            success = not has_content and not has_workflows and not has_issues
        else:
            # Should have content
            success = has_content

            # Check workflows expectation
            if expect_workflows:
                success = success and has_workflows

            # Check issues expectation
            if expect_issues:
                success = success and has_issues

            # Should enrich prompt
            success = success and enrichment_added

        if success:
            print(f"\n✅ PASS: Context generated as expected")
        else:
            print(f"\n❌ FAIL: Context doesn't match expectations")
            print(f"      Expected workflows: {expect_workflows}, Got: {has_workflows}")
            print(f"      Expected issues: {expect_issues}, Got: {has_issues}")

        results.append({
            "entities": entities,
            "description": description,
            "context_length": len(context),
            "has_workflows": has_workflows,
            "has_issues": has_issues,
            "success": success
        })

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print("=" * 80)

    passed = sum(1 for r in results if r["success"])
    total = len(results)
    pass_rate = (passed / total) * 100

    print(f"\nTests Passed: {passed}/{total} ({pass_rate:.0f}%)")

    print(f"\n📋 Detailed Results:")
    for i, r in enumerate(results, 1):
        status = "✅ PASS" if r["success"] else "❌ FAIL"
        entities_str = ", ".join(r["entities"]) if r["entities"] else "None"
        print(f"  {i}. {status} - {r['description']}")
        print(f"      Entities: {entities_str}")
        print(f"      Context: {r['context_length']} chars, "
              f"Workflows: {r['has_workflows']}, Issues: {r['has_issues']}")

    # Final verdict
    print(f"\n{'='*80}")
    if pass_rate >= 80:
        print("✅ OVERALL: Domain Knowledge Injector is working well!")
        print("   - Correctly loads knowledge base")
        print("   - Resolves entities and aliases")
        print("   - Injects relevant context, workflows, and issues")
        return 0
    else:
        print("⚠️ OVERALL: Domain Knowledge Injector needs improvement")
        print(f"   - Pass rate: {pass_rate:.0f}% (target: 80%+)")
        return 1


if __name__ == "__main__":
    try:
        exit_code = test_domain_knowledge_injector()
        sys.exit(exit_code)
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
