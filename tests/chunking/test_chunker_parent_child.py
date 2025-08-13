import unittest
from scripts.chunking.chunking_strategies import parent_child_chunker
from scripts.chunking.rules_v3 import ChunkRule
from scripts.utils.logger import LoggerManager

class TestParentChildChunker(unittest.TestCase):

    def setUp(self):
        self.logger = LoggerManager.get_logger("test_chunking")
        self.meta = {"doc_id": "test_doc_123", "doc_type": "manual"}
        self.rule = ChunkRule(
            strategy="parent_child",
            max_tokens=1000,
            min_tokens=5,
            overlap=0
        )
        self.sample_text = """
# Section 1

This is the first paragraph of section 1.

This is the second paragraph of section 1.

## Section 1.1

This is a paragraph in a subsection.

# Section 2

This is the only paragraph in section 2.
"""

    def test_parent_child_chunking(self):
        chunks = parent_child_chunker(self.sample_text, self.meta, self.rule, self.logger)

        # There should be 3 parent chunks and 4 child chunks
        self.assertEqual(len(chunks), 7)

        parent_chunks = [c for c in chunks if c.parent_id is None]
        child_chunks = [c for c in chunks if c.parent_id is not None]

        self.assertEqual(len(parent_chunks), 3)
        self.assertEqual(len(child_chunks), 4)

        # Check parent chunks
        self.assertIn("Section 1", parent_chunks[0].title)
        self.assertIn("Section 1.1", parent_chunks[1].title)
        self.assertIn("Section 2", parent_chunks[2].title)

        # Check child chunks and their parent_id
        parent_1_id = parent_chunks[0].id
        parent_2_id = parent_chunks[1].id
        parent_3_id = parent_chunks[2].id

        children_of_parent1 = [c for c in child_chunks if c.parent_id == parent_1_id]
        self.assertEqual(len(children_of_parent1), 2)
        self.assertIn("first paragraph", children_of_parent1[0].text)
        self.assertIn("second paragraph", children_of_parent1[1].text)

        children_of_parent2 = [c for c in child_chunks if c.parent_id == parent_2_id]
        self.assertEqual(len(children_of_parent2), 1)
        self.assertIn("subsection", children_of_parent2[0].text)

        children_of_parent3 = [c for c in child_chunks if c.parent_id == parent_3_id]
        self.assertEqual(len(children_of_parent3), 1)
        self.assertIn("section 2", children_of_parent3[0].text)

if __name__ == '__main__':
    unittest.main()
