# Email RAG System - Phase 5-8: Adaptive Retrieval & Intelligence

**Document Version:** 1.0
**Created:** 2025-01-24
**Status:** Planning
**Prerequisites:** Phase 1-4 Complete (Intent Detection, Specialized Retrievers, Context Assembly, Quality Enhancements)

---

## Overview

This document outlines the next evolution of the Email RAG system: **adaptive, self-improving retrieval** that learns from initial results and dynamically adjusts strategies to maximize answer quality.

**Core Innovation:** Instead of a fixed pipeline, the system will:
1. Execute initial retrieval
2. Assess quality/potential of results
3. Dynamically decide: accept, rerank, reformulate, or ask for clarification
4. Iterate until confidence threshold is met

**Phases:**
- **Phase 5**: Reranking & Quality Assessment (foundation - 2-3 days)
- **Phase 6**: Query Intelligence (reformulation & entity knowledge - 3-4 days)
- **Phase 7**: Adaptive Retrieval (dynamic decision making - 3-4 days)
- **Phase 8**: Conversational Mode (optional enhancement - 2-3 days)

**Total Estimated Time:** 10-14 days (can be done incrementally)

---

## Phase 5: Reranking & Quality Assessment

**Objective:** Add a second-stage relevance filter and ability to assess retrieval quality before generating answers.

**Status:** Not Started
**Estimated Time:** 2-3 days
**Priority:** HIGH (quick wins, foundational for later phases)

### Tasks

#### Task 5.1: Implement Cross-Encoder Reranker (4-6 hours)

**Goal:** Add reranking capability using a cross-encoder model.

**Subtasks:**
1. **Research & Select Model** (1 hour)
   - Option A: `cross-encoder/ms-marco-MiniLM-L-12-v2` (fast, good quality)
   - Option B: `cross-encoder/ms-marco-MiniLM-L-6-v2` (faster, slightly lower quality)
   - Option C: Cohere Rerank API (requires API key, costs ~$1-2 per 1K queries)
   - **Decision criterion:** Speed vs quality vs cost
   - **Recommendation:** Start with Option A (free, good balance)

2. **Create Reranker Module** (2 hours)
   - File: `scripts/retrieval/reranker.py`
   - Class: `CrossEncoderReranker`
   - Methods:
     - `__init__(model_name: str)` - Load model
     - `rerank(query: str, chunks: List[Chunk], top_k: int) -> List[Chunk]` - Rerank and return top-k
     - `get_relevance_scores(query: str, chunks: List[Chunk]) -> List[float]` - Return scores

3. **Integrate into Retrieval Pipeline** (1 hour)
   - Modify `EmailOrchestratorAgent` to support reranking
   - Add config option: `use_reranking: true/false`
   - Change retrieval flow:
     ```python
     # Before:
     chunks = faiss_retriever.retrieve(query, top_k=15)

     # After:
     candidates = faiss_retriever.retrieve(query, top_k=100)  # Cast wide net
     if config.use_reranking:
         chunks = reranker.rerank(query, candidates, top_k=15)  # Precision filter
     else:
         chunks = candidates[:15]
     ```

4. **Update Chunk Metadata** (30 min)
   - Store reranking scores in chunk metadata
   - Field: `meta["rerank_score"]` (0-1, higher = more relevant)
   - Field: `meta["faiss_score"]` (rename from `similarity`)
   - This enables comparison and debugging

5. **Test on Existing Queries** (1 hour)
   - Rerun 13 test queries with reranking enabled
   - Compare:
     - Chunks retrieved (with vs without reranking)
     - Answer quality
     - Focus on "What happened with Primo VE last month?" query
   - Save results: `test_results_with_reranking_[timestamp]/`

**Deliverables:**
- `scripts/retrieval/reranker.py` - Reranker module
- Updated `EmailOrchestratorAgent` with reranking support
- Config option in `configs/email_orchestrator.yaml`
- Test results comparing with/without reranking

**Success Criteria:**
- [ ] Reranker loads and runs without errors
- [ ] Scores are captured in chunk metadata
- [ ] Test queries run with reranking enabled
- [ ] At least 1-2 queries show measurable improvement

---

#### Task 5.2: Build Quality Assessment Agent (3-4 hours)

**Goal:** Create an agent that evaluates whether retrieval results have potential to answer the query.

**Subtasks:**
1. **Create Quality Assessment Module** (2 hours)
   - File: `scripts/agents/retrieval_quality_agent.py`
   - Class: `RetrievalQualityAgent`
   - Method: `assess_quality(query: str, chunks: List[Chunk]) -> QualityAssessment`
   - Returns:
     ```python
     QualityAssessment(
         has_potential: bool,  # Can these chunks answer the query?
         confidence: float,    # 0-1
         issues: List[str],    # ["too_generic", "wrong_time_period", "off_topic"]
         suggestions: List[str],  # ["rerank", "reformulate", "expand_temporal"]
         reasoning: str        # Explanation
     )
     ```

2. **Implement Assessment Logic** (1.5 hours)
   - Use LLM (gpt-4o-mini) to evaluate:
     ```python
     prompt = f"""
     Query: "{query}"

     Retrieved {len(chunks)} chunks. Here are the top 5:
     {format_chunks_preview(chunks[:5])}

     Assess whether these chunks have potential to answer the query:

     1. Coverage: Do chunks contain relevant information?
     2. Specificity: Are chunks specific enough (not too generic)?
     3. Temporal alignment: If temporal query, are dates correct?
     4. Topical relevance: Are chunks on-topic?

     Return JSON:
     {{
         "has_potential": true/false,
         "confidence": 0.0-1.0,
         "issues": ["issue1", "issue2"],
         "suggestions": ["action1", "action2"],
         "reasoning": "explanation"
     }}
     """
     ```

3. **Add Heuristic Checks** (30 min)
   - Fast pre-checks before LLM call:
     - Average rerank score < 0.3 → Low potential
     - All chunks from wrong time period → Temporal mismatch
     - High diversity but low relevance → Off-topic
   - If heuristics fail, skip LLM call (save cost)

4. **Test Assessment Accuracy** (1 hour)
   - Run on 13 test queries
   - For each, get quality assessment
   - Manually verify: Does assessment match actual answer quality?
   - Calculate accuracy: Good assessment vs actual outcome

**Deliverables:**
- `scripts/agents/retrieval_quality_agent.py` - Quality assessment agent
- Test results showing assessment accuracy
- Documentation of assessment criteria

**Success Criteria:**
- [ ] Agent correctly identifies low-quality retrievals (e.g., "What happened last month?")
- [ ] Agent correctly identifies high-quality retrievals (e.g., "Chicago bibliography")
- [ ] Assessment accuracy > 80% on test set
- [ ] Suggestions are actionable (rerank, reformulate, etc.)

---

#### Task 5.3: Fix Similarity Score Capture (1 hour)

**Goal:** Ensure FAISS similarity scores are properly captured and displayed in reports.

**Current Issue:** All scores show 0.0000 in reports.

**Subtasks:**
1. **Debug FAISS Score Capture** (30 min)
   - Check: `FaissRetriever.retrieve_vector()` returns scores
   - Check: Scores are stored in `Chunk.meta["similarity"]`
   - Find where scores are lost in pipeline

2. **Fix Score Propagation** (15 min)
   - Ensure scores pass through:
     - FAISS retriever → Chunk metadata
     - Multi-aspect retriever → Final results
     - EmailOrchestrator → Retrieved chunks

3. **Update Report Generator** (15 min)
   - Ensure `query_report_generator.py` reads scores correctly
   - Display both FAISS and rerank scores in reports

**Deliverables:**
- Fixed score propagation
- Reports showing actual similarity scores

**Success Criteria:**
- [ ] Reports show non-zero similarity scores
- [ ] Scores make sense (higher = more similar)
- [ ] Both FAISS and rerank scores are visible

---

### Phase 5 Summary

**Before Phase 5:**
- Query → FAISS (top 15) → LLM → Answer
- No visibility into retrieval quality
- One-shot, no adaptation

**After Phase 5:**
- Query → FAISS (top 100) → Rerank (top 15) → Quality Check → LLM → Answer
- Quality assessment before generation
- Foundation for adaptive strategies

**Testing:** Rerun test suite, compare metrics before/after.

---

## Phase 6: Query Intelligence

**Objective:** Add query analysis, reformulation for vague queries, and domain knowledge injection.

**Status:** Not Started
**Estimated Time:** 3-4 days
**Priority:** MEDIUM-HIGH (addresses vague query problem)

### Tasks

#### Task 6.1: Build Query Analysis Agent (2-3 hours)

**Goal:** Analyze queries to determine clarity, entities, and intent quality.

**Subtasks:**
1. **Create Query Analyzer Module** (1.5 hours)
   - File: `scripts/agents/query_analyzer.py`
   - Class: `QueryAnalyzer`
   - Method: `analyze(query: str) -> QueryAnalysis`
   - Returns:
     ```python
     QueryAnalysis(
         is_specific: bool,      # Specific vs vague
         entities: List[str],    # ["Primo VE", "NDE", "Chicago citation"]
         intent_quality: float,  # How clear is the intent? (0-1)
         issues: List[str],      # ["too_vague", "multiple_intents"]
         clarification_questions: List[str],  # If vague, what to ask
         suggested_reformulations: List[str]  # Alternative phrasings
     )
     ```

2. **Implement Analysis Logic** (1 hour)
   - Use LLM to analyze:
     ```python
     prompt = f"""
     Analyze this query for clarity and specificity:
     Query: "{query}"

     Assess:
     1. Specificity: Is it specific or vague/broad?
     2. Entities: Extract domain entities (products, features, etc.)
     3. Intent quality: How clear is what the user wants?
     4. Issues: Any problems (vagueness, ambiguity)?
     5. Clarifications: If vague, what questions would help?
     6. Reformulations: Rewrite as specific sub-queries

     Return JSON: {{...}}
     """
     ```

3. **Test on Query Types** (30 min)
   - Specific: "How do I configure facets in Primo?" → Should be clear
   - Vague: "What happened with Primo VE last month?" → Should flag as vague
   - Verify analysis accuracy

**Deliverables:**
- `scripts/agents/query_analyzer.py` - Query analyzer
- Test results showing analysis quality

**Success Criteria:**
- [ ] Correctly identifies specific vs vague queries
- [ ] Extracts relevant entities (Primo, NDE, etc.)
- [ ] Suggests useful clarifications for vague queries
- [ ] Reformulations are more specific than original

---

#### Task 6.2: Implement Query Reformulation (3-4 hours)

**Goal:** Automatically reformulate vague queries into specific sub-queries.

**Subtasks:**
1. **Create Reformulation Module** (2 hours)
   - File: `scripts/agents/query_reformulator.py`
   - Class: `QueryReformulator`
   - Method: `reformulate(query: str, context: dict) -> List[str]`
   - Strategy:
     ```python
     # Vague: "What happened with Primo VE last month?"
     # Becomes:
     [
         "What new features were released in Primo VE in November 2025?",
         "What bugs or issues were reported with Primo VE in November 2025?",
         "What configuration changes were made in Primo VE in November 2025?",
         "What announcements were made about Primo VE in November 2025?"
     ]
     ```

2. **Integrate Multi-Query Retrieval** (1 hour)
   - Modify retrieval to handle multiple sub-queries:
     ```python
     if query_analysis.is_specific:
         chunks = standard_retrieval(query)
     else:
         sub_queries = reformulator.reformulate(query)
         all_chunks = []
         for sq in sub_queries:
             all_chunks.extend(faiss_retriever.retrieve(sq, top_k=30))
         # Deduplicate and rerank
         chunks = reranker.rerank(query, all_chunks, top_k=15)
     ```

3. **Add Deduplication** (30 min)
   - When merging results from multiple sub-queries, remove duplicates
   - Use content hash or chunk ID

4. **Test on Vague Queries** (30 min)
   - Run "What happened last month?" with reformulation
   - Compare chunks retrieved vs without reformulation
   - Measure if answer quality improves

**Deliverables:**
- `scripts/agents/query_reformulator.py` - Reformulation module
- Updated orchestrator to support multi-query retrieval
- Test results on vague queries

**Success Criteria:**
- [ ] Vague queries are reformulated into 3-5 specific sub-queries
- [ ] Multi-query retrieval returns more relevant chunks
- [ ] "What happened last month?" query shows improvement
- [ ] No degradation on specific queries

---

#### Task 6.3: Build Domain Knowledge Base (2-3 hours)

**Goal:** Create a knowledge base of domain entities to inject context.

**Subtasks:**
1. **Create Knowledge Base Structure** (1 hour)
   - File: `data/knowledge/primo_domain_knowledge.yaml`
   - Format:
     ```yaml
     entities:
       "Primo VE":
         type: product
         description: "Ex Libris discovery and delivery system"
         aliases: ["Primo", "PrimoVE"]
         related_to: ["NDE", "Discovery", "Search"]
         context: |
           Primo VE is a cloud-based discovery system by Ex Libris.
           Key components: NDE (UI), Discovery (search), Fulfillment (delivery)
           Release cycle: Quarterly (Feb, May, Aug, Nov)
           Common issues: Citation styles, performance, configuration

       "NDE":
         type: feature
         description: "Next Discovery Experience - new UI for Primo VE"
         aliases: ["Next Discovery Experience", "New UI"]
         related_to: ["Primo VE", "UI", "User Interface"]
         context: |
           NDE is the modernized user interface for Primo VE.
           Rollout: Gradual 2024-2025, general availability Nov 2025
           Features: Improved accessibility, modern design, search history

       "CSL":
         type: technology
         description: "Citation Style Language"
         aliases: ["Citation Style Language", "citation styles"]
         related_to: ["Chicago", "APA", "MLA", "bibliography"]
         context: |
           CSL is an XML-based format for citation styles.
           Used in Primo VE for generating citations.
           Updated via GitHub CSL repository
           Common issue: Delays in updating to new editions (e.g., Chicago 18th)

     # Add 10-15 key entities from Primo domain
     ```

2. **Create Knowledge Injection Module** (1 hour)
   - File: `scripts/agents/knowledge_injector.py`
   - Class: `DomainKnowledgeInjector`
   - Methods:
     - `extract_entities(query: str) -> List[str]`
     - `get_context(entities: List[str]) -> str`
     - `augment_prompt(query: str, chunks: List[Chunk]) -> str`

3. **Integrate into Prompt Building** (30 min)
   - Add domain context to LLM prompt:
     ```python
     prompt = f"""
     Query: {query}

     Domain Context:
     {domain_knowledge}

     Retrieved Information:
     {chunks}

     Please answer the query using the retrieved information and domain context.
     """
     ```

4. **Test Impact** (30 min)
   - Run queries that use domain terms (NDE, CSL, etc.)
   - Check if answers are better informed
   - Look for reduced confusion about terminology

**Deliverables:**
- `data/knowledge/primo_domain_knowledge.yaml` - Knowledge base
- `scripts/agents/knowledge_injector.py` - Injection module
- Test results showing impact

**Success Criteria:**
- [ ] Knowledge base covers 10-15 key domain entities
- [ ] Entities are correctly extracted from queries
- [ ] Context is injected into prompts
- [ ] Answers show better understanding of domain terminology

---

### Phase 6 Summary

**Before Phase 6:**
- Vague queries fail (retrieval can't handle broad questions)
- No domain context (LLM doesn't understand Primo terminology)

**After Phase 6:**
- Vague queries → Reformulated → Multi-query retrieval → Better results
- Domain knowledge injected → LLM understands context
- Query analysis provides visibility into query quality

**Testing:** Compare vague query performance before/after.

---

## Phase 7: Adaptive Retrieval

**Objective:** Dynamic decision-making based on retrieval quality assessment.

**Status:** Not Started
**Estimated Time:** 3-4 days
**Priority:** HIGH (implements the adaptive loop)

### Tasks

#### Task 7.1: Build Adaptive Decision Agent (3-4 hours)

**Goal:** Agent that decides which strategy to use based on quality assessment.

**Subtasks:**
1. **Create Decision Agent** (2 hours)
   - File: `scripts/agents/adaptive_decision_agent.py`
   - Class: `AdaptiveDecisionAgent`
   - Method: `decide_strategy(query, initial_results, quality_assessment) -> Strategy`
   - Decision logic:
     ```python
     def decide_strategy(self, query, results, quality):
         if quality.has_potential and quality.confidence > 0.7:
             return Strategy.ACCEPT  # Use results as-is

         if quality.confidence > 0.5:
             if "rerank_needed" in quality.suggestions:
                 return Strategy.RERANK  # Rerank existing results

         if quality.confidence > 0.3:
             if "reformulate" in quality.suggestions:
                 return Strategy.REFORMULATE  # Try different query
             if "expand_temporal" in quality.suggestions:
                 return Strategy.EXPAND_TEMPORAL  # Wider date range

         # Low confidence - need clarification
         return Strategy.ASK_CLARIFICATION
     ```

2. **Define Strategy Enum** (30 min)
   - Strategies:
     - `ACCEPT` - Use current results
     - `RERANK` - Apply stronger reranking
     - `REFORMULATE` - Rewrite query and retry
     - `EXPAND_TEMPORAL` - Widen time window
     - `EXPAND_RETRIEVAL` - Retrieve more candidates (k=200)
     - `ASK_CLARIFICATION` - Need user input

3. **Implement Strategy Executors** (1 hour)
   - Each strategy has an executor:
     ```python
     class StrategyExecutor:
         def execute_rerank(self, query, results):
             return stronger_reranker.rerank(query, results, top_k=10)

         def execute_reformulate(self, query, initial_results):
             new_queries = reformulator.reformulate(query)
             return multi_query_retrieval(new_queries)

         def execute_expand_temporal(self, query, current_constraint):
             new_constraint = expand_time_window(current_constraint)
             return retrieve_with_constraint(query, new_constraint)
     ```

4. **Test Decision Logic** (30 min)
   - Run on different quality scenarios:
     - High quality → Should accept
     - Medium quality with low scores → Should rerank
     - Low quality, vague query → Should reformulate
     - Very low quality → Should ask clarification

**Deliverables:**
- `scripts/agents/adaptive_decision_agent.py` - Decision agent
- Strategy executors
- Test results showing decision accuracy

**Success Criteria:**
- [ ] Agent correctly identifies which strategy to use
- [ ] Strategies are executed successfully
- [ ] No infinite loops (max 2-3 iterations)
- [ ] Improved results after adaptation

---

#### Task 7.2: Implement Adaptive Loop (3-4 hours)

**Goal:** Integrate adaptive decision-making into the main pipeline.

**Subtasks:**
1. **Create Adaptive Pipeline** (2 hours)
   - File: `scripts/pipeline/adaptive_runner.py`
   - Class: `AdaptiveRetrievalPipeline`
   - Flow:
     ```python
     def run_adaptive_retrieval(self, query, max_iterations=3):
         iteration = 0
         current_query = query

         while iteration < max_iterations:
             # Step 1: Retrieve
             results = self.retrieve(current_query)

             # Step 2: Assess quality
             quality = self.quality_agent.assess(query, results)

             # Step 3: Decide strategy
             strategy = self.decision_agent.decide(query, results, quality)

             # Step 4: Execute or accept
             if strategy == Strategy.ACCEPT:
                 return results  # Good enough!

             # Step 5: Adapt and retry
             results = self.execute_strategy(strategy, query, results)
             iteration += 1

         # Max iterations reached - return best we have
         return results
     ```

2. **Add Iteration Tracking** (30 min)
   - Log each iteration:
     - Query used
     - Results retrieved
     - Quality assessment
     - Strategy chosen
     - Outcome
   - Store in run artifacts for debugging

3. **Implement Safety Limits** (30 min)
   - Max iterations: 3 (prevent infinite loops)
   - Confidence threshold: 0.7 (accept if above)
   - Timeout: 30 seconds total (prevent long delays)

4. **Test Adaptive Loop** (1 hour)
   - Test cases:
     - **Case 1**: Specific query → Should accept first retrieval
     - **Case 2**: Vague query → Should reformulate once, then accept
     - **Case 3**: Very vague → Should reformulate twice, maybe ask clarification
   - Measure: Iterations needed, final quality, time taken

**Deliverables:**
- `scripts/pipeline/adaptive_runner.py` - Adaptive pipeline
- Iteration logging
- Test results showing adaptive behavior

**Success Criteria:**
- [ ] Adaptive loop runs without errors
- [ ] Specific queries converge in 1 iteration (just accept)
- [ ] Vague queries improve after 1-2 iterations
- [ ] No timeouts or infinite loops
- [ ] Final quality > initial quality

---

#### Task 7.3: Update Orchestrator to Use Adaptive Pipeline (1-2 hours)

**Goal:** Integrate adaptive pipeline into EmailOrchestratorAgent.

**Subtasks:**
1. **Add Config Option** (15 min)
   - `configs/email_orchestrator.yaml`:
     ```yaml
     adaptive_retrieval:
       enabled: true
       max_iterations: 3
       confidence_threshold: 0.7
       strategies_enabled:
         - rerank
         - reformulate
         - expand_temporal
     ```

2. **Modify EmailOrchestratorAgent** (45 min)
   - Add adaptive mode:
     ```python
     def retrieve(self, query, top_k=None):
         if self.config.adaptive_retrieval.enabled:
             return self.adaptive_pipeline.run(query)
         else:
             return self.standard_retrieve(query, top_k)
     ```

3. **Update Reports** (30 min)
   - Show adaptive iterations in report:
     ```markdown
     ## Adaptive Retrieval Flow

     **Iteration 1:**
     - Query: "What happened with Primo VE last month?"
     - Retrieved: 14 chunks
     - Quality: Low (confidence: 0.4)
     - Strategy: REFORMULATE

     **Iteration 2:**
     - Sub-queries: ["New features Nov 2025", "Issues Nov 2025", ...]
     - Retrieved: 32 chunks (merged)
     - Quality: Good (confidence: 0.8)
     - Strategy: ACCEPT

     **Final:** 15 chunks after reranking
     ```

**Deliverables:**
- Updated EmailOrchestratorAgent with adaptive mode
- Config options
- Enhanced reports showing iterations

**Success Criteria:**
- [ ] Adaptive mode can be toggled on/off
- [ ] Reports show iteration details
- [ ] No breaking changes to existing functionality
- [ ] Test suite still passes

---

### Phase 7 Summary

**Before Phase 7:**
- Fixed pipeline: retrieve → generate
- No adaptation if results are poor
- Vague queries fail silently

**After Phase 7:**
- Adaptive pipeline: retrieve → assess → decide → adapt → retry if needed
- System learns from retrieval quality
- Multiple strategies available
- Converges to best results

**Testing:** Run full test suite with adaptive mode, measure improvement.

---

## Phase 8: Conversational Mode (Optional)

**Objective:** Enable multi-turn conversations where system can ask clarifying questions.

**Status:** Not Started
**Estimated Time:** 2-3 days
**Priority:** LOW (nice-to-have, can defer)

### Tasks

#### Task 8.1: Design Conversation State Management (1-2 hours)

**Goal:** Track conversation history and context across turns.

**Subtasks:**
1. **Create Conversation Manager** (1 hour)
   - File: `scripts/conversation/manager.py`
   - Class: `ConversationManager`
   - State:
     ```python
     ConversationState(
         session_id: str,
         turns: List[Turn],
         context: dict,
         clarifications_asked: int,
         max_clarifications: int = 2
     )

     Turn(
         turn_id: int,
         user_message: str,
         system_message: str,
         query_used: str,
         results: List[Chunk],
         type: str  # "query", "clarification_request", "answer"
     )
     ```

2. **Design Conversation Flow** (30 min)
   - Flow diagram:
     ```
     User Query
       ↓
     Analyze Clarity
       ↓
     [Clear] → Retrieve → Answer
       ↓
     [Unclear] → Ask Clarification → Wait for Response
       ↓
     User Response → Merge with Original → Retry
     ```

3. **Set Limits** (30 min)
   - Max clarifications: 2 per query (prevent endless back-and-forth)
   - Timeout: 5 minutes (auto-accept after timeout)

**Deliverables:**
- `scripts/conversation/manager.py` - Conversation manager
- Design document for conversation flow

**Success Criteria:**
- [ ] State tracking works across turns
- [ ] Clarification limits enforced
- [ ] History is preserved for context

---

#### Task 8.2: Implement Clarification Request Generation (2-3 hours)

**Goal:** Generate useful clarification questions when query is unclear.

**Subtasks:**
1. **Create Clarification Generator** (1.5 hours)
   - File: `scripts/conversation/clarification_generator.py`
   - Class: `ClarificationGenerator`
   - Method: `generate_questions(query, analysis) -> List[str]`
   - Example:
     ```python
     # Query: "What happened with Primo VE last month?"
     # Generates:
     [
         "Are you asking about new releases and features?",
         "Are you asking about issues and bugs reported?",
         "Are you asking about configuration changes?",
         "Are you interested in all of the above?"
     ]
     ```

2. **Add UI Support** (1 hour)
   - Streamlit: Show clarification questions as radio buttons or checkboxes
   - User selects option(s)
   - System reformulates query based on selection

3. **Test Clarification Flow** (30 min)
   - User: "What happened last month?"
   - System: "Are you asking about [features/bugs/config]?"
   - User: "Bugs"
   - System: Retrieves with focused query about bugs

**Deliverables:**
- `scripts/conversation/clarification_generator.py`
- UI components for clarification
- Test flow end-to-end

**Success Criteria:**
- [ ] Clarification questions are relevant
- [ ] User can select and continue
- [ ] Query is reformulated based on selection
- [ ] Retrieval improves after clarification

---

#### Task 8.3: Integrate Conversational Mode into UI (2-3 hours)

**Goal:** Add conversational interface to Streamlit UI.

**Subtasks:**
1. **Add Conversation Toggle** (30 min)
   - UI checkbox: "Enable conversational mode"
   - If enabled, system can ask questions
   - If disabled, standard one-shot query

2. **Implement Chat Interface** (1.5 hours)
   - Chat-style UI (like ChatGPT):
     - User message
     - System message (question or answer)
     - User can respond
   - Store conversation history

3. **Handle Multi-Turn State** (1 hour)
   - Use Streamlit session state
   - Track conversation ID
   - Show full conversation thread

**Deliverables:**
- Updated UI with conversational mode
- Chat interface
- Multi-turn support

**Success Criteria:**
- [ ] User can toggle conversational mode
- [ ] System can ask clarification questions
- [ ] User can respond and continue
- [ ] Full conversation is visible
- [ ] Works seamlessly with adaptive retrieval

---

### Phase 8 Summary

**Before Phase 8:**
- One-shot Q&A
- No follow-up or clarification
- User must reformulate themselves

**After Phase 8:**
- Multi-turn conversations
- System can ask for clarification
- Natural back-and-forth
- Better user experience

**Note:** This phase is optional and can be deferred if time is limited.

---

## Integration & Testing

### Integration Strategy

**Phases 5-7 are sequential** (each builds on previous):
- Phase 5 → Reranking & quality assessment (foundation)
- Phase 6 → Query intelligence (tools for adaptation)
- Phase 7 → Adaptive loop (combines everything)

**Phase 8 is independent** (can be done anytime after Phase 7):
- Works on top of adaptive pipeline
- Purely UI/UX enhancement

**Recommended Order:**
1. Complete Phase 5 fully (2-3 days) → Test and validate
2. Complete Phase 6 fully (3-4 days) → Test and validate
3. Complete Phase 7 fully (3-4 days) → Test and validate
4. (Optional) Phase 8 if needed (2-3 days)

### Testing Strategy

**After Each Phase:**
1. **Unit Tests** - Test individual components
2. **Integration Tests** - Test phase works with existing system
3. **Regression Tests** - Rerun 13-query test suite, compare metrics
4. **Case Study Tests** - Specific failing queries (e.g., "What happened last month?")

**Metrics to Track:**
- Retrieval quality (via quality assessment scores)
- Answer quality (manual review)
- Iterations needed (for adaptive mode)
- Latency (end-to-end time)
- Success rate (% queries with good answers)

**Test Suite Updates:**
1. Add 5-10 new test questions focusing on:
   - Vague queries (Phase 6 target)
   - Temporal queries with edge cases
   - Queries requiring domain knowledge
   - Multi-faceted questions

2. Create comparison reports:
   - Baseline (Phase 1-4 only)
   - + Reranking (Phase 5)
   - + Query intelligence (Phase 6)
   - + Adaptive retrieval (Phase 7)

### Success Criteria (Overall)

**Must-Have (Phase 5-7):**
- [ ] Zero-result queries reduced by 50%+
- [ ] Vague queries ("What happened...") now work
- [ ] Average quality assessment confidence > 0.7
- [ ] No regression on queries that already worked
- [ ] Latency < 10 seconds (including adaptations)

**Nice-to-Have (Phase 8):**
- [ ] Conversational mode works smoothly
- [ ] Users can clarify vague queries interactively
- [ ] Better UX for complex information needs

---

## Configuration

### New Config Files

**1. `configs/reranking.yaml`**
```yaml
reranking:
  enabled: true
  model: "cross-encoder/ms-marco-MiniLM-L-12-v2"
  initial_k: 100  # Candidates from FAISS
  final_k: 15     # After reranking
  score_threshold: 0.3  # Min score to include
```

**2. `configs/adaptive_retrieval.yaml`**
```yaml
adaptive:
  enabled: true
  max_iterations: 3
  confidence_threshold: 0.7
  timeout_seconds: 30

  strategies:
    rerank:
      enabled: true
      trigger_confidence: 0.5

    reformulate:
      enabled: true
      trigger_confidence: 0.3
      max_sub_queries: 5

    expand_temporal:
      enabled: true
      trigger_confidence: 0.4
      expansion_factor: 2  # 30 days → 60 days

    ask_clarification:
      enabled: false  # Enable after Phase 8
      trigger_confidence: 0.2
```

**3. `configs/query_intelligence.yaml`**
```yaml
query_analysis:
  enabled: true
  vague_threshold: 0.5  # Below this = vague
  entity_extraction: true

reformulation:
  enabled: true
  max_sub_queries: 5

domain_knowledge:
  enabled: true
  knowledge_base: "data/knowledge/primo_domain_knowledge.yaml"
```

**4. `configs/conversation.yaml`** (Phase 8)
```yaml
conversation:
  enabled: false  # Toggle for conversational mode
  max_clarifications: 2
  timeout_minutes: 5
  save_history: true
```

---

## Documentation Updates

### Files to Update

1. **`docs/architecture/ADAPTIVE_RETRIEVAL.md`** (NEW)
   - Architecture diagram
   - Component descriptions
   - Data flow
   - Decision logic

2. **`docs/features/RERANKING.md`** (NEW)
   - How reranking works
   - Model selection guide
   - Performance impact
   - Configuration options

3. **`docs/features/QUERY_INTELLIGENCE.md`** (NEW)
   - Query analysis
   - Reformulation strategies
   - Domain knowledge injection
   - Examples and use cases

4. **`TESTING_GUIDE.md`** (UPDATE)
   - Add adaptive mode testing
   - Iteration analysis
   - Quality assessment metrics

5. **`README.md`** (UPDATE)
   - Add Phase 5-8 to feature list
   - Update architecture overview

---

## Dependencies

### New Python Packages

```toml
# Add to pyproject.toml

[tool.poetry.dependencies]
# For cross-encoder reranking
sentence-transformers = "^2.2.2"  # Includes cross-encoders

# Alternative: Cohere for reranking API
# cohere = "^4.37"  # Optional, if using Cohere Rerank

# For conversation state management
pydantic = "^2.5.0"  # Already have, for data models
```

### Installation

```bash
poetry add sentence-transformers
# OR if using Cohere:
# poetry add cohere
```

---

## Cost Analysis

### Estimated Additional Costs (OpenAI API)

**Phase 5 (Reranking):**
- Cross-encoder: Free (local model)
- OR Cohere API: ~$1-2 per 1K queries
- Quality assessment: ~$0.002 per query (gpt-4o-mini)
  - 1K queries/month = $2/month

**Phase 6 (Query Intelligence):**
- Query analysis: ~$0.002 per query
- Reformulation: ~$0.003 per reformulated query
- Estimated: $5-10/month for 1K queries

**Phase 7 (Adaptive Loop):**
- Multiple iterations: 1.5x cost on average (some queries iterate)
- Estimated: $15-20/month for 1K queries with 50% iteration rate

**Phase 8 (Conversational):**
- Clarification generation: ~$0.002 per clarification
- Minimal additional cost (only for unclear queries)

**Total Estimated Additional Cost:**
- $20-30/month for 1K queries/month
- Using gpt-4o-mini keeps costs low
- Main cost is quality assessment + reformulation

**Cost Optimization:**
- Use heuristics before LLM calls (saves 30-40%)
- Cache reformulations for common vague queries
- Limit iterations to 2-3 max

---

## Risk Analysis

### Technical Risks

1. **Reranking Latency**
   - Risk: Reranking 100 docs takes 1-2 seconds
   - Mitigation: Acceptable for this use case (not real-time chat)
   - Alternative: Use faster model or Cohere API

2. **Infinite Loops**
   - Risk: Adaptive loop never converges
   - Mitigation: Hard limit of 3 iterations, 30-second timeout
   - Monitoring: Track iteration counts in logs

3. **Over-Adaptation**
   - Risk: System changes query too much, loses original intent
   - Mitigation: Always compare to original query in assessment
   - Testing: Verify adapted queries are still relevant

4. **Quality Assessment Accuracy**
   - Risk: Quality agent misjudges retrieval quality
   - Mitigation: Combine LLM with heuristics
   - Validation: Manual review of 100 assessments

### Business Risks

1. **Increased Complexity**
   - Risk: System becomes harder to debug
   - Mitigation: Comprehensive logging of all decisions
   - Documentation: Clear architecture diagrams

2. **Cost Escalation**
   - Risk: Multiple LLM calls per query
   - Mitigation: Use mini model, cache results, heuristics
   - Monitoring: Track cost per query

3. **User Confusion** (Phase 8)
   - Risk: Users don't understand clarification questions
   - Mitigation: Clear UI design, examples
   - Testing: User testing before full rollout

---

## Future Enhancements (Beyond Phase 8)

**Possible Phase 9+ Ideas:**

1. **Learning from User Feedback**
   - Track which answers users find helpful
   - Use feedback to improve quality assessment
   - Fine-tune reranking based on implicit feedback

2. **Graph RAG Integration**
   - Build knowledge graph of email threads
   - Use graph traversal for "related emails"
   - Improve thread_summary queries

3. **Semantic Caching**
   - Cache embeddings for common query patterns
   - Cache reformulations for frequent vague queries
   - Reduce latency and cost

4. **Multi-Modal Support**
   - Extract and analyze attachments (PDFs, images)
   - Include attachment content in retrieval
   - Better context for technical discussions

5. **Advanced Analytics**
   - Dashboard showing query patterns
   - Retrieval quality trends over time
   - User behavior analysis

---

## Appendix: Example Flows

### Example 1: Specific Query (No Adaptation Needed)

**Query:** "How do I configure facets in Primo?"

**Flow:**
1. Query analysis: Specific (confidence: 0.9)
2. FAISS retrieve: 100 candidates
3. Rerank: Top 15
4. Quality assessment: Good (confidence: 0.85)
5. Decision: ACCEPT
6. Generate answer: ✅

**Iterations:** 1
**Time:** 3 seconds
**Outcome:** Excellent answer with links

---

### Example 2: Vague Query (Reformulation)

**Query:** "What happened with Primo VE last month?"

**Flow:**
1. Query analysis: Vague (confidence: 0.3)
2. FAISS retrieve: 100 candidates
3. Rerank: Top 15
4. Quality assessment: Low (confidence: 0.4)
   - Issues: ["too_generic", "mixed_topics"]
   - Suggestions: ["reformulate"]
5. Decision: REFORMULATE
6. Generate sub-queries:
   - "New features in Primo VE November 2025"
   - "Issues reported in Primo VE November 2025"
   - "Configuration changes in Primo VE November 2025"
7. Multi-query retrieval: 45 candidates total
8. Rerank: Top 15
9. Quality assessment: Good (confidence: 0.75)
10. Decision: ACCEPT
11. Generate answer: ✅

**Iterations:** 2
**Time:** 8 seconds
**Outcome:** Comprehensive answer covering features, issues, changes

---

### Example 3: Very Vague Query (Conversational - Phase 8)

**Query:** "Tell me about Primo"

**Flow:**
1. Query analysis: Extremely vague (confidence: 0.1)
2. FAISS retrieve: 100 candidates (very mixed)
3. Quality assessment: Very low (confidence: 0.2)
   - Issues: ["too_broad", "no_clear_intent"]
   - Suggestions: ["ask_clarification"]
4. Decision: ASK_CLARIFICATION
5. Generate questions:
   - "What aspect of Primo are you interested in?"
     - [ ] Features and capabilities
     - [ ] Recent issues and bugs
     - [ ] Configuration and setup
     - [ ] Training and documentation
6. **User selects:** "Recent issues and bugs"
7. Reformulate: "What are the recent issues and bugs in Primo?"
8. Restart retrieval with focused query → Success ✅

**Iterations:** 2 (1 clarification + 1 retrieval)
**Time:** 15 seconds (includes user response time)
**Outcome:** Focused answer about recent issues

---

## Contact & Questions

For questions about this plan:
- Review existing documentation in `docs/phases/`
- Check `docs/architecture/` for system design
- Refer to Phase 1-4 completion documents for context

---

**End of Document**
