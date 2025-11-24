# Email RAG System - Architectural Overview

**Date:** 2025-11-22
**Status:** DRAFT - Architectural Planning
**Purpose:** Define abstraction layers, generalization strategy, and separation concerns for email-specific RAG functionality

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Current State Analysis](#current-state-analysis)
3. [Challenges Taxonomy](#challenges-taxonomy)
4. [Architectural Principles](#architectural-principles)
5. [Proposed Architecture](#proposed-architecture)
6. [Standalone Email RAG Consideration](#standalone-email-rag-consideration)
7. [Migration Path](#migration-path)
8. [Decision Matrix](#decision-matrix)

---

## Executive Summary

### The Core Tension

**Domain-Specific vs General-Purpose RAG**

We're building categorization for a **Primo VE mailing list**, but the underlying platform is a **general multi-source RAG system**. Adding domain-specific logic (Primo features, product context) risks:
- Breaking generalization
- Coupling email processing to a single use case
- Making the system unusable for other email sources

### Key Questions

1. **Generalization**: How do we categorize emails from different sources (mailing lists, personal email, support tickets)?
2. **Domain Context**: How do we inject domain knowledge (Primo features) without hardcoding it?
3. **Multi-Source**: How do we handle mixed email datasets (multiple mailing lists, different contexts)?
4. **Separation**: Should email RAG be a standalone project?

### Recommendation Preview

**Layered Architecture with Context Injection**

```
┌─────────────────────────────────────────────────────┐
│   Application Layer (Multi-Source RAG Platform)    │
├─────────────────────────────────────────────────────┤
│   Email RAG Module (Plugin/Extension)              │
│   ├─ Generic Email Processing (abstraction)        │
│   ├─ Context Providers (domain knowledge injection)│
│   └─ Email-Specific Strategies (categorization)    │
├─────────────────────────────────────────────────────┤
│   Core RAG Infrastructure (embeddings, retrieval)  │
└─────────────────────────────────────────────────────┘
```

**Decision:** Keep email RAG as a **module** for now, but architect it for **easy extraction** if complexity grows.

---

## Current State Analysis

### What We Have

#### Generic Components (Reusable)
```
scripts/ingestion/
  └─ email_loader.py         # Load EML, MSG, MBOX (general)
scripts/chunking/
  └─ chunker_v3.py           # Chunk emails by paragraphs (general)
scripts/embeddings/
  └─ unified_embedder.py     # Embed text chunks (general)
scripts/retrieval/
  └─ retrieval_manager.py    # Semantic search (general)
```

**Status:** ✅ These are properly abstracted and work for any email source

#### Domain-Specific Components (Primo-Focused)
```
scripts/categorization/
  └─ category_discovery.py   # K-means clustering (general algorithm)
      ├─ EMAIL_STOPWORDS     # ❌ 480 person names from Primo list
      ├─ _is_system_artifact()  # ❌ Outlook-specific (reactions, etc.)
      └─ generate_category_name() # ❌ No domain context in prompt
```

**Status:** ⚠️ These components are general algorithms but contain domain-specific data

#### Email-Specific Orchestration (Mailing List Focus)
```
scripts/agents/email/
  └─ email_orchestrator.py   # Intent detection, specialized retrievers
      ├─ IntentDetector      # ✅ General (detects query intent)
      ├─ SenderRetriever     # ✅ General (email metadata)
      ├─ TemporalRetriever   # ✅ General (date filtering)
      └─ ThreadRetriever     # ❌ Assumes mailing list threads
```

**Status:** ⚠️ Mostly general but assumes mailing list structure

### What's Missing

1. **Domain Context Abstraction**: No way to inject product-specific knowledge
2. **Email Source Metadata**: No distinction between mailing list, personal email, support tickets
3. **Multi-Source Handling**: Can't mix different email types with different categorization needs
4. **Configuration-Driven Context**: Domain knowledge is hardcoded, not configurable

---

## Challenges Taxonomy

### 1. Generalization Challenge

**Problem:** Building for "Primo VE mailing list" but need to support:
- Personal email (Gmail, Outlook)
- Support tickets (Zendesk, Jira)
- Company internal email
- Multiple mailing lists (different topics)
- Mixed sources (all of the above in one project)

**Current State:**
- ❌ Stopwords hardcoded for Primo participants
- ❌ Category naming assumes mailing list topics
- ❌ Thread detection assumes mailing list format

**Impact:**
- System breaks if applied to personal email
- Can't reuse categorization for different domains
- Each new email source requires code changes

### 2. Domain Context Challenge

**Problem:** "Research Assistant" needs context to categorize correctly

**Example:**
```
Email: "The Research Assistant guardrails are too restrictive"

Without Context → Category: "Research Support" ❌
With Context    → Category: "Feature: Research Assistant" ✅
```

**Current State:**
- ❌ No way to provide product glossary
- ❌ No domain-specific feature detection
- ❌ LLM categorization lacks context

**Impact:**
- Category names misleading (Research Support vs Research Assistant Feature)
- Can't distinguish product features from general topics
- Requires manual category renaming

### 3. Multi-Source Challenge

**Problem:** What if we ingest emails from multiple sources?

**Scenario:**
```
Project: "Work_Emails"
Sources:
  1. Primo VE mailing list (product discussions)
  2. Library staff email (operational communication)
  3. Vendor support tickets (technical issues)
```

**Questions:**
- Should these share categories or have separate taxonomies?
- How do we prevent "Research Assistant" from Primo mixing with "Research Assistant" (job title) from staff email?
- Do we categorize once across all sources, or per-source?

**Current State:**
- ❌ No source-level metadata
- ❌ No support for per-source categorization
- ❌ All emails mixed into single categorization

**Impact:**
- Can't handle mixed email datasets
- Category meanings become ambiguous
- Retrieval results confusing (mixed contexts)

### 4. Scale & Complexity Challenge

**Problem:** Email RAG is getting complex

**Current Email-Specific Components:**
```
scripts/
├─ agents/email/
│   ├─ email_orchestrator.py      # 450 lines
│   ├─ intent_detector.py         # 200 lines
│   ├─ sender_retriever.py        # 180 lines
│   ├─ temporal_retriever.py      # 220 lines
│   ├─ thread_retriever.py        # 190 lines
│   └─ multi_aspect_retriever.py  # 240 lines
├─ categorization/
│   ├─ category_discovery.py      # 500 lines
│   └─ evaluate_categories.py     # 300 lines
├─ connectors/
│   ├─ outlook_connector.py       # 800 lines
│   └─ outlook_helper_utils.py    # 600 lines
├─ utils/
│   └─ email_utils.py             # 400 lines
```

**Total:** ~4,000 lines of email-specific code (and growing)

**Impact:**
- Maintenance burden increasing
- Harder to understand for new developers
- Email features overshadowing core RAG functionality
- Potential for tighter coupling over time

### 5. Configuration vs Code Challenge

**Problem:** Domain knowledge currently lives in code

**Current:**
```python
# Hardcoded in category_discovery.py
EMAIL_STOPWORDS = set([
    'john', 'smith', 'mary', ...,  # 480 names
])

PRODUCT_FEATURES = None  # Doesn't exist!
```

**Desired:**
```yaml
# data/projects/Primo_List/email_config.yml
domain_context:
  product_name: "Primo VE"
  organization: "Ex Libris / Clarivate"

  product_features:
    - name: "Research Assistant"
      keywords: ["research assistant", "ai search", "guardrails"]
      description: "AI-powered search helper"

    - name: "NDE UI"
      keywords: ["nde", "new discovery", "ui transition"]
      description: "New Discovery Environment user interface"

  stopwords:
    source: "data/projects/Primo_List/primo_participants.txt"
    type: "person_names"
```

**Impact:**
- Can't reconfigure for new domains without code changes
- Domain knowledge scattered across codebase
- No clear separation between logic and data

---

## Architectural Principles

### 1. Separation of Concerns

**Principle:** Separate email-generic logic from domain-specific context

```
Email Processing (Generic)     Domain Context (Specific)
┌──────────────────────┐       ┌──────────────────────┐
│ Load email files     │       │ Product glossary     │
│ Parse headers/body   │       │ Feature list         │
│ Extract threads      │       │ Participant names    │
│ Chunk text           │       │ Domain stopwords     │
│ Embed chunks         │       │ Category taxonomy    │
│ Cluster embeddings   │       │ Business rules       │
└──────────────────────┘       └──────────────────────┘
         ↓                              ↓
         └──────────────┬───────────────┘
                        ↓
              Categorization Engine
              (Combines both layers)
```

**Implementation:** Context providers pattern

### 2. Dependency Injection

**Principle:** Don't hardcode domain knowledge - inject it

```python
# BAD (current)
class CategoryDiscovery:
    def __init__(self):
        self.stopwords = EMAIL_STOPWORDS  # Hardcoded

# GOOD (proposed)
class CategoryDiscovery:
    def __init__(self, context_provider: DomainContextProvider):
        self.context = context_provider
        self.stopwords = self.context.get_stopwords()
```

### 3. Strategy Pattern

**Principle:** Different email sources → different strategies

```python
class EmailCategorizationStrategy(ABC):
    @abstractmethod
    def categorize(self, email: Email) -> Category:
        pass

class MailingListStrategy(EmailCategorizationStrategy):
    """For mailing lists (threads, topics, discussions)"""
    def categorize(self, email):
        # Cluster by topic, extract threads
        pass

class SupportTicketStrategy(EmailCategorizationStrategy):
    """For support tickets (priority, issue type, status)"""
    def categorize(self, email):
        # Categorize by urgency, issue category
        pass

class PersonalEmailStrategy(EmailCategorizationStrategy):
    """For personal email (sender, importance, folder)"""
    def categorize(self, email):
        # Categorize by sender, keywords, importance
        pass
```

### 4. Plugin Architecture

**Principle:** Domain-specific handlers as plugins

```
Email RAG Core (Always Loaded)
├─ Generic email processing
├─ Base categorization algorithms
└─ Context provider interface

Plugins (Load on Demand)
├─ primo_ve_plugin.py       # Primo-specific features
├─ jira_ticket_plugin.py    # Jira support tickets
└─ gmail_personal_plugin.py # Personal Gmail
```

### 5. Configuration Over Code

**Principle:** Domain knowledge in config files, not Python code

```yaml
# Load from: data/projects/{project}/email_domain.yml
domain:
  type: "mailing_list"
  product: "Primo VE"
  context_provider: "scripts.email.contexts.ProductMailingListContext"

  features:
    - "Research Assistant"
    - "NDE UI"

  categorization_strategy: "topic_based"
  stopwords_file: "primo_participants.txt"
```

---

## Proposed Architecture

### Layer 1: Core Email Processing (Generic)

**Location:** `scripts/email/core/`

**Components:**
```python
# scripts/email/core/email_processor.py
class EmailProcessor:
    """Generic email processing (any email source)."""

    def load_emails(self, source: Path) -> List[Email]:
        """Load emails from file/folder."""
        pass

    def extract_metadata(self, email: Email) -> Dict:
        """Extract sender, date, subject, etc."""
        pass

    def extract_threads(self, emails: List[Email]) -> List[Thread]:
        """Detect conversation threads."""
        pass

    def chunk_emails(self, emails: List[Email]) -> List[Chunk]:
        """Split emails into chunks."""
        pass
```

**Principle:** No domain knowledge, works for ANY email source

### Layer 2: Domain Context Providers

**Location:** `scripts/email/contexts/`

**Base Interface:**
```python
# scripts/email/contexts/base.py
class DomainContextProvider(ABC):
    """Provides domain-specific knowledge for email categorization."""

    @abstractmethod
    def get_stopwords(self) -> Set[str]:
        """Return domain-specific stopwords (names, common terms)."""
        pass

    @abstractmethod
    def get_product_features(self) -> List[ProductFeature]:
        """Return known product features/components."""
        pass

    @abstractmethod
    def get_category_taxonomy(self) -> Dict:
        """Return expected category structure."""
        pass

    @abstractmethod
    def enhance_llm_prompt(self, base_prompt: str, context: Dict) -> str:
        """Add domain context to LLM prompts."""
        pass

    @abstractmethod
    def is_system_artifact(self, email: Email) -> bool:
        """Detect system-generated emails (notifications, auto-replies)."""
        pass
```

**Implementations:**

```python
# scripts/email/contexts/mailing_list.py
class MailingListContext(DomainContextProvider):
    """Generic mailing list context (any mailing list)."""

    def __init__(self, config_path: Path):
        self.config = self.load_config(config_path)

    def get_stopwords(self) -> Set[str]:
        # Load from config file
        stopwords_file = self.config['stopwords_file']
        return load_stopwords_from_file(stopwords_file)

    def enhance_llm_prompt(self, base_prompt: str, context: Dict) -> str:
        return f"""
        You are categorizing emails from a mailing list.
        Topic: {self.config['topic']}

        {base_prompt}
        """

# scripts/email/contexts/product_mailing_list.py
class ProductMailingListContext(MailingListContext):
    """Mailing list about a specific product (e.g., Primo VE)."""

    def get_product_features(self) -> List[ProductFeature]:
        return [
            ProductFeature(
                name=f['name'],
                keywords=f['keywords'],
                description=f['description']
            )
            for f in self.config['product_features']
        ]

    def enhance_llm_prompt(self, base_prompt: str, context: Dict) -> str:
        features = self.get_product_features()
        feature_list = "\n".join([f"- {f.name}: {f.description}" for f in features])

        return f"""
        You are categorizing emails from the {self.config['product_name']} mailing list.

        Product: {self.config['product_name']}
        Organization: {self.config['organization']}

        Known Product Features:
        {feature_list}

        When categorizing:
        - If emails discuss a product feature, name it "Feature: [feature_name]"
        - If emails are general discussions, use descriptive topic names

        {base_prompt}
        """

# scripts/email/contexts/personal_email.py
class PersonalEmailContext(DomainContextProvider):
    """Personal email context (Gmail, Outlook personal)."""

    def get_category_taxonomy(self) -> Dict:
        # Personal email categories are different
        return {
            'primary': ['work', 'personal', 'financial'],
            'secondary': ['receipts', 'newsletters', 'social']
        }

    def is_system_artifact(self, email: Email) -> bool:
        # Different system artifacts for personal email
        if 'newsletter' in email.subject.lower():
            return True
        if email.sender.endswith('noreply@'):
            return True
        return False
```

### Layer 3: Categorization Engine (Configurable)

**Location:** `scripts/email/categorization/`

```python
# scripts/email/categorization/engine.py
class EmailCategorizationEngine:
    """Main categorization engine with context injection."""

    def __init__(
        self,
        context_provider: DomainContextProvider,
        strategy: CategorizationStrategy
    ):
        self.context = context_provider
        self.strategy = strategy

    def discover_categories(
        self,
        emails: List[Email],
        n_categories: int
    ) -> CategoryMapping:
        """Discover categories using injected context."""

        # Filter system artifacts (context-aware)
        emails = [e for e in emails if not self.context.is_system_artifact(e)]

        # Load stopwords from context
        stopwords = self.context.get_stopwords()

        # Cluster emails (generic algorithm)
        clusters = self.strategy.cluster(emails, n_categories)

        # Extract keywords (TF-IDF with context stopwords)
        keywords = self.extract_keywords(clusters, stopwords)

        # Generate category names (with domain context)
        categories = self.generate_names(clusters, keywords)

        return categories

    def generate_names(
        self,
        clusters: List[EmailCluster],
        keywords: Dict[int, List[str]]
    ) -> Dict[int, str]:
        """Generate category names with domain context."""

        category_names = {}

        for cluster_id, cluster in enumerate(clusters):
            # Check if cluster matches a known product feature
            feature = self.context.match_product_feature(keywords[cluster_id])

            if feature:
                # Use feature name
                category_names[cluster_id] = f"Feature: {feature.name}"
            else:
                # Use LLM with enhanced prompt
                base_prompt = f"Keywords: {keywords[cluster_id]}"
                enhanced_prompt = self.context.enhance_llm_prompt(
                    base_prompt,
                    {'cluster': cluster}
                )
                category_names[cluster_id] = self.llm_generate(enhanced_prompt)

        return category_names
```

### Layer 4: Email Source Metadata

**Location:** Email chunk metadata

```python
# When ingesting emails, add source metadata
chunk.meta = {
    'doc_type': 'outlook_eml',
    'subject': email.subject,
    'sender': email.sender,

    # NEW: Source-level metadata
    'email_source': {
        'type': 'mailing_list',        # mailing_list, personal, support_ticket
        'name': 'Primo VE List',        # Human-readable name
        'domain': 'primo_ve',           # Domain identifier
        'context_provider': 'ProductMailingListContext'
    },

    # NEW: Categorization metadata (if performed)
    'category': {
        'name': 'Feature: Research Assistant',
        'cluster_id': 0,
        'confidence': 0.85,
        'source': 'context-aware',      # vs 'generic'
        'discovery_date': '2025-11-22'
    }
}
```

**Benefits:**
- Can filter retrieval by email source
- Can apply different categorization per source
- Can track which context was used

### Configuration File Structure

**Per-Project Email Domain Config:**

```yaml
# data/projects/Primo_List/email_domain.yml

# Email source identification
source:
  type: "mailing_list"
  name: "Primo VE Community List"
  description: "Discussion list for Primo VE library discovery system"

# Domain context provider
context:
  provider: "ProductMailingListContext"
  config:
    product_name: "Primo VE"
    organization: "Ex Libris / Clarivate"
    ecosystem:
      - "Alma"
      - "FOLIO"
      - "SLSP"

    # Product features (for category detection)
    product_features:
      - name: "Research Assistant"
        keywords: ["research assistant", "ai search", "guardrails", "chatgpt"]
        description: "AI-powered search assistance feature"

      - name: "NDE UI"
        keywords: ["nde", "new discovery environment", "ui transition", "cohort"]
        description: "New Discovery Environment user interface"

      - name: "Alma Integration"
        keywords: ["alma", "fulfillment", "slsp", "integration"]
        description: "Integration with Alma library management system"

      - name: "Custom Code Packages"
        keywords: ["code package", "javascript", "customization", "css"]
        description: "Custom UI code and styling"

    # Stopwords
    stopwords:
      person_names:
        source: "data/projects/Primo_List/primo_participants.txt"
        description: "List participants (extracted from emails)"

      mailing_list_artifacts:
        - "primo"
        - "listserv"
        - "unsubscribe"
        - "digest"

      common_words:
        - "re"
        - "fwd"
        - "thanks"
        - "regards"

    # System artifact detection rules
    system_artifacts:
      patterns:
        - "reacted to your message"
        - "out of office"
        - "automatic reply"
        - "delivery failure"

      min_content_length: 50

# Categorization strategy
categorization:
  strategy: "topic_based"  # topic_based, sender_based, temporal, hierarchical

  algorithm: "kmeans"  # kmeans, bertopic, hdbscan

  parameters:
    n_categories: 5
    min_cluster_size: 10

  keyword_extraction: "tfidf"  # tfidf, frequency, llm

  naming_strategy: "context_aware_llm"  # llm, feature_detection, hybrid

# Multi-source handling (if mixing email sources)
multi_source:
  enabled: false

  # If enabled, categorize per-source or globally?
  scope: "per_source"  # per_source, global
```

**For Different Email Sources:**

```yaml
# data/projects/Work_Email/email_domain.yml

source:
  type: "personal"
  name: "Work Gmail Account"

context:
  provider: "PersonalEmailContext"
  config:
    categorization_focus: "sender_and_topic"

    important_senders:
      - "boss@company.com"
      - "hr@company.com"

    expected_categories:
      - "Work: Projects"
      - "Work: Administrative"
      - "Personal"
      - "Newsletters"
      - "Receipts"

categorization:
  strategy: "sender_based"
  algorithm: "kmeans"
  parameters:
    n_categories: 5
```

---

## Standalone Email RAG Consideration

### Should Email RAG Be Separated?

**Current Complexity:**
- ~4,000 lines of email-specific code
- 6 specialized agents (orchestrator, intent, retrievers)
- Custom categorization pipeline
- Outlook connector (complex integration)
- Growing feature set (threads, temporal, categorization)

### Pros of Separation

#### 1. Focus & Specialization
- Email RAG becomes a dedicated system
- Can optimize deeply for email use cases
- Clearer scope and responsibilities

#### 2. Independent Evolution
- Can evolve email features without affecting core RAG
- Can version separately (email-rag v2.0 vs core-rag v1.5)
- Can have different release cycles

#### 3. Reusability
- Can be used as a library by other projects
- Email-specific features packaged together
- Clearer API boundary

#### 4. Maintainability
- Separate codebase easier to understand
- Email domain experts can focus on one repo
- Reduced cognitive load

### Cons of Separation

#### 1. Duplication
- Need to duplicate core RAG infrastructure (embeddings, retrieval)
- Or create dependency (email-rag depends on core-rag)
- Either way, more complexity

#### 2. Integration Overhead
- Need clear API contract between systems
- Versioning compatibility issues
- Deployment becomes more complex

#### 3. Data Management
- Need to coordinate data between systems
- Shared storage or API-based access?
- More complex backup/restore

#### 4. Premature Separation
- Only ~4,000 lines of code (not huge yet)
- Still exploring email features
- Separation could slow innovation

### Recommendation: Modular Separation (Middle Ground)

**Keep in same repo, but architect for extraction:**

```
Multi-Source_RAG_Platform/
├─ scripts/
│   ├─ core/           # Core RAG (embeddings, retrieval, prompting)
│   │
│   ├─ email/          # Email RAG Module (can be extracted later)
│   │   ├─ __init__.py
│   │   ├─ core/       # Generic email processing
│   │   ├─ contexts/   # Domain context providers
│   │   ├─ agents/     # Email-specific agents
│   │   ├─ categorization/  # Categorization pipeline
│   │   └─ connectors/ # Email source connectors
│   │
│   ├─ ingestion/      # Multi-format ingestion
│   ├─ chunking/       # Text chunking
│   └─ ui/             # Streamlit UI
```

**Key Principles:**
1. **Clear module boundary:** All email code in `scripts/email/`
2. **Minimal dependencies:** Email module only depends on core RAG
3. **Plugin interface:** Email module registers as a plugin
4. **Separate config:** Email domain configs in project-specific files
5. **Independent tests:** Email tests can run separately

**When to Separate:**
- Email code exceeds 10,000 lines
- Multiple projects using only email RAG functionality
- Email RAG needs different tech stack (e.g., specialized email database)
- Team grows and needs dedicated email RAG developers

---

## Migration Path

### Phase 1: Refactor for Modularity (Current → Modular)

**Goal:** Organize email code into clear module structure

**Tasks:**
1. Create `scripts/email/` directory structure
2. Move email-specific code into module:
   - `agents/email/` → `email/agents/`
   - `categorization/` → `email/categorization/`
   - `utils/email_utils.py` → `email/utils.py`
3. Create clear module boundaries
4. Update imports throughout codebase

**Timeline:** 1-2 days
**Risk:** Low (just reorganization)

### Phase 2: Implement Context Providers (Modular → Context-Aware)

**Goal:** Extract domain knowledge into configuration

**Tasks:**
1. Create `DomainContextProvider` interface
2. Implement `MailingListContext` and `ProductMailingListContext`
3. Create `email_domain.yml` config for Primo_List
4. Update `CategoryDiscovery` to use context providers
5. Extract stopwords to config file
6. Update LLM prompts to use context

**Timeline:** 2-3 days
**Risk:** Medium (changes categorization logic)

### Phase 3: Multi-Source Support (Context-Aware → Multi-Source)

**Goal:** Support multiple email sources in one project

**Tasks:**
1. Add `email_source` metadata to chunks
2. Implement per-source categorization
3. Update retrieval to filter by source
4. Create UI for source selection
5. Test with mixed email dataset

**Timeline:** 3-4 days
**Risk:** Medium (new feature, needs testing)

### Phase 4: Strategy Pattern (Multi-Source → Pluggable)

**Goal:** Support different categorization strategies

**Tasks:**
1. Define `CategorizationStrategy` interface
2. Implement `TopicBasedStrategy` (current approach)
3. Implement `SenderBasedStrategy` (for personal email)
4. Implement `TicketStrategy` (for support tickets)
5. Make strategy configurable per email source

**Timeline:** 4-5 days
**Risk:** Medium (new abstraction layer)

### Phase 5: Optional - Extract to Standalone (Pluggable → Standalone)

**Goal:** Create separate Email RAG package

**Tasks:**
1. Create new repo: `email-rag`
2. Extract `scripts/email/` module
3. Define API contract with core RAG
4. Create Python package
5. Publish to PyPI
6. Update main RAG to import email-rag

**Timeline:** 1-2 weeks
**Risk:** High (major architectural change)
**Trigger:** Only if complexity demands it

---

## Decision Matrix

### For Current Primo VE Use Case

| Approach | Complexity | Generalization | Maintainability | Time to Implement |
|----------|-----------|----------------|-----------------|-------------------|
| **Status Quo** (hardcoded domain knowledge) | Low | ❌ None | ❌ Hard | ✅ 0 days |
| **Context Providers** (config-driven) | Medium | ✅ Good | ✅ Good | ⚠️ 2-3 days |
| **Multi-Source** (per-source categorization) | High | ✅ Excellent | ✅ Good | ⚠️ 5-6 days |
| **Standalone** (separate package) | Very High | ✅ Excellent | ⚠️ Complex | ❌ 2 weeks |

**Recommendation for NOW:** **Context Providers** (Phase 2)
- Solves domain knowledge problem (Research Assistant → Feature: Research Assistant)
- Enables reuse for other mailing lists
- Manageable implementation effort (2-3 days)
- Doesn't require full separation

### For Future Multi-Source Use Case

**Scenario:** Mix Primo emails + personal emails + support tickets

| Requirement | Solution Needed |
|-------------|----------------|
| Different categorization per source | ✅ Multi-Source Support (Phase 3) |
| Source-specific domain knowledge | ✅ Context Providers (Phase 2) |
| Different categorization strategies | ✅ Strategy Pattern (Phase 4) |
| Independent email RAG system | ⚠️ Standalone (Phase 5) - only if very complex |

**Recommendation:** Implement Phases 2-4, defer Phase 5 until proven necessary

---

## Concrete Next Steps

### Immediate (Next Session)

**Implement Context Provider for Primo VE (Phase 2 Start)**

1. Create `scripts/email/contexts/base.py` with `DomainContextProvider` interface
2. Create `scripts/email/contexts/product_mailing_list.py` with `ProductMailingListContext`
3. Create `data/projects/Primo_List/email_domain.yml` config
4. Update `category_discovery.py` to accept context provider
5. Test with existing Primo dataset

**Expected Outcome:**
- Category name: "Research Support" → "Feature: Research Assistant"
- Stopwords loaded from config file (not hardcoded)
- LLM prompt includes product context

**Time:** 3-4 hours

### Short-term (This Week)

6. Extract person names to `primo_participants.txt`
7. Add product features to config (Research Assistant, NDE UI, etc.)
8. Implement feature detection logic
9. Re-run discovery and evaluate improvements
10. Document new configuration format

**Expected Outcome:**
- Feature categories correctly identified
- Config-driven categorization working
- Documentation for adding new domains

**Time:** 1-2 days

### Medium-term (Next 2 Weeks)

**Complete Phase 2 (Context-Aware) + Start Phase 3 (Multi-Source)**

11. Implement additional context providers (PersonalEmailContext, etc.)
12. Add email source metadata to chunks
13. Test with different email sources
14. Implement per-source categorization option
15. Update UI to show source-based filtering

**Expected Outcome:**
- Can handle multiple email sources in one project
- Each source can have different domain context
- Retrieval can filter by source

**Time:** 1-2 weeks

---

## Conclusion

### Summary of Challenges

1. **Domain Knowledge Problem**: "Research Assistant" needs product context ✅ Solvable with context providers
2. **Generalization Problem**: Hardcoded for Primo VE ✅ Solvable with config-driven approach
3. **Multi-Source Problem**: Can't mix different email types ✅ Solvable with source metadata
4. **Complexity Problem**: Email RAG growing complex ⚠️ Monitor, separate if needed

### Recommended Architecture

**Layered, Context-Driven, Plugin-Ready:**

```
┌─────────────────────────────────────────────────────────┐
│  Application: Multi-Source RAG Platform                │
├─────────────────────────────────────────────────────────┤
│  Email RAG Module (scripts/email/)                     │
│  ├─ Core: Generic email processing                     │
│  ├─ Contexts: Domain knowledge providers (pluggable)   │
│  ├─ Strategies: Categorization strategies (pluggable)  │
│  └─ Agents: Email-specific orchestration               │
├─────────────────────────────────────────────────────────┤
│  Configuration: Per-project domain configs (YAML)      │
├─────────────────────────────────────────────────────────┤
│  Core RAG: Embeddings, retrieval, prompting            │
└─────────────────────────────────────────────────────────┘
```

**Key Principles:**
- **Abstraction**: Generic processing + pluggable domain context
- **Configuration**: Domain knowledge in config files, not code
- **Modularity**: Clear boundaries, minimal coupling
- **Extensibility**: New domains = new config + new context provider (no code changes to core)

### Answer to Original Question

> "What I'm worried about is the possibility of generalization."

**Answer:** The architecture above addresses this by:

1. **Separating generic from specific**: Email processing logic is generic, domain knowledge is injected
2. **Configuration-driven**: New domains don't require code changes, just new configs
3. **Pluggable contexts**: Each email source can have its own context provider
4. **Multi-source ready**: Can handle mixed email datasets with different contexts
5. **Extraction-ready**: If email RAG needs separation, module boundaries are clear

**The "Research Assistant" problem is solvable** with context providers. The system can be general-purpose while still supporting domain-specific categorization.

---

**Document Status:** DRAFT - Ready for review and implementation
**Next Action:** Implement Phase 2 (Context Providers) to validate architecture
**Review Date:** After Phase 2 implementation
