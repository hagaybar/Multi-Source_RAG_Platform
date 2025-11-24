# Master Roadmap - Multi-Source RAG Platform
**Last Updated:** 2025-11-21
**Status:** Active Planning Document

---

## 📋 Executive Summary

This document consolidates all future development plans for the Multi-Source RAG Platform. Plans are prioritized and organized by implementation timeline.

**Current Status:**
- ✅ Email Agentic Strategy Phase 1-4: **COMPLETE** (252 tests, production-ready)
- ✅ Outlook Integration Phase 1-5: **COMPLETE** (production-ready)
- 🔄 Next Priority: UI Redesign (HIGH) or Documentation (HIGH)

---

## 🎯 Priority Matrix

| Priority | Feature | Timeline | Effort | Status | Impact |
|----------|---------|----------|--------|--------|--------|
| **P0 (Critical)** | Documentation Suite | 1 week | 3 hours | Ready | High user value |
| **P1 (High)** | UI Redesign - Quick Wins | 2 weeks | 1 week | Ready | User experience |
| **P1 (High)** | UI Redesign - Full | 8 weeks | 6-8 weeks | Planned | User experience |
| **P2 (Medium)** | Automated Email Sync | 3 weeks | 2-3 weeks | Planned | Automation |
| **P3 (Low)** | Advanced Features | TBD | Varies | Ideas | Enhancement |
| **P4 (Deferred)** | Outlook CLI Validation | When needed | 1-2 hours | Backlog | Nice-to-have |

---

## 🚀 Active Plans (Ready for Implementation)

### P0: Documentation Suite ⭐ **START FIRST**
**Timeline:** 1 week (3 hours total effort)
**Priority:** CRITICAL
**Status:** Ready to implement
**Owner:** TBD

#### Rationale
- Current docs scattered across multiple files
- No user-friendly getting started guide
- High ROI (3 hours effort, major user impact)
- Reduces support burden

#### Deliverables

**1. USER_GUIDE.md** (45 min)
```markdown
# User Guide
- Introduction and prerequisites
- Quick start walkthrough
- Common workflows:
  - Creating projects
  - Ingesting emails
  - Querying with natural language
  - Understanding results
- Best practices and optimization tips
```

**2. TROUBLESHOOTING.md** (30 min)
```markdown
# Troubleshooting Guide
- Common issues and solutions
- Diagnostic steps for errors
- Error message explanations
- Contact support information
```

**3. FAQ.md** (20 min)
```markdown
# Frequently Asked Questions
- General questions (What is this? Who is it for?)
- Setup questions (Requirements? Installation?)
- Usage questions (How to query? Supported formats?)
- Technical questions (Performance? Limitations?)
```

**4. ARCHITECTURE.md** (30 min)
```markdown
# Architecture Overview
- System diagram (components and data flow)
- Component descriptions:
  - Ingestion pipeline
  - Chunking strategies
  - Embedding & indexing
  - Retrieval strategies
  - Email orchestrator
- Cross-platform considerations
- Design decisions and rationale
```

**5. DEPLOYMENT_GUIDE.md** (30 min)
```markdown
# Deployment Guide
- System requirements (OS, Python version, dependencies)
- Installation methods:
  - UI setup wizard
  - Manual installation
  - Docker deployment
- Configuration reference (config.yml)
- Upgrade procedures
```

**6. CHANGELOG.md** (15 min)
```markdown
# Changelog
## v1.1.0 (2025-11-21)
- Email Agentic Strategy Phase 1-4
- Dynamic top_k adjustment
- LLM-enhanced features
- 252 tests passing

## v1.0.0 (Initial)
- Core RAG pipeline
- Outlook integration
- Multi-format support
```

**7. README_QUICKSTART.md** (15 min)
```markdown
# Quick Start
- 5-minute setup guide
- First query example
- Links to detailed docs
```

#### Success Metrics
- Setup time: 30+ min → <10 min (for 90% of users)
- Support requests: Baseline → 50% reduction
- User satisfaction: Unknown → >4.5/5 stars
- Documentation coverage: 40% → 90%

#### Implementation Steps
1. Create template structure (15 min)
2. Fill in content for each guide (2.5 hours)
3. Add diagrams (15 min - optional)
4. Review and polish (15 min)

---

### P1: UI Redesign Plan 🎨
**Timeline:** 8 weeks (4 phases)
**Priority:** HIGH
**Status:** Planned
**Owner:** TBD
**Depends on:** None (can start immediately)

#### Problem Statement
**Current Issues:**
- UI optimized for debugging, not end users
- Too much technical information exposed
- Not intuitive for simple questions
- Poor mobile experience
- No progressive disclosure

**User Pain Points:**
- "I don't understand what 'intent detection' means"
- "Too much technical jargon"
- "Just want a simple answer"
- "Can't use on my phone"

#### Solution Overview
Transform UI from debug tool → user-friendly assistant

**Design Principles:**
1. **Simple by default, powerful when needed**
2. **Conversational interface** (like ChatGPT)
3. **Visual clarity** (hierarchy, spacing, typography)
4. **Mobile-first** (responsive design)
5. **Performance feedback** (loading states, progress)

#### Phase 1: Quick Wins + Core Redesign (Week 1-2) ⭐
**Effort:** 1 week
**Impact:** IMMEDIATE user experience improvement

**Quick Wins (Implementable in 1 day):**
- [ ] Hide debug info by default
  - Move intent/strategy/metadata behind "Show Details" toggle
  - Show only answer and sources
- [ ] Better answer formatting
  - Add markdown rendering
  - Format lists properly
  - Add section headings
- [ ] Source attribution
  - Show email titles/dates instead of chunk IDs
  - Add "View Email" links
  - Group by thread
- [ ] User-friendly error messages
  - Replace technical errors with friendly messages
  - Add retry buttons
  - Show partial results when possible

**Core Redesign:**
- [ ] Simplified search interface
  - Single search box (no advanced options by default)
  - Auto-complete suggestions
  - Recent queries dropdown
- [ ] Clean answer display component
  - Card-based layout
  - Clear hierarchy (question → answer → sources)
  - Expandable sections
- [ ] Progressive disclosure toggle
  - "Advanced Details" button
  - Shows: intent, strategy, metadata, debug info
- [ ] Mobile-responsive layout
  - Single column on mobile
  - Touch-friendly buttons
  - Collapsible sections

#### Phase 2: Smart Components (Week 3-4)
**Effort:** 2 weeks
**Impact:** Specialized views for different query types

**Specialized Result Views:**
- [ ] Action Items view
  ```
  ☐ Review budget proposal
     Due: Friday | Assigned: Bob
     From: Alice - Budget Email (Jan 15)
  ```
- [ ] Decisions view
  ```
  ✓ Approved $50K budget
    Decided by: Sarah Wilson
    Date: Jan 15, 2025
    Source: Budget Approval Email
  ```
- [ ] Thread Summary view
  ```
  📧 8 emails from Jan 13-20
  👥 Participants: Alice, Bob, Sarah
  Key Points: [bullets]
  [Read Full Thread →]
  ```

**Additional Components:**
- [ ] Source attribution cards (email preview)
- [ ] Email preview modals (click to expand)
- [ ] Suggested questions carousel
- [ ] Query history sidebar

#### Phase 3: Enhanced UX (Week 5-6)
**Effort:** 2 weeks
**Impact:** Polished user experience

- [ ] Auto-complete and suggestions
  - Based on query history
  - Based on project data
  - Smart suggestions (e.g., sender names, dates)
- [ ] Query history
  - Persistent history (localStorage)
  - Quick re-run of previous queries
  - Search through history
- [ ] Loading states
  - Contextual progress messages:
    - "Analyzing your question..."
    - "Searching 1,234 emails..."
    - "Finding relevant discussions..."
    - "Assembling answer..."
- [ ] Error handling improvements
  - User-friendly messages
  - Recovery suggestions
  - Partial results display

#### Phase 4: Polish (Week 7-8)
**Effort:** 2 weeks
**Impact:** Professional finish

- [ ] Animations and transitions
  - Smooth page transitions
  - Loading animations
  - Expand/collapse animations
- [ ] Dark mode support
  - Toggle in settings
  - Persistent preference
  - System preference detection
- [ ] Accessibility improvements
  - ARIA labels for screen readers
  - Keyboard navigation (Tab, Enter, Esc)
  - Focus indicators
  - Alt text for images
- [ ] User testing and refinement
  - 5-10 user tests
  - Iterate based on feedback
  - A/B testing for key features

#### Success Metrics
- **Time to answer:** ~2 min → <30 seconds
- **User satisfaction:** Unknown → >4.5/5 stars
- **Task completion rate:** Unknown → >90%
- **Mobile usage:** 0% → >30% of sessions
- **Technical knowledge required:** High → None (simple mode), Low (advanced mode)

#### Technical Stack
- **Framework:** Keep existing Streamlit (or migrate to React/Svelte)
- **Styling:** Tailwind CSS for rapid prototyping
- **Components:** shadcn/ui or Material-UI
- **State Management:** Streamlit session state (or Zustand/Redux)

#### Rollout Strategy
1. **Week 1-2:** Deploy Phase 1 to beta users
2. **Week 3-4:** Gather feedback, deploy Phase 2
3. **Week 5-6:** Iterate, deploy Phase 3
4. **Week 7-8:** Final polish, full release

---

### P2: Automated Email Sync 🔄
**Timeline:** 3 weeks
**Priority:** MEDIUM
**Status:** Planned
**Owner:** TBD
**Depends on:** None (can start anytime)

#### Problem Statement
**Current Issues:**
- Email ingestion requires manual execution via UI or CLI
- Users must remember to run ingestion regularly
- No automatic detection of new emails
- Database becomes stale without updates
- No notifications when sync fails or emails are ready

#### Solution Overview
Automated daemon that syncs emails on a schedule (daily, hourly, custom cron)

**Key Features:**
- ⏰ Scheduled sync (configurable intervals)
- 🔄 Incremental sync (only new emails)
- 📊 Web dashboard (status monitoring)
- 🔔 Notifications (Email, Slack, Discord)
- 🖥️ Cross-platform (Windows, Linux, macOS)
- 💾 Persistent state (survives restarts)

#### Architecture

```
SyncDaemon (APScheduler)
  ├─> SyncOrchestrator (per project)
  │    ├─> IncrementalEmailFetcher (Outlook/IMAP)
  │    ├─> PipelineRunner (ingest→chunk→embed)
  │    └─> NotificationService (alerts)
  ├─> StateManager (SQLite for persistence)
  │    └─> sync_state.db
  │         - Last sync timestamp
  │         - Email counts
  │         - Error history
  └─> HealthCheckEndpoint (monitoring)
```

#### Phase 1: Core Scheduler (Week 1)
**Effort:** 1 week

**Components:**
- [ ] **SyncDaemon** (`scripts/automation/sync_daemon.py`)
  - APScheduler for cron-based scheduling
  - Job persistence (survive restarts)
  - Misfire handling
  - Health check endpoint

- [ ] **StateManager** (`scripts/automation/state_manager.py`)
  - SQLite database (`data/sync_state.db`)
  - Tables:
    - `sync_history` (sync runs with stats)
    - `project_state` (last sync per project)
  - CRUD operations for state

**Configuration:**
```yaml
# Project config.yml
sync:
  enabled: true
  schedule:
    cron: "0 2 * * *"  # Daily at 2 AM
    timezone: "America/New_York"
  incremental:
    enabled: true
    lookback_hours: 1  # Overlap to prevent gaps
```

#### Phase 2: Incremental Sync (Week 1-2)
**Effort:** 1 week

**Components:**
- [ ] **IncrementalOutlookConnector** (`scripts/connectors/incremental_outlook_connector.py`)
  - `fetch_since(datetime)` method
  - Outlook filter: `ReceivedTime >= since`
  - Handle timezone differences

- [ ] **IncrementalIMAPConnector** (`scripts/connectors/incremental_imap_connector.py`)
  - IMAP SINCE search criteria
  - Support for Gmail, Exchange

- [ ] **SyncOrchestrator** (`scripts/automation/sync_orchestrator.py`)
  - Coordinate sync workflow:
    1. Get last sync timestamp
    2. Fetch new emails since timestamp
    3. Run pipeline (ingest→chunk→embed)
    4. Update state
    5. Send notification
  - Error handling with retries
  - Transaction-like behavior (all-or-nothing)

#### Phase 3: Notifications & Monitoring (Week 2)
**Effort:** 5-7 days

**Components:**
- [ ] **NotificationService** (`scripts/automation/notification_service.py`)
  - Multi-channel support:
    - Email (SMTP)
    - Slack (webhook)
    - Discord (webhook)
    - System notifications (desktop)
  - Success summaries (X new emails indexed)
  - Failure alerts (with details and retry count)
  - Configurable verbosity (silent, errors-only, full)

- [ ] **Web Dashboard** (`scripts/ui/ui_sync_dashboard.py`)
  - Overview cards:
    - Active projects
    - Last sync time
    - Total emails indexed
  - Project status table
  - Sync history chart (30-day)
  - Manual trigger button

**Configuration:**
```yaml
# Project config.yml
sync:
  notifications:
    on_success: false
    on_failure: true
    channels: ["slack"]
    slack_webhook: "${SLACK_WEBHOOK_URL}"
```

#### Phase 4: Platform Deployment (Week 3)
**Effort:** 3-5 days

**Linux (systemd):**
- [ ] Service definition (`deployment/email-sync.service`)
- [ ] Installation script
- [ ] Auto-start on boot
- [ ] Log rotation

**Windows (Service):**
- [ ] Service wrapper with pywin32
- [ ] Installation script (`deployment/install_windows_service.py`)
- [ ] Service registration
- [ ] Event log integration

**Docker:**
- [ ] Docker Compose config (`docker-compose.sync.yml`)
- [ ] Health check configuration
- [ ] Volume mounts for data persistence
- [ ] Environment variable support

#### Success Metrics
- **Sync duration:** < 30 seconds for typical daily updates
- **CPU usage (idle):** < 1%
- **Memory usage:** < 100MB
- **Success rate:** > 99%
- **Notification delivery:** > 95%

#### Future Enhancements (Phase 2)
- Webhook integration for real-time sync
- Multi-source sync (Gmail, Exchange, IMAP)
- ML-based intelligent scheduling
- Advanced monitoring (Grafana, Prometheus)
- Conflict resolution for modified emails

---

## 📚 Backlog (Deferred or Low Priority)

### Outlook CLI Validation Tool
**Priority:** P4 (Deferred)
**Effort:** 1-2 hours
**Status:** Implement when automation becomes priority

**Purpose:** Enable scripting and CI/CD integration

**Features:**
- Basic validation check command
- Auto-fix for common issues
- JSON output for automation
- Verbose diagnostics mode

**Command:**
```bash
python -m scripts.tools.outlook_helper_check --project Primo_List --json
```

**Use Cases:**
- Pre-flight checks before extraction
- Health monitoring in scheduled jobs
- CI/CD pipeline integration
- Batch validation

**Decision:** Add when users request automation features

---

### Advanced Testing Strategy
**Priority:** P4 (Deferred)
**Effort:** 2-3 hours
**Status:** Add tests as edge cases discovered

**Current Coverage:** 252 tests (100% passing)

**Additional Test Areas:**
- Integration tests (end-to-end workflows)
- Manual test checklist (setup, upgrades)
- Edge case coverage (network issues, corrupted emails)

**Decision:** Current coverage is excellent; add tests reactively

---

### Runtime OpenAI Model Selection
**Priority:** P3 (Low)
**Effort:** 2-3 hours
**Status:** Future enhancement
**Requested:** 2025-11-23

**Purpose:** Allow users to choose OpenAI completion model at runtime for testing and cost optimization

**Features:**
- **UI Model Selector:** Dropdown in Streamlit UI to select model
  - GPT-4o (current default)
  - GPT-4o-mini (faster, cheaper)
  - GPT-3.5-turbo (cheapest)
  - Custom model support

- **CLI Flag:** `--model gpt-4o-mini` parameter
  ```bash
  python -m scripts.pipeline.runner ask \
    --project Primo_List \
    --query "What topics were discussed?" \
    --model gpt-3.5-turbo
  ```

- **Session Persistence:** Remember model choice within session
- **Cost Display:** Show estimated cost per query based on model
- **Performance Comparison:** Side-by-side comparison mode

**Implementation:**
```python
# In OpenAICompleter
def __init__(self, model: str = None):
    self.model = model or os.getenv("DEFAULT_LLM_MODEL", "gpt-4o")

# In UI
model_choice = st.selectbox(
    "LLM Model",
    ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"],
    help="Choose model - affects cost and quality"
)
```

**Benefits:**
- Testing: Compare model performance on same queries
- Cost optimization: Use cheaper models for simple queries
- Experimentation: Test new models as they're released
- Development: Faster iteration with mini models

**Use Cases:**
- Developer testing different prompts
- Cost-conscious users for bulk queries
- Comparing model capabilities
- Testing before production deployment

**Decision:** Implement when users need multi-model testing workflows

---

### Corpus Expansion & Image Features
**Priority:** P3 (Low)
**Effort:** 4 weeks
**Status:** Ideas (needs refinement)

**Original Plan from "Second Month":**
- Expand corpus to 50-100 documents
- Image-aware setup (detect embedded images)
- OCR extraction (tesseract)
- XLSX-aware chunker v2
- Agent hub skeleton

**Decision:** Most features not needed currently. Consider:
- OCR extraction: Useful if users have image-heavy docs
- XLSX improvements: Useful if table data is complex

**Recommendation:** Revisit when users request these features

---

## 💡 Feature Ideas (Not Planned)

These ideas have been discussed but not yet planned in detail:

### Email-Specific Features
- [ ] **Attachment Search** - Search within email attachments (PDF, DOCX, XLSX)
- [ ] **Thread Visualization** - Visual graph of email conversation threads
- [ ] **Smart Summaries** - Auto-generate summaries for long threads (GPT-4)
- [ ] **Email Templates** - Common response templates based on query intent

### Platform Enhancements
- [ ] **Multi-Language Support** - Translate UI, support non-English emails
- [ ] **API Gateway** - RESTful API for external integrations
- [ ] **Mobile App** - Native iOS/Android apps
- [ ] **Browser Extension** - Quick email search from browser toolbar

### Advanced RAG Features
- [ ] **Semantic Caching** - Cache frequently asked questions (reduce LLM cost)
- [ ] **Feedback Loop** - User ratings to improve retrieval quality
- [ ] **Citation Tracking** - Track which sources are most useful
- [ ] **Answer Streaming** - Stream LLM responses for better UX

### Integration Features
- [ ] **Slack Bot** - Query emails directly from Slack
- [ ] **Microsoft Teams Bot** - Teams interface for email search
- [ ] **Zapier Integration** - Connect to 1000+ apps via Zapier
- [ ] **Calendar Integration** - Connect with Outlook/Google Calendar events

**Decision:** Track ideas, prioritize based on user demand

---

## 🗑️ Obsolete Plans (Completed or No Longer Relevant)

### ✅ Email Agentic Strategy Phase 1-4
**Status:** COMPLETE (merged to main on 2025-11-21)
**Location:** `docs/automation/EMAIL_AGENTIC_STRATEGY_*.md`

**What was delivered:**
- ✅ Phase 1: Intent Detection (7 types, LLM fallback)
- ✅ Phase 2: Specialized Retrievers (sender, temporal, thread, multi-aspect)
- ✅ Phase 3: Context Assembly (email cleaning, deduplication)
- ✅ Phase 4: Quality Enhancements (validation, action/decision extraction)
- ✅ Critical integration fix (EmailOrchestratorAgent now active)
- ✅ Dynamic top_k adjustment (10-20 chunks based on intent)
- ✅ 252 tests (100% passing)

**Action:** Archive original plan documents, keep completion reports

### ✅ Outlook Integration Phase 1-5
**Status:** COMPLETE
**Location:** `docs/automation/OUTLOOK_*.md`

**What was delivered:**
- ✅ Phase 1-5: Outlook connector with WSL support
- ✅ Email extraction and ingestion
- ✅ Multi-format support (MSG, MBOX, EML)
- ✅ UI setup wizard

**Action:** Keep documentation, mark as complete

---

## 📋 Implementation Priority Queue

### This Week (Critical)
1. **Documentation Suite** - 3 hours, massive user impact
   - Create USER_GUIDE.md
   - Create TROUBLESHOOTING.md
   - Create FAQ.md
   - Create ARCHITECTURE.md
   - Create DEPLOYMENT_GUIDE.md
   - Create CHANGELOG.md

### Next 2 Weeks (High Priority)
2. **UI Redesign Phase 1** - Quick wins + core redesign
   - Hide debug info by default (1 day)
   - Better answer formatting (1 day)
   - Simplified search interface (2-3 days)
   - Mobile-responsive layout (2-3 days)

### Next 1-2 Months (Medium Priority)
3. **UI Redesign Phases 2-3** - Smart components + UX
   - Specialized result views (1 week)
   - Auto-complete and history (1 week)
   - Loading states and error handling (3-4 days)

4. **Automated Email Sync** - All phases
   - Core scheduler (1 week)
   - Incremental sync (1 week)
   - Notifications + dashboard (5-7 days)
   - Platform deployment (3-5 days)

### Long-term (3+ Months)
5. **UI Redesign Phase 4** - Polish
6. **Feature ideas** - Based on user feedback
7. **Backlog items** - As needed

---

## 📊 Resource Allocation

**Documentation Suite:**
- **Owner:** TBD
- **Effort:** 3 hours
- **Dependencies:** None
- **Risk:** Low

**UI Redesign:**
- **Owner:** TBD (Front-end developer)
- **Effort:** 6-8 weeks
- **Dependencies:** None
- **Risk:** Medium (user testing needed)

**Automated Email Sync:**
- **Owner:** TBD (Backend developer)
- **Effort:** 2-3 weeks
- **Dependencies:** None
- **Risk:** Low (well-scoped)

---

## 🎯 Success Criteria

### Documentation
- [ ] Setup time < 10 minutes for 90% of users
- [ ] Support requests reduced by 50%
- [ ] User satisfaction > 4.5/5 stars

### UI Redesign
- [ ] Time to answer < 30 seconds
- [ ] User satisfaction > 4.5/5 stars
- [ ] Mobile usage > 30% of sessions

### Automated Email Sync
- [ ] Sync duration < 30 seconds
- [ ] Success rate > 99%
- [ ] CPU usage < 1% (idle)

---

## 📞 Next Steps

1. **Review this roadmap** with stakeholders
2. **Assign owners** to P0 and P1 items
3. **Create GitHub issues** for each feature
4. **Start with Documentation Suite** (immediate high ROI)
5. **Parallel track:** UI Redesign Phase 1 (quick wins)
6. **Monthly review:** Adjust priorities based on user feedback

---

**Last Updated:** 2025-11-21
**Next Review:** 2025-12-21
