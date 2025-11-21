# Automated Daily Email Extraction & Sync

**Priority:** MEDIUM
**Status:** Planning
**Target:** Continuous email database updates without manual intervention
**Estimated Effort:** 2-3 weeks

---

## 🎯 Problem Statement

**Current State:**
- Email ingestion requires manual execution via UI or CLI
- Users must remember to run ingestion regularly
- No automatic detection of new emails
- Database becomes stale without regular updates
- No notifications when sync fails or new emails are available

**Desired State:**
- Automatic daily (or configurable interval) email extraction
- Incremental sync - only fetch new emails since last run
- Error handling with notifications (email/Slack/webhooks)
- Configurable schedule (daily, hourly, weekly, custom cron)
- Status dashboard showing last sync time, email count, errors
- Works across platforms (Windows, Linux, macOS)
- Low resource usage when running in background

---

## 📋 Requirements

### Functional Requirements

1. **Scheduled Execution**
   - Run email ingestion at configurable intervals
   - Default: Daily at 2 AM local time
   - Support cron syntax for advanced scheduling
   - Respect timezone configuration

2. **Incremental Sync**
   - Track last sync timestamp per project
   - Only fetch emails newer than last sync
   - Handle timezone differences between email server and local system
   - Prevent duplicate processing with existing deduplication

3. **Multi-Project Support**
   - Each project can have independent sync schedule
   - Ability to enable/disable sync per project
   - Prioritize projects (sync critical projects first)

4. **Error Handling**
   - Retry logic with exponential backoff
   - Detailed error logging
   - Notification on repeated failures (3+ consecutive)
   - Graceful degradation (skip problematic emails, continue with rest)

5. **Notifications**
   - Success summary (X new emails indexed)
   - Error alerts (sync failed, credentials expired, etc.)
   - Channels: Email, Slack webhook, Discord, system notifications
   - Configurable verbosity (silent, errors-only, full report)

6. **Status Monitoring**
   - Web dashboard showing sync status
   - Last successful sync timestamp
   - Email count growth over time
   - Error history with details
   - Manual trigger button for immediate sync

### Non-Functional Requirements

1. **Performance**
   - Minimal CPU/memory usage when idle
   - Efficient incremental sync (< 30 seconds for typical daily updates)
   - Parallel processing for multiple projects
   - Rate limiting to respect email server limits

2. **Reliability**
   - Persistent state (survive system restarts)
   - Lock files to prevent concurrent runs
   - Transaction-like behavior (all-or-nothing per email)

3. **Security**
   - Secure credential storage (OS keychain integration)
   - No plaintext passwords in config files
   - Optional encryption for stored emails
   - Audit log of all sync operations

4. **Maintainability**
   - Clear logs with rotation policy
   - Health check endpoint for monitoring tools
   - Dry-run mode for testing
   - Easy enable/disable without config changes

---

## 🏗️ Architecture Design

### Components

```
┌─────────────────────────────────────────────────────────┐
│              Sync Scheduler (Main Daemon)               │
│  - Task queue management                                │
│  - Schedule parsing & execution                         │
│  - Health monitoring                                    │
└────────────┬───────────────────────────────────────────┘
             │
             ├──> SyncOrchestrator (per project)
             │     ├─> EmailFetcher (Outlook/IMAP/etc)
             │     ├─> IncrementalIngestionManager
             │     ├─> PipelineRunner (ingest→chunk→embed)
             │     └─> NotificationService
             │
             ├──> StateManager (persistent storage)
             │     └─> sync_state.db (SQLite)
             │          - project_id
             │          - last_sync_timestamp
             │          - last_email_date
             │          - status (success/failed)
             │          - email_count
             │          - error_message
             │
             └──> ConfigManager
                   └─> projects/*/sync_config.yml
```

### State Tracking Database

**Schema: `sync_state.db`**
```sql
CREATE TABLE sync_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_name TEXT NOT NULL,
    sync_start TIMESTAMP NOT NULL,
    sync_end TIMESTAMP,
    status TEXT CHECK(status IN ('running', 'success', 'failed')),
    emails_fetched INTEGER DEFAULT 0,
    emails_new INTEGER DEFAULT 0,
    emails_duplicate INTEGER DEFAULT 0,
    errors TEXT,
    triggered_by TEXT DEFAULT 'schedule'  -- 'schedule', 'manual', 'webhook'
);

CREATE TABLE project_state (
    project_name TEXT PRIMARY KEY,
    last_successful_sync TIMESTAMP,
    last_email_date TIMESTAMP,  -- Most recent email in database
    enabled BOOLEAN DEFAULT 1,
    next_scheduled_sync TIMESTAMP,
    consecutive_failures INTEGER DEFAULT 0
);
```

---

## 🔧 Implementation Plan

### Phase 1: Core Scheduler (Week 1)

#### 1.1 Sync Daemon
**File:** `scripts/automation/sync_daemon.py`

```python
class SyncDaemon:
    """Main daemon process for automated email sync."""

    def __init__(self, config_path: Path):
        self.state_manager = StateManager()
        self.scheduler = Scheduler()  # Using APScheduler
        self.projects = self._load_projects()

    def start(self):
        """Start the daemon (blocking)."""
        for project in self.projects:
            if project.sync_enabled:
                self._schedule_project(project)

        self.scheduler.start()
        self._run_health_check_loop()

    def _schedule_project(self, project):
        """Schedule a project's sync based on its config."""
        schedule = project.sync_config.get("schedule", "0 2 * * *")  # Daily at 2 AM
        self.scheduler.add_job(
            func=self._sync_project,
            trigger="cron",
            args=[project],
            **parse_cron(schedule),
            id=f"sync_{project.name}",
            replace_existing=True,
            misfire_grace_time=300  # 5 min grace period
        )
```

**Key Features:**
- APScheduler for reliable cron-based scheduling
- Persistent job store (survives restarts)
- Misfire handling (run missed jobs on startup)

#### 1.2 State Manager
**File:** `scripts/automation/state_manager.py`

```python
class StateManager:
    """Manage sync state in SQLite database."""

    def __init__(self, db_path: Path = None):
        self.db_path = db_path or Path("data/sync_state.db")
        self._init_db()

    def get_last_sync(self, project_name: str) -> dict:
        """Get last successful sync info."""
        return {
            "timestamp": datetime,
            "last_email_date": datetime,
            "email_count": int
        }

    def record_sync_start(self, project_name: str) -> int:
        """Record sync start, return sync_id."""

    def record_sync_end(self, sync_id: int, status: str, stats: dict):
        """Record sync completion with statistics."""
```

### Phase 2: Incremental Sync (Week 1-2)

#### 2.1 Enhanced Email Fetcher
**File:** `scripts/connectors/incremental_outlook_connector.py`

```python
class IncrementalOutlookConnector(OutlookConnector):
    """Outlook connector with incremental sync support."""

    def fetch_since(
        self,
        folder_path: str,
        since: datetime,
        max_count: int = None
    ) -> List[RawDoc]:
        """
        Fetch only emails received after 'since' timestamp.

        Uses Outlook filter: ReceivedTime >= since
        """
        items = folder.Items
        items.Sort("[ReceivedTime]", Descending=False)
        items = items.Restrict(f"[ReceivedTime] >= '{since.strftime('%m/%d/%Y %H:%M')}'")

        # Process items...
```

**IMAP Support:**
```python
class IncrementalIMAPConnector:
    """IMAP connector with incremental sync."""

    def fetch_since(self, since: datetime) -> List[RawDoc]:
        """Use IMAP SINCE search criteria."""
        date_str = since.strftime("%d-%b-%Y")
        status, messages = self.imap.search(None, f'SINCE {date_str}')
```

#### 2.2 Sync Orchestrator
**File:** `scripts/automation/sync_orchestrator.py`

```python
class SyncOrchestrator:
    """Orchestrates incremental sync for a single project."""

    def sync_project(self, project: ProjectManager) -> SyncResult:
        """
        Perform incremental sync:
        1. Get last sync timestamp
        2. Fetch new emails since timestamp
        3. Run pipeline (ingest → chunk → embed)
        4. Update state
        5. Send notification
        """
        state = self.state_manager.get_last_sync(project.name)
        since = state.get("last_email_date") or (datetime.now() - timedelta(days=30))

        sync_id = self.state_manager.record_sync_start(project.name)

        try:
            # Fetch new emails
            connector = self._get_connector(project)
            new_emails = connector.fetch_since(since)

            if not new_emails:
                return SyncResult(status="success", new_emails=0, message="No new emails")

            # Save to disk
            self._save_emails(project, new_emails)

            # Run pipeline
            runner = PipelineRunner(project, project.config)
            runner.add_step("ingest")
            runner.add_step("chunk")
            runner.add_step("embed")

            for msg in runner.run_steps():
                self.logger.info(msg)

            # Update state
            stats = {
                "emails_fetched": len(new_emails),
                "emails_new": len(runner.chunks),  # After deduplication
            }
            self.state_manager.record_sync_end(sync_id, "success", stats)

            return SyncResult(status="success", **stats)

        except Exception as e:
            self.state_manager.record_sync_end(sync_id, "failed", {"error": str(e)})
            raise
```

### Phase 3: Notifications & Monitoring (Week 2)

#### 3.1 Notification Service
**File:** `scripts/automation/notification_service.py`

```python
class NotificationService:
    """Send notifications via multiple channels."""

    def __init__(self, config: dict):
        self.channels = []

        if config.get("email"):
            self.channels.append(EmailNotifier(config["email"]))
        if config.get("slack_webhook"):
            self.channels.append(SlackNotifier(config["slack_webhook"]))
        if config.get("discord_webhook"):
            self.channels.append(DiscordNotifier(config["discord_webhook"]))

    def send_success(self, project_name: str, stats: dict):
        """Send success notification with stats."""
        message = f"""
        ✅ Email Sync Successful - {project_name}

        📧 New emails: {stats['emails_new']}
        🔄 Duplicates skipped: {stats['emails_duplicate']}
        ⏱️  Duration: {stats['duration_seconds']}s
        """
        self._send_all(message, level="info")

    def send_failure(self, project_name: str, error: str, consecutive_failures: int):
        """Send failure alert."""
        severity = "🚨 CRITICAL" if consecutive_failures >= 3 else "⚠️  Warning"
        message = f"""
        {severity} Email Sync Failed - {project_name}

        ❌ Error: {error}
        🔢 Consecutive failures: {consecutive_failures}

        Please check the logs or run manual sync.
        """
        self._send_all(message, level="error")
```

**Configuration:**
```yaml
# In project config.yml
sync:
  notifications:
    email:
      enabled: true
      recipients: ["admin@example.com"]
      on_success: false  # Only notify on errors
      on_failure: true

    slack_webhook: "https://hooks.slack.com/services/..."

    verbosity: "errors-only"  # "silent", "errors-only", "full"
```

#### 3.2 Web Dashboard
**File:** `scripts/ui/ui_sync_dashboard.py`

```python
def render_sync_dashboard():
    """Streamlit dashboard for sync monitoring."""

    st.header("📧 Email Sync Dashboard")

    # Overview cards
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Active Projects", len(active_projects))
    with col2:
        st.metric("Last Sync", last_sync_time)
    with col3:
        st.metric("Total Emails", total_emails)

    # Project status table
    st.subheader("Project Status")
    df = load_sync_status()
    st.dataframe(df)

    # Sync history chart
    st.subheader("Sync History (Last 30 Days)")
    chart_data = load_sync_history(days=30)
    st.line_chart(chart_data)

    # Manual trigger
    st.subheader("Manual Sync")
    selected_project = st.selectbox("Project", project_names)
    if st.button("Trigger Sync Now"):
        trigger_manual_sync(selected_project)
```

### Phase 4: Platform Support & Deployment (Week 3)

#### 4.1 Systemd Service (Linux)
**File:** `deployment/email-sync.service`

```ini
[Unit]
Description=RAG Platform Email Sync Daemon
After=network.target

[Service]
Type=simple
User=raguser
WorkingDirectory=/opt/rag-platform
ExecStart=/opt/rag-platform/.venv/bin/python -m scripts.automation.sync_daemon
Restart=on-failure
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

**Installation:**
```bash
sudo cp deployment/email-sync.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable email-sync
sudo systemctl start email-sync
```

#### 4.2 Windows Service
**File:** `deployment/install_windows_service.py`

```python
import win32serviceutil
import win32service

class EmailSyncService(win32serviceutil.ServiceFramework):
    _svc_name_ = "RAGEmailSync"
    _svc_display_name_ = "RAG Platform Email Sync"

    def SvcDoRun(self):
        from scripts.automation.sync_daemon import SyncDaemon
        daemon = SyncDaemon()
        daemon.start()
```

**Installation:**
```powershell
python deployment/install_windows_service.py install
sc start RAGEmailSync
```

#### 4.3 Docker Support
**File:** `docker-compose.sync.yml`

```yaml
version: '3.8'

services:
  email-sync:
    build: .
    command: python -m scripts.automation.sync_daemon
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
    environment:
      - TZ=America/New_York
      - OPEN_AI=${OPEN_AI}
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "python", "-m", "scripts.automation.health_check"]
      interval: 5m
      timeout: 30s
      retries: 3
```

---

## ⚙️ Configuration

### Project-Level Config
**File:** `data/projects/*/config.yml`

```yaml
sync:
  enabled: true

  schedule:
    type: "cron"  # or "interval"
    cron: "0 2 * * *"  # Daily at 2 AM
    # interval: "24h"  # Alternative: simple interval
    timezone: "America/New_York"

  incremental:
    enabled: true
    lookback_hours: 1  # Overlap to prevent missing emails

  retry:
    max_attempts: 3
    backoff_multiplier: 2  # 1s, 2s, 4s
    max_backoff_seconds: 60

  notifications:
    on_success: false
    on_failure: true
    channels: ["slack"]  # "email", "slack", "discord"

  limits:
    max_emails_per_sync: 1000
    max_duration_minutes: 30
    rate_limit_delay_seconds: 0.1  # Between emails
```

### Global Daemon Config
**File:** `config/sync_daemon.yml`

```yaml
daemon:
  log_level: "INFO"
  log_file: "logs/sync_daemon.log"
  log_rotation: "daily"
  log_retention_days: 30

  state_db: "data/sync_state.db"

  health_check:
    enabled: true
    port: 8765
    endpoint: "/health"

  concurrency:
    max_parallel_projects: 3
    thread_pool_size: 5

  notifications:
    email:
      smtp_host: "smtp.gmail.com"
      smtp_port: 587
      smtp_user: "${SMTP_USER}"
      smtp_password: "${SMTP_PASSWORD}"
      from_address: "noreply@example.com"

    slack_webhook: "${SLACK_WEBHOOK_URL}"
```

---

## 🧪 Testing Strategy

### Unit Tests

```python
# tests/automation/test_sync_orchestrator.py

def test_incremental_sync_with_new_emails(mock_outlook):
    """Test that only new emails are fetched and processed."""
    # Setup: Database has emails up to 2025-11-20
    state_manager.set_last_sync("test_project", datetime(2025, 11, 20))

    # Mock: Outlook has 5 new emails from 2025-11-21
    mock_outlook.fetch_since.return_value = [
        create_mock_email(date=datetime(2025, 11, 21, i))
        for i in range(5)
    ]

    # Execute
    result = orchestrator.sync_project(project)

    # Assert
    assert result.status == "success"
    assert result.new_emails == 5
    assert mock_outlook.fetch_since.called_with(datetime(2025, 11, 20))

def test_sync_with_no_new_emails():
    """Test that sync succeeds gracefully with no new emails."""
    mock_outlook.fetch_since.return_value = []

    result = orchestrator.sync_project(project)

    assert result.status == "success"
    assert result.new_emails == 0

def test_sync_failure_records_error():
    """Test that failures are properly recorded."""
    mock_outlook.fetch_since.side_effect = ConnectionError("Outlook not available")

    with pytest.raises(ConnectionError):
        orchestrator.sync_project(project)

    state = state_manager.get_project_state("test_project")
    assert state["consecutive_failures"] == 1
```

### Integration Tests

```python
# tests/automation/test_sync_daemon_integration.py

def test_scheduled_sync_executes():
    """Test that daemon executes scheduled sync."""
    # Setup: Schedule sync for 1 second in future
    daemon = SyncDaemon()
    daemon.schedule_project("test_project", interval_seconds=1)

    # Wait for execution
    time.sleep(2)

    # Assert
    sync_history = state_manager.get_sync_history("test_project", limit=1)
    assert len(sync_history) > 0
    assert sync_history[0]["status"] == "success"
```

---

## 📊 Success Metrics

### Performance Metrics

| Metric | Target | How to Measure |
|--------|--------|----------------|
| **Sync Duration** | < 30s for daily sync | `sync_end - sync_start` in logs |
| **CPU Usage (Idle)** | < 1% | `top` or `ps` monitoring |
| **Memory Usage** | < 100MB | Process memory inspection |
| **Email Processing Rate** | > 10 emails/sec | `emails_fetched / duration` |

### Reliability Metrics

| Metric | Target | How to Measure |
|--------|--------|----------------|
| **Sync Success Rate** | > 99% | `success_count / total_syncs` |
| **Duplicate Detection** | 100% accuracy | Manual audit of database |
| **Missing Emails** | 0 | Compare with email server |
| **Notification Delivery** | > 95% | Notification service logs |

---

## 🚀 Rollout Plan

### Stage 1: Beta Testing (Week 4)
- Deploy on 1-2 test projects
- Run for 1 week with manual verification
- Monitor for errors, missed emails, duplicates
- Collect user feedback on notification frequency

### Stage 2: Gradual Rollout (Week 5)
- Deploy to 25% of projects
- Monitor performance and error rates
- Enable notifications only on failures initially
- Document any edge cases

### Stage 3: Full Deployment (Week 6)
- Deploy to all projects
- Enable success notifications (opt-in)
- Publish user documentation
- Set up monitoring dashboard

---

## 📚 User Documentation

### Quick Start Guide

```markdown
# Enabling Automated Email Sync

1. Edit your project config: `data/projects/YOUR_PROJECT/config.yml`

2. Add sync configuration:
   ```yaml
   sync:
     enabled: true
     schedule:
       cron: "0 2 * * *"  # Daily at 2 AM
   ```

3. Start the sync daemon:
   ```bash
   # Linux/Mac
   poetry run python -m scripts.automation.sync_daemon

   # Or install as service
   sudo systemctl start email-sync
   ```

4. Monitor sync status:
   - Web dashboard: http://localhost:8501 → "Sync Dashboard" tab
   - Logs: `tail -f logs/sync_daemon.log`
   - Database: `sqlite3 data/sync_state.db "SELECT * FROM sync_history ORDER BY id DESC LIMIT 10"`
```

---

## 🔒 Security Considerations

1. **Credential Storage**
   - Never store plaintext passwords in config
   - Use OS keychain (macOS Keychain, Windows Credential Manager, Linux Secret Service)
   - Environment variables for CI/CD

2. **Access Control**
   - Daemon runs with minimal permissions
   - State database has restricted file permissions (600)
   - Logs don't contain sensitive data (email content)

3. **Network Security**
   - TLS for all email connections
   - Webhook URLs use HTTPS only
   - Rate limiting to prevent abuse

---

## 🔄 Future Enhancements

### Phase 2 Features (Later)

1. **Webhook Integration**
   - Real-time sync triggered by email server webhooks
   - Reduce latency from "next scheduled run" to seconds

2. **Multi-Source Sync**
   - Sync from multiple email accounts simultaneously
   - Gmail, Exchange, custom IMAP sources

3. **Intelligent Scheduling**
   - ML-based prediction of email arrival patterns
   - Adjust sync frequency dynamically
   - "Quiet hours" - skip sync during low-activity periods

4. **Conflict Resolution**
   - Handle emails modified after initial sync
   - Version control for email content
   - Merge strategies for metadata updates

5. **Advanced Monitoring**
   - Grafana dashboard with Prometheus metrics
   - Alerting rules (PagerDuty, Opsgenie)
   - Anomaly detection (unusual email volume, sync duration)

---

## 🛠️ Dependencies

**New Python Packages:**
```toml
[tool.poetry.dependencies]
APScheduler = "^3.10"  # Cron-based scheduling
watchdog = "^3.0"      # File system monitoring (optional)
pywin32 = { version = "^306", markers = "sys_platform == 'win32'" }  # Windows service
```

**Optional:**
```toml
prometheus-client = "^0.19"  # Metrics export
sentry-sdk = "^1.40"         # Error tracking
```

---

## 📝 Acceptance Criteria

- [ ] Daemon runs continuously without crashes for 7+ days
- [ ] Successfully syncs 3+ projects on daily schedule
- [ ] Incremental sync fetches only new emails (verified manually)
- [ ] Deduplication prevents duplicate indexing (100% accuracy)
- [ ] Notifications sent on failures within 5 minutes
- [ ] Dashboard shows real-time sync status
- [ ] Works on Windows and Linux
- [ ] System restart doesn't lose state
- [ ] Manual sync trigger works from UI
- [ ] Documentation complete with examples
- [ ] Unit test coverage > 80%
- [ ] Integration tests pass

---

**Status:** Ready for development
**Owner:** TBD
**Timeline:** 3 weeks (estimated)
**Priority:** Medium (after UI redesign)

---

## 🔗 Related Documents

- [Outlook Integration Plan](../automation/outlook_integration_plan.md)
- [Phase 4 Completion](../archive/EMAIL_PHASE4_COMPLETION.md)
- [UI Redesign Plan](UI_REDESIGN_PLAN.md)
