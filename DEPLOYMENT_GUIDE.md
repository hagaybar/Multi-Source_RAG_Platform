# Deployment Guide

**Version:** 1.1.0
**Last Updated:** 2025-11-23

This guide covers installation, configuration, and deployment of the Multi-Source RAG Platform.

---

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Installation Methods](#installation-methods)
3. [Configuration Reference](#configuration-reference)
4. [Environment Variables](#environment-variables)
5. [Running the Application](#running-the-application)
6. [Upgrading](#upgrading)
7. [Production Deployment](#production-deployment)

---

## System Requirements

### Hardware

**Minimum:**
- CPU: 2 cores
- RAM: 8GB
- Disk: 5GB free space

**Recommended:**
- CPU: 4+ cores
- RAM: 16GB
- Disk: 50GB+ free space (for large document collections)

### Software

**Required:**
- **Operating System:** Linux, macOS, or Windows (with WSL2)
- **Python:** 3.10 or higher (3.11+ recommended)
- **Poetry:** 1.5+ (for dependency management)

**Optional:**
- **Outlook:** Windows desktop app (for email integration)
- **Docker:** For containerized deployment (future)

### API Access

**Required:**
- **OpenAI API Key** - Get at https://platform.openai.com/api-keys
- **Billing enabled** on OpenAI account

**Optional:**
- **LiteLLM** account (for alternative LLM providers)

---

## Installation Methods

### Method 1: Poetry (Recommended)

**1. Install Poetry**
```bash
# Linux/macOS
curl -sSL https://install.python-poetry.org | python3 -

# Windows (PowerShell)
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | py -

# Verify
poetry --version
```

**2. Clone Repository**
```bash
git clone <repository-url>
cd Multi-Source_RAG_Platform
```

**3. Install Dependencies**
```bash
# Install all dependencies
poetry install

# Activate virtual environment
poetry shell
```

**4. Set Up Environment**
```bash
# Copy example env file
cp .env.example .env

# Edit .env and add your OpenAI API key
nano .env  # or vim, code, etc.

# Should contain:
# OPENAI_API_KEY=sk-proj-your-key-here
```

**5. Verify Installation**
```bash
# Test import
python -c "import scripts; print('Success!')"

# Check CLI
python -m app.cli --help
```

---

### Method 2: pip (Alternative)

**1. Create Virtual Environment**
```bash
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate  # Windows
```

**2. Install from requirements**
```bash
pip install -r requirements.txt  # If available
# or
pip install poetry
poetry export -f requirements.txt --output requirements.txt
pip install -r requirements.txt
```

---

### Method 3: Development Setup

**For contributors and developers:**

```bash
# Clone with development branch
git clone -b develop <repository-url>
cd Multi-Source_RAG_Platform

# Install with dev dependencies
poetry install --with dev

# Install pre-commit hooks
pre-commit install

# Run tests
pytest
```

---

## Configuration Reference

### Project Configuration (`config.yml`)

**Location:** `data/projects/<project_name>/config.yml`

**Full Template:**

```yaml
# Project Info
project:
  name: My_Project
  description: Project description here
  language: en  # or 'he' for Hebrew

# Paths (relative to project root)
paths:
  input_dir: input
  output_dir: output
  raw_dir: raw
  faiss_dir: output/faiss
  metadata_dir: output/metadata
  logs_dir: output/logs

# LLM Configuration
llm:
  provider: openai          # openai, litellm, azure, etc.
  model: gpt-4o-mini        # gpt-4o-mini, gpt-4o, gpt-3.5-turbo
  temperature: 0.4          # 0.0 (deterministic) to 1.0 (creative)
  max_tokens: 400           # Maximum response length
  prompt_strategy: auto     # NEW v1.1.0: auto, email, default, v2

# Embedding Configuration
embedding:
  provider: litellm              # litellm, openai, local
  model: text-embedding-3-large  # Recommended
  endpoint: https://api.openai.com/v1/embeddings
  mode: batch                    # batch or streaming
  use_async_batch: true          # Async batch processing (faster)
  embed_batch_size: 64           # Chunks per batch
  skip_duplicates: true          # Skip already-embedded chunks
  image_enrichment: false        # Enable image descriptions

# Agent Configuration (Optional)
agents:
  enable_image_insight: false    # AI image descriptions
  image_agent_model: gpt-4o      # Model for image analysis
  image_prompt: "..."            # Custom image prompt
  output_mode: append_to_chunk   # How to add image descriptions

# Data Sources (Optional)
sources:
  outlook:
    enabled: true
    account_name: user@example.com
    folder_path: "Inbox > Project Emails"
    days_back: 30
    max_emails: 500              # null for unlimited
    include_attachments: false
```

### Chunking Rules (`configs/chunk_rules.yaml`)

**Location:** `configs/chunk_rules.yaml` (global for all projects)

**Template:**

```yaml
# PDF Documents
pdf:
  strategy: by_paragraph
  min_tokens: 50
  max_tokens: 300
  overlap: 20

# Word Documents
docx:
  strategy: by_paragraph
  min_tokens: 50
  max_tokens: 300
  overlap: 20

# PowerPoint
pptx:
  strategy: by_slide
  min_tokens: 30
  max_tokens: 250
  overlap: 10

# Emails
outlook_eml:
  strategy: by_email_block
  min_tokens: 20
  max_tokens: 300
  overlap: 5

eml:
  strategy: by_email_block
  min_tokens: 20
  max_tokens: 300
  overlap: 5

# Plain Text
txt:
  strategy: by_paragraph
  min_tokens: 50
  max_tokens: 300
  overlap: 20
```

**Strategy Options:**
- `by_paragraph` - Split on blank lines, merge to meet token limits
- `by_email_block` - Email-specific splitting
- `by_page` - Keep page boundaries (for PDFs)
- `by_slide` - One or more slides per chunk (for PPTX)

---

## Environment Variables

### Required Variables

**`.env` file:**

```bash
# OpenAI API Key (REQUIRED)
OPENAI_API_KEY=sk-proj-your-key-here

# Optional: Set log level
LOG_LEVEL=INFO  # DEBUG, INFO, WARNING, ERROR

# Optional: Custom LiteLLM base URL
LITELLM_BASE_URL=https://api.openai.com/v1
```

### Optional Variables

```bash
# Azure OpenAI (if using Azure)
AZURE_OPENAI_API_KEY=your-azure-key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/

# Custom embedding endpoint
EMBEDDING_ENDPOINT=https://custom-embedding-api.com/

# Proxy settings (if needed)
HTTP_PROXY=http://proxy.example.com:8080
HTTPS_PROXY=http://proxy.example.com:8080
```

---

## Running the Application

### Streamlit UI (Recommended)

```bash
# Activate environment
poetry shell

# Start UI
streamlit run scripts/ui/ui_v3.py

# Opens automatically at http://localhost:8501
```

**Custom Port:**
```bash
streamlit run scripts/ui/ui_v3.py --server.port 8080
```

**Remote Access:**
```bash
streamlit run scripts/ui/ui_v3.py \
  --server.port 8501 \
  --server.address 0.0.0.0
```

---

### CLI Usage

**Create Project:**
```bash
poetry run python -m app.cli create-project \
  --name "My_Project" \
  --description "Testing RAG"
```

**Run Pipeline:**
```bash
poetry run python -m app.cli run-pipeline \
  --project data/projects/My_Project \
  --steps ingest chunk embed
```

**Query:**
```bash
poetry run python -m app.cli ask \
  --project data/projects/My_Project \
  --query "What are the main findings?"
```

---

### Python Script

```python
from pathlib import Path
from scripts.core.project_manager import ProjectManager
from scripts.pipeline.runner import PipelineRunner

# Load project
project = ProjectManager(Path("data/projects/My_Project"))

# Run pipeline
runner = PipelineRunner(project, project.config)
runner.add_step('ingest')
runner.add_step('chunk')
runner.add_step('embed')

for message in runner.run_steps():
    print(message)

# Query
query = "What are the main findings?"
for message in runner.step_retrieve(query):
    print(message)

for message in runner.step_ask(query):
    print(message)

print(runner.last_answer)
```

---

## Upgrading

### From v1.0.0 to v1.1.0

**1. Pull Latest Code**
```bash
git pull origin main  # or feature/email-categorization
```

**2. Update Dependencies**
```bash
poetry lock
poetry install
```

**3. No Configuration Changes Required**
- All v1.1.0 features are backward compatible
- Existing projects work without modification

**4. Optional: Enable New Features**
```yaml
# config.yml
llm:
  prompt_strategy: auto  # Or email, default, v2
```

**5. Restart Application**
```bash
# Stop UI (Ctrl+C)
# Restart
streamlit run scripts/ui/ui_v3.py
```

**New Features in v1.1.0:**
- ✅ Pipeline Validator (automatic)
- ✅ Smart Disk Fallback (automatic)
- ✅ Config Respect (set `prompt_strategy`)
- ✅ Email Agentic Strategy (automatic for email projects)

---

### General Upgrade Procedure

```bash
# 1. Backup your projects
cp -r data/projects data/projects.backup

# 2. Pull latest code
git pull origin main

# 3. Check for migration notes
cat CHANGELOG.md | head -50

# 4. Update dependencies
poetry lock
poetry install

# 5. Run any migration scripts (if provided)
# poetry run python scripts/migrations/<migration_script>.py

# 6. Test with one project
# Run a simple query and verify it works

# 7. Restart application
```

---

## Production Deployment

### Deployment Checklist

**Before Deployment:**
- [ ] All tests passing (`pytest`)
- [ ] Dependencies locked (`poetry.lock` committed)
- [ ] `.env` configured with production API keys
- [ ] Logs configured (check `LOG_LEVEL`)
- [ ] Disk space sufficient (check `df -h`)
- [ ] Backup strategy in place

**Security:**
- [ ] `.env` not in git (check `.gitignore`)
- [ ] API keys rotated if needed
- [ ] File permissions set correctly (`chmod 600 .env`)
- [ ] Firewall configured (if exposing UI remotely)

**Performance:**
- [ ] Async batch embedding enabled
- [ ] Deduplication enabled
- [ ] Chunk sizes optimized
- [ ] Index sharding if >100K documents

---

### Systemd Service (Linux)

**Create service file:**
```bash
sudo nano /etc/systemd/system/rag-platform.service
```

**Content:**
```ini
[Unit]
Description=Multi-Source RAG Platform
After=network.target

[Service]
Type=simple
User=your-user
WorkingDirectory=/path/to/Multi-Source_RAG_Platform
Environment="PATH=/path/to/venv/bin"
ExecStart=/path/to/venv/bin/streamlit run scripts/ui/ui_v3.py --server.port 8501
Restart=always

[Install]
WantedBy=multi-user.target
```

**Enable and start:**
```bash
sudo systemctl daemon-reload
sudo systemctl enable rag-platform
sudo systemctl start rag-platform
sudo systemctl status rag-platform
```

---

### Nginx Reverse Proxy (Optional)

**For remote access with HTTPS:**

```nginx
server {
    listen 80;
    server_name rag.example.com;

    location / {
        proxy_pass http://localhost:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }
}
```

**Enable HTTPS:**
```bash
sudo certbot --nginx -d rag.example.com
```

---

### Monitoring

**Log Monitoring:**
```bash
# Application logs
tail -f data/projects/*/logs/app/app.log

# Streamlit logs
tail -f ~/.streamlit/streamlit.log

# System logs
journalctl -u rag-platform -f
```

**Disk Usage:**
```bash
# Check FAISS index sizes
du -sh data/projects/*/output/faiss/

# Check total project size
du -sh data/projects/
```

**Performance Monitoring:**
```bash
# Python process
ps aux | grep streamlit

# Memory usage
free -h

# Disk I/O
iostat -x 1
```

---

### Backup & Recovery

**What to Backup:**
```bash
# Full project backup
tar -czf backup-$(date +%Y%m%d).tar.gz data/projects/

# Config only
tar -czf configs-$(date +%Y%m%d).tar.gz data/projects/*/config.yml configs/

# Critical data only (indices + metadata)
tar -czf indices-$(date +%Y%m%d).tar.gz data/projects/*/output/
```

**Restore:**
```bash
# Extract backup
tar -xzf backup-20251123.tar.gz

# Verify
ls data/projects/

# Test with query
poetry run python -m app.cli ask --project data/projects/My_Project --query "test"
```

---

### Troubleshooting Deployment

**Port Already in Use:**
```bash
# Find process using port 8501
lsof -i :8501

# Kill process
kill <PID>

# Or use different port
streamlit run scripts/ui/ui_v3.py --server.port 8080
```

**Permission Denied:**
```bash
# Fix file permissions
chmod -R 755 data/projects/
chmod 600 .env
```

**Out of Memory:**
```bash
# Check memory usage
free -h

# Increase swap (temporary)
sudo swapoff -a
sudo dd if=/dev/zero of=/swapfile bs=1G count=8
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

---

## Next Steps

**After Deployment:**
1. Test with known-good data
2. Monitor logs for errors
3. Set up regular backups
4. Configure monitoring/alerting
5. Document your specific setup

**For Production Use:**
- See [USER_GUIDE.md](USER_GUIDE.md) for usage best practices
- See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for common issues
- See [ARCHITECTURE.md](ARCHITECTURE.md) for system details

---

**Need help?** Check documentation or create a GitHub issue.
