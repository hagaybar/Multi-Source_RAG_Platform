from email import policy
from email.parser import BytesParser
from pathlib import Path
import mailbox
try:
    import extract_msg
    HAS_MSG_SUPPORT = True
except ImportError:
    HAS_MSG_SUPPORT = False

try:
    from libratom.lib.pst import PSTFile
    HAS_PST_SUPPORT = True
except ImportError:
    HAS_PST_SUPPORT = False


def load_eml(path: str | Path) -> tuple[str, dict]:
    """
    Return (body_text, metadata) for one .eml file.
    Metadata = {"source": str(path), "content_type": "email"}.
    """
    path = Path(path)
    with path.open("rb") as fp:
        msg = BytesParser(policy=policy.default).parse(fp)

    # Prefer text/plain parts
    text = ""
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == "text/plain":
                text = part.get_content().strip()
                break
    else:
        text = msg.get_body(preferencelist=("plain",)).get_content().strip()

    return text, {
        "source": str(path),
        "content_type": "email",
        "doc_type": "eml",  # Required for rule selection
    }


def load_msg(path: str | Path) -> tuple[str, dict]:
    """
    Return (body_text, metadata) for one .msg file (Outlook message).
    Metadata = {"source": str(path), "content_type": "email"}.
    """
    if not HAS_MSG_SUPPORT:
        raise ImportError("extract-msg library is required for .msg file support. Install with: pip install extract-msg")
    
    path = Path(path)
    
    try:
        msg = extract_msg.Message(str(path))
        
        # Extract text content, prefer plain text over HTML
        text = ""
        if msg.body:
            text = msg.body.strip()
        elif msg.htmlBody:
            # Basic HTML stripping - could be enhanced with BeautifulSoup if needed
            import re
            text = re.sub(r'<[^>]+>', '', msg.htmlBody).strip()
        
        # Extract metadata
        metadata = {
            "source": str(path),
            "content_type": "email",
            "doc_type": "msg",  # Required for rule selection
            "subject": msg.subject or "",
            "sender": msg.sender or "",
            "date": str(msg.date) if msg.date else "",
        }
        
        msg.close()
        return text, metadata
        
    except Exception as e:
        raise RuntimeError(f"Failed to parse MSG file {path}: {e}") from e


def load_mbox(path: str | Path) -> list[tuple[str, dict]]:
    """
    Return list of (body_text, metadata) for all messages in an .mbox file.
    Each message returns metadata = {"source": str(path), "content_type": "email"}.
    """
    path = Path(path)
    messages = []
    
    try:
        mbox = mailbox.mbox(str(path))
        
        for i, message in enumerate(mbox):
            # Extract text content
            text = ""
            if message.is_multipart():
                for part in message.walk():
                    if part.get_content_type() == "text/plain":
                        payload = part.get_payload(decode=True)
                        if payload:
                            text = payload.decode('utf-8', errors='ignore').strip()
                            break
            else:
                payload = message.get_payload(decode=True)
                if payload:
                    text = payload.decode('utf-8', errors='ignore').strip()
            
            # Extract metadata
            metadata = {
                "source": str(path),
                "content_type": "email",
                "doc_type": "mbox",  # Required for rule selection
                "message_index": i,
                "subject": message.get("Subject", ""),
                "sender": message.get("From", ""),
                "date": message.get("Date", ""),
                "message_id": message.get("Message-ID", f"mbox_{i}"),
            }
            
            if text:  # Only add messages with content
                messages.append((text, metadata))
        
        mbox.close()
        return messages
        
    except Exception as e:
        raise RuntimeError(f"Failed to parse MBOX file {path}: {e}") from e


def load_pst(path: str | Path) -> list[tuple[str, dict]]:
    """
    Return list of (body_text, metadata) for all messages in a .pst file.
    Each message returns metadata = {"source": str(path), "content_type": "email"}.
    """
    if not HAS_PST_SUPPORT:
        raise ImportError("libratom library is required for .pst file support. Install with: pip install libratom")
    
    path = Path(path)
    messages = []
    
    try:
        with PSTFile(str(path)) as pst:
            for i, message in enumerate(pst.messages()):
                # Extract text content
                text = ""
                if hasattr(message, 'body_plain') and message.body_plain:
                    text = message.body_plain.strip()
                elif hasattr(message, 'body_html') and message.body_html:
                    # Basic HTML stripping
                    import re
                    text = re.sub(r'<[^>]+>', '', message.body_html).strip()
                
                # Extract metadata
                metadata = {
                    "source": str(path),
                    "content_type": "email",
                    "doc_type": "pst",  # Required for rule selection
                    "message_index": i,
                    "subject": getattr(message, 'subject', '') or "",
                    "sender": getattr(message, 'sender_name', '') or "",
                    "date": str(getattr(message, 'delivery_time', '')) or "",
                    "folder": getattr(message, 'folder_name', '') or "",
                }
                
                if text:  # Only add messages with content
                    messages.append((text, metadata))
        
        return messages
        
    except Exception as e:
        raise RuntimeError(f"Failed to parse PST file {path}: {e}") from e
