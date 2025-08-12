from email import policy
from email.parser import BytesParser
from pathlib import Path


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
