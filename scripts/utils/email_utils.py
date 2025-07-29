import re


def clean_email_text(
    text: str,
    remove_quoted_lines: bool = True,
    remove_reply_blocks: bool = True,
    remove_signature: bool = True,
    signature_delimiter: str = "-- ",
) -> str:
    """
    Clean email text by removing quoted lines, reply blocks, and/or signatures.
    Args:
        text: The raw email text.
        remove_quoted_lines: Remove lines starting with '>' (quoted replies).
        remove_reply_blocks: Remove blocks starting with 'On ... wrote:' or 'From:'.
        remove_signature: Remove lines after the signature delimiter.
        signature_delimiter: The delimiter that marks the start of a signature (default: '-- ').
    Returns:
        Cleaned email text.
    """
    lines = text.splitlines()
    cleaned_lines = []
    in_signature = False
    in_reply_block = False
    reply_block_pattern = re.compile(r"^(On .+ wrote:|From: .+)$")

    for line in lines:
        # Remove signature
        if remove_signature and line.strip().startswith(signature_delimiter):
            in_signature = True
        if in_signature:
            continue

        # Remove reply blocks
        if remove_reply_blocks and reply_block_pattern.match(line.strip()):
            in_reply_block = True
        if in_reply_block:
            continue

        # Remove quoted lines
        if remove_quoted_lines and line.strip().startswith(">"):
            continue

        cleaned_lines.append(line)

    return "\n".join(cleaned_lines).strip()
