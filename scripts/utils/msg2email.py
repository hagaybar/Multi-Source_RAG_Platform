import extract_msg, email, email.policy, pathlib


def msg_to_eml(msg_path: str | pathlib.Path, eml_path: str | pathlib.Path):
    m = extract_msg.Message(msg_path)
    eml = email.message.EmailMessage()
    eml["Subject"] = m.subject
    eml["From"] = m.sender
    eml["To"] = ", ".join(m.to)
    eml.set_content(m.body)
    with open(eml_path, "wb") as f:
        email.generator.BytesGenerator(f, policy=email.policy.default).flatten(eml)
