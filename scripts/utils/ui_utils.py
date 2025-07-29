import streamlit as st


def render_answer_with_direction(answer: str):
    # Simple heuristic: Hebrew starts with Unicode range 0x0590–0x05FF
    is_rtl = any('\u0590' <= char <= '\u05ff' for char in answer)

    direction = "rtl" if is_rtl else "ltr"
    styled_html = f"""
    <div style='direction: {direction}; text-align: left; white-space: pre-wrap; font-size: 1.1em;'>
        {answer}
    </div>
    """
    st.markdown(styled_html, unsafe_allow_html=True)


def detect_text_direction(text: str) -> str:
    """
    Detects directionality of a string: 'rtl' for Hebrew, 'ltr' otherwise.
    """
    return "rtl" if any('\u0590' <= char <= '\u05ff' for char in text) else "ltr"


def render_text_block(text: str):
    """
    Renders a text block in Streamlit with automatic directionality (LTR/RTL).
    """
    direction = detect_text_direction(text)
    styled_html = f"""
    <div style='direction: {direction}; text-align: left; white-space: pre-wrap; font-size: 1.05em;'>
        {text}
    </div>
    """
    st.markdown(styled_html, unsafe_allow_html=True)
