import re


def q289(text):
    """Tokenize Unicode text into word runs and individual non-whitespace punctuation
    symbols. Preserve case and token order.
    """
    return re.findall("\\w+|[^\\w\\s]", text, flags=re.UNICODE)
