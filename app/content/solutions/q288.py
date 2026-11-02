import unicodedata


def q288(text):
    """Normalize Unicode text with NFKC, lowercase it, and collapse all whitespace to
    single spaces. Strip leading and trailing whitespace.
    """
    return " ".join(unicodedata.normalize("NFKC", text).lower().split())
