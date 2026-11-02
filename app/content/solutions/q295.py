def q295(text):
    """Create simple whitespace-marked BPE pre-tokens: prefix each whitespace-delimited
    word with the visible marker U+2581. This educational variant collapses whitespace
    runs.
    """
    return ["\u2581" + word for word in text.split()]
