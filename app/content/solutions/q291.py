def q291(tokens, vocabulary, end_token="<eot>", unknown_token="<unk>"):
    """Encode tokens with a supplied vocabulary containing unknown/end markers, map unseen
    tokens to unknown, and append the end marker.
    """
    return [vocabulary.get(t, vocabulary[unknown_token]) for t in tokens] + [
        vocabulary[end_token]
    ]
