def q597(chunks):
    """Parse an iterable of arbitrary UTF-8 byte chunks into SSE data strings. Decode
    incrementally; accept LF, CRLF, or CR line endings, comments, and multi-line data
    fields. Dispatch only on blank lines; discard an unfinished final event per SSE
    semantics. Provider sentinels/JSON are left as data strings.
    """
    import codecs
    import itertools

    # A UTF-8 character may be split across transport chunks; retain decoder state.
    decoder = codecs.getincrementaldecoder("utf-8")()
    line = ""
    pending_cr = False
    data = []
    events = []

    def finish_line(text):
        if text == "":
            if data:
                events.append("\n".join(data))
                data.clear()
        elif not text.startswith(":"):
            field, separator, value = text.partition(":")
            if value.startswith(" "):
                value = value[1:]
            if field == "data":
                data.append(value if separator else "")

    for chunk in itertools.chain(chunks, [None]):
        text = decoder.decode(chunk if chunk is not None else b"", final=chunk is None)
        for character in text:
            if pending_cr:
                pending_cr = False
                if character == "\n":
                    continue
            if character in ("\r", "\n"):
                finish_line(line)
                line = ""
                pending_cr = character == "\r"
            else:
                line += character
    return events
