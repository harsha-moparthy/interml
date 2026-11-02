def q350(messages, encode, role_ids, end_id, assistant_prefix=True):
    """Encode a simple explicit chat format: role token, encoded content, end token per
    message, optionally followed by an assistant role token. This is a workbook format,
    not a vendor-specific tokenizer template.
    """
    result = []
    for message in messages:
        result.append(role_ids[message["role"]])
        result.extend(encode(message["content"]))
        result.append(end_id)
    if assistant_prefix:
        result.append(role_ids["assistant"])
    return result
