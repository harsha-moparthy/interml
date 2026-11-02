import numpy as np


def q384(next_logits, prompt, max_new_tokens, eos_id):
    """Generate greedily using a supplied next_logits(token_list) callback. Append at most
    max_new_tokens tokens and stop after appending EOS; return prompt plus completion.
    """
    tokens = list(prompt)
    for _ in range(max_new_tokens):
        token = int(np.argmax(next_logits(tokens)))
        tokens.append(token)
        if token == eos_id:
            break
    return tokens
