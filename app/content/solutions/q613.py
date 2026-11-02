import torch


def beam_search(next_logits, prompt, eos_id, *, width=4, max_new_tokens=32, alpha=0.6):
    """Reference beam search; next_logits(tokens) returns one vocabulary logit vector.

    Return (tokens, normalized_score). Length excludes the prompt and includes EOS.
    Finished beams are carried unchanged. Stop when every retained beam has ended,
    or the token budget expires; this avoids an unsound early-stop bound.
    Work: O(T*width*V*log(V)) plus model calls; history storage O(width*T).
    """
    if width < 1 or max_new_tokens < 1 or alpha < 0:
        raise ValueError(
            "positive beam width/token budget and nonnegative alpha required"
        )
    prompt = list(prompt)
    beams = [(prompt, 0.0, False)]

    def score(beam):
        tokens, log_probability, _ = beam
        generated = max(1, len(tokens) - len(prompt))
        return log_probability / (((5 + generated) / 6) ** alpha)

    for _ in range(max_new_tokens):
        candidates = []
        for tokens, log_probability, finished in beams:
            if finished:
                candidates.append((tokens, log_probability, True))
                continue
            log_probs = torch.as_tensor(next_logits(tokens)).double().log_softmax(-1)
            if log_probs.ndim != 1 or not 0 <= eos_id < len(log_probs):
                raise ValueError("invalid logits or EOS ID")
            values, ids = log_probs.topk(min(width, len(log_probs)))
            for value, token in zip(values.tolist(), ids.tolist()):
                candidates.append(
                    (tokens + [token], log_probability + value, token == eos_id)
                )
        beams = sorted(candidates, key=score, reverse=True)[:width]
        if all(finished for _, _, finished in beams):
            break
    best = max(beams, key=score)
    return best[0], score(best)
