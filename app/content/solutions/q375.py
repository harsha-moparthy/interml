import re
import string


def q375(prediction, references):
    """Return the maximum exact-match score after lowercasing, removing ASCII punctuation
    and English articles, and collapsing whitespace.
    """

    def normalize(s):
        s = s.lower().translate(str.maketrans("", "", string.punctuation))
        return " ".join(re.sub("\\b(a|an|the)\\b", " ", s).split())

    return int(any((normalize(prediction) == normalize(r) for r in references)))
