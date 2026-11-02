import re
from fractions import Fraction


def q381(prediction, reference):
    """Verify exact equivalence of rational answers written as integers, decimals, or a/b,
    allowing surrounding whitespace and a trailing percent sign. Reject expressions
    outside this numeric grammar.
    """

    def parse(s):
        s = s.strip()
        percent = s.endswith("%")
        s = s[:-1].strip() if percent else s
        if not re.fullmatch("[+-]?(?:\\d+(?:\\.\\d*)?|\\.\\d+)(?:/[+-]?\\d+)?", s):
            raise ValueError("unsupported answer")
        parts = s.split("/")
        value = Fraction(parts[0])
        if len(parts) == 2:
            value /= Fraction(parts[1])
        return value / 100 if percent else value

    try:
        return parse(prediction) == parse(reference)
    except (ValueError, ZeroDivisionError):
        return False
