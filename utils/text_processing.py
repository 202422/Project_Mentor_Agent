import re
from typing import List


def simple_tokenize(text: str) -> List[str]:
    return re.findall(r"\w+", text.lower())
