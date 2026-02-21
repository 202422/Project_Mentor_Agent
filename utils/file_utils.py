import os
from typing import List


def list_files(root: str, exts: List[str] | None = None) -> List[str]:
    results = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if exts and not any(fn.endswith(e) for e in exts):
                continue
            results.append(os.path.join(dirpath, fn))
    return results
