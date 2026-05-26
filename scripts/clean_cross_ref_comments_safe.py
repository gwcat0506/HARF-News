#!/usr/bin/env python3
"""Remove cross-file reference text from comments and docstrings only."""
from __future__ import annotations

import re
import tokenize
from io import BytesIO
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "fakenewsdetection"
SKIP = ("multimodal_deepfake/models/xbert.py", "FND-CLIP-Fake-News-Detection/data/util.py")

CROSS_REF = re.compile(
    r"("
    r"동일|그대로 사용|그대로 유지|그대로 평가|원본 VER3|원본과 동일|원본 동일|원본 HARF|"
    r"원본 run_|원본 학습|원본 MiRAGe|Baseline 원본|follows |based on baseline|"
    r"hammer_miragenews|harfnet\.py|harfnet split|fnd_clip 방식|ResNet baseline|"
    r"wo_bi_oam 와|hier3\.py|VER3와|VER3 \(원본|공정한 비교|동일 전처리|"
    r"동일한 |동일 형식|동일하게|완전 동일|과 동일|와 동일|"
    r"harfnet_DGM4|harfnet 과|정렬한|\[학습 설정"
    r")",
    re.I,
)


def comment_tail_has_cross_ref(line: str) -> bool:
    if "#" not in line:
        return False
    return bool(CROSS_REF.search(line[line.index("#") :]))


def strip_hash_comment(line: str) -> str:
    idx = line.index("#")
    return line[:idx].rstrip() + ("\n" if line.endswith("\n") else "")


def clean_docstring_text(text: str) -> str:
    lines = text.splitlines(keepends=True)
    out: list[str] = []
    changed = False
    for ln in lines:
        if CROSS_REF.search(ln):
            changed = True
            continue
        out.append(ln)
    return "".join(out) if changed else text


def process_tokens(source: str, path: Path) -> str:
    out: list[tokenize.TokenInfo] = []
    try:
        tokens = list(tokenize.tokenize(BytesIO(source.encode()).readline))
    except tokenize.TokenError:
        return source

    for tok in tokens:
        if tok.type == tokenize.COMMENT and CROSS_REF.search(tok.string):
            continue
        if tok.type == tokenize.STRING and tok.string.startswith(('"""', "'''")):
            inner = tok.string[3:-3]
            if "\n" in inner or len(inner) > 80:
                cleaned = clean_docstring_text(inner)
                if cleaned != inner:
                    quote = tok.string[:3]
                    new_s = quote + cleaned + quote
                    out.append(tok._replace(string=new_s))
                    continue
        out.append(tok)

    return tokenize.untokenize(out).decode()


def process_file(path: Path) -> bool:
    source = path.read_text(encoding="utf-8")
    new_lines: list[str] = []
    changed = False
    for line in source.splitlines(keepends=True):
        if line.lstrip().startswith("#") and CROSS_REF.search(line):
            changed = True
            continue
        if comment_tail_has_cross_ref(line):
            new_lines.append(strip_hash_comment(line))
            changed = True
        else:
            new_lines.append(line)

    interim = "".join(new_lines)
    try:
        final = process_tokens(interim, path)
    except Exception:
        final = interim

    if final != source:
        path.write_text(final, encoding="utf-8")
        return True
    return changed


def main() -> None:
    n = 0
    for path in sorted(ROOT.rglob("*.py")):
        if any(s in path.as_posix() for s in SKIP):
            continue
        if process_file(path):
            n += 1
            print(path.relative_to(ROOT.parent))
    print(f"updated {n} files")


if __name__ == "__main__":
    main()
