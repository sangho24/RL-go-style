import re
import unicodedata
from typing import Optional

# 신진서 이름 패턴 (띄어쓰기/하이픈/랭크 등 섞여도 잡기 위함)
_SHIN_PATTERNS = [
    "신진서",
    "Shin Jinseo",
    "Shin Jin-seo",
    "Jinseo Shin",
]

def normalize_name(name: Optional[str]) -> str:
    """
    SGF 속 이름 문자열을 비교용으로 정규화:
    - NFKC 정규화
    - 소문자
    - 공백 제거
    - 영문/한글 외 문자 제거(숫자, 기호, 괄호, 랭크 등)
    """
    if not name:
        return ""
    s = unicodedata.normalize("NFKC", name)
    s = s.lower()
    s = re.sub(r"\s+", "", s)            # 공백 제거
    s = re.sub(r"[^a-z가-힣]", "", s)    # 영문/한글만 남기기
    return s

def is_shin_jinseo(name: Optional[str]) -> bool:
    """주어진 이름 문자열이 신진서 사범을 가리키는지 대략 판별."""
    norm = normalize_name(name)
    if not norm:
        return False
    for pat in _SHIN_PATTERNS:
        pat_norm = normalize_name(pat)
        if pat_norm and pat_norm in norm:
            return True
    return False
