# utils_ratio.py
import math
from typing import Dict, Tuple

def estimate_param_count(model) -> int:
    return sum(p.numel() for p in model.parameters())

def suggest_ratios_for_target(
    base_params: int, target_params: int,
    base_attn_ratio: float, base_ffn_ratio: float
) -> Tuple[float, float]:
    """
    아주 러프하지만 잘 작동하는 휴리스틱:
    - FFN이 파라미터 대부분 → FFN을 더 과감히 줄이고,
    - 부족하면 어텐션 비율을 약간 올린다.
    """
    if target_params >= base_params:
        return 0.0, 0.0

    shrink = target_params / base_params  # 남겨야 하는 비중(0~1)

    # 대략: 파라미터의 60~70%가 FFN, 20~30%가 attn이라 가정
    # shrink에 맞게 ff/attn 비율 역으로 조정
    ffn = 1 - shrink
    attn = 0.6 * (1 - shrink)

    # 초기값과 섞어서 너무 과도한 값 방지
    ffn = min(max((ffn + base_ffn_ratio) / 2, 0.0), 0.9)
    attn = min(max((attn + base_attn_ratio) / 2, 0.0), 0.9)

    return attn, ffn

def human(n: int) -> str:
    units = ["", "K", "M", "B"]
    idx = 0
    x = float(n)
    while x >= 1000 and idx < len(units)-1:
        x /= 1000.0
        idx += 1
    return f"{x:.2f}{units[idx]}"
