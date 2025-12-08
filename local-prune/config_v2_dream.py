# config.py
from dataclasses import dataclass
@dataclass
class PruneConfig:
    # ---- 모델 로드 ----
    model_code_path = "/root/LaViDa/lavida-ckpts/lavida-dream-v1.0-instruct"
    model_name: str = "/root/LaViDa/lavida-ckpts/lavida-dream-v1.0-instruct"
    dtype: str = "float16"
    device_map: str = "cuda"

    # ---- 프루닝 타깃 파라미터 수(대략치) ----
    ratio = 0.5
    ffn_manual_ratio = 0.2
    attn_head_keep_ratio = 0.9
    norm: int = 2
    # ---- I/O ----
    out_dir: str = f"/root/LaViDa/lavida-ckpts" # 결과 저장 루트 폴더
    make_hf_repo: bool = False       # HF 업로드할지 여부
    hf_repo_prefix: str = "yourname" # 업로드 시 계정/조직명



    # targets는 태그 이름만 쓰고, 숫자는 무시(수동비율 사용)
    targets = {
        "manual": None
    }

    prune_vision_tower: bool = False # SigLIP는 기본 미프루닝
    seed: int = 42                   # 재현성

    # ---- 시각화/리포트 ----
    dump_graph_json: bool = True
    list_layers_txt: bool = True

    def __post_init__(self):
        if self.targets is None:
            self.targets = {
                "0.5B": 500_000_000,
                "1B":   1_000_000_000,
                "2B":   2_000_000_000,
            }
