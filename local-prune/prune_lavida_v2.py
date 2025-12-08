# prune_lavida.py
# 완전 수동 Generalized Pruner (FFN + Attention Heads)
# - Torch-Pruning 미사용 (그래프 트레이스/메모리 이슈 회피)
# - CPU에서도 안전하게 동작
# - 저장 폴더명을 최종 파라미터 수로 자동 네이밍

import os
import re
import json
import math
import torch
import torch.nn as nn
from transformers import AutoTokenizer
from safetensors.torch import load_file, save_file

from config_v2 import PruneConfig
from utils_ratio import estimate_param_count, human

###############################################
# 1) GitHub LaViDa 모델 import
###############################################
import sys
sys.path.append("/root/autodl-tmp/LaViDa")

# from llava.model.language_model.llada.modeling_llada import LLaDAModelLM
# from llava.model.language_model.llada.configuration_llada import LLaDAConfig

from llava.model.language_model.llava_llada import LlavaLladaForMaskedDiffusion,LlavaLladaConfig

###############################################
# Dummy Vision Tower
###############################################
class DummyVisionTower(nn.Module):
    def __init__(self, embed_dim=1024):
        super().__init__()
        self.embed_dim = embed_dim
    def forward(self, *args, **kwargs):
        B, T = 1, 1
        return torch.zeros(B, T, self.embed_dim).to(next(self.parameters()).device)

###############################################
#  Load LaViDa weights (safetensors shards)
###############################################
def load_llada_weights(model, model_dir):
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    with open(index_path, "r") as f:
        index = json.load(f)
    weight_map = index["weight_map"]

    for shard_file in sorted(set(weight_map.values())):
        shard_path = os.path.join(model_dir, shard_file)
        print(f"[Load] shard: {shard_path}")
        shard_tensor = load_file(shard_path)
        # strict=False : 일부 모듈 누락/추가 허용
        model.load_state_dict(shard_tensor, strict=False)
    print("[Load] Completed all shards")

###############################################
# Utilities
###############################################
def to_cpu(model: nn.Module):
    model.to("cpu")
    torch.cuda.empty_cache()

def disable_activation_checkpointing(model: LlavaLladaForMaskedDiffusion):
    try:
        model.model.set_activation_checkpointing(None)
        print("[AC] activation checkpointing disabled")
    except Exception as e:
        print(f"[AC] skip disable: {e}")

def safe_slug_from_human(human_str: str) -> str:
    # "6.12B" -> "6p12B"
    s = human_str.replace(".", "p")
    s = re.sub(r"[^A-Za-z0-9_]+", "", s)
    return s

###############################################
# Extract and preserve non-pruned components
###############################################
def extract_preserved_components(model):
    """
    提取需要保留的组件（不参与剪枝但推理需要的部分）
    返回一个字典，包含这些组件的 state_dict
    """
    preserved = {}
    
    # 检查并提取 vision_tower
    # vision_tower可能在model.model中，或者通过get_vision_tower()方法获取
    vision_tower = None
    if hasattr(model, "get_vision_tower"):
        try:
            vision_tower = model.get_vision_tower()
        except:
            pass
    elif hasattr(model, "model") and hasattr(model.model, "get_vision_tower"):
        try:
            vision_tower = model.model.get_vision_tower()
        except:
            pass
    elif hasattr(model, "vision_tower"):
        vision_tower = model.vision_tower
    elif hasattr(model, "model") and hasattr(model.model, "vision_tower"):
        vision_tower = model.model.vision_tower
    
    if vision_tower is not None and not isinstance(vision_tower, DummyVisionTower):
        # save full path of vision_tower
        if hasattr(model, "model") and hasattr(model.model, "vision_tower") and model.model.vision_tower is vision_tower:
            preserved["model.vision_tower"] = vision_tower.state_dict()
            print("[Preserve] model.vision_tower extracted")
        else:
            preserved["vision_tower"] = vision_tower.state_dict()
            print("[Preserve] vision_tower extracted")
    
    # check and extract mm_projector
    if hasattr(model, "model") and hasattr(model.model, "mm_projector"):
        if model.model.mm_projector is not None:
            preserved["model.mm_projector"] = model.model.mm_projector.state_dict()
            print("[Preserve] model.mm_projector extracted")
    
    # check and extract image_newline (nn.Parameter, not Module)
    if hasattr(model, "model") and hasattr(model.model, "image_newline"):
        image_newline = model.model.image_newline
        if image_newline is not None:
            # image_newline is nn.Parameter, save parameter value (not state_dict)
            preserved["model.image_newline"] = {"image_newline": image_newline.data.clone()}
            print("[Preserve] model.image_newline extracted")
    elif hasattr(model, "image_newline"):
        image_newline = model.image_newline
        if image_newline is not None:
            # image_newline is nn.Parameter, save parameter value (not state_dict)
            preserved["image_newline"] = {"image_newline": image_newline.data.clone()}
            print("[Preserve] image_newline extracted")
    
    return preserved

def restore_preserved_components(model, preserved_dict):
    """
    将保留的组件恢复到模型中
    """
    for key, state_dict in preserved_dict.items():
        # special handling image_newline (it is nn.Parameter, not Module)
        if key.endswith("image_newline"):
            # find corresponding Parameter by key path
            parts = key.split(".")
            module = model
            found = True
            for part in parts[:-1]:  # except the last part (image_newline)
                if hasattr(module, part):
                    module = getattr(module, part)
                else:
                    print(f"[Warning] Cannot find module path: {key}")
                    found = False
                    break
            
            if found:
                if hasattr(module, "image_newline"):
                    # restore Parameter value
                    param_name = list(state_dict.keys())[0]  # usually "image_newline"
                    if param_name in state_dict:
                        # check dimension match
                        saved_param = state_dict[param_name]
                        if module.image_newline.shape == saved_param.shape:
                            module.image_newline.data.copy_(saved_param)
                            print(f"[Restore] {key} restored")
                        else:
                            print(f"[Warning] {key} shape mismatch: {module.image_newline.shape} vs {saved_param.shape}, skipping")
                    else:
                        print(f"[Warning] Parameter '{param_name}' not found in state_dict for {key}")
                else:
                    print(f"[Warning] image_newline not found in module: {key}")
            continue
            
        # normal module restoration (vision_tower, mm_projector, etc.)
        parts = key.split(".")
        module = model
        found = True
        for part in parts:
            if hasattr(module, part):
                module = getattr(module, part)
            else:
                print(f"[Warning] Cannot find module path: {key}")
                found = False
                break
        
        if found:
            module.load_state_dict(state_dict, strict=False)
            print(f"[Restore] {key} restored")

###############################################
# Importance Scores
###############################################
@torch.no_grad()
def ffn_channel_scores(ff_proj: nn.Linear, up_proj: nn.Linear | None, norm: int = 2):
    # importance score of each FFN channel (row): L2 norm sum
    if up_proj is not None:
        s = ff_proj.weight.detach().norm(dim=1, p=norm) + up_proj.weight.detach().norm(dim=1, p=norm)
    else:
        s = ff_proj.weight.detach().norm(dim=1, p=norm)
    return s  # [hidden]

@torch.no_grad()
def attn_head_scores_from_KVQO_proj(attn_out: nn.Linear, q_proj: nn.Linear, k_proj: nn.Linear, v_proj: nn.Linear, n_heads: int, d_model: int, p: int = 2):
    """
    By grouping KVQO-proj input by head group (size head_dim), use the L2 norm sum of each head input column as the score.
    """
    Q = q_proj.weight.detach()  # [d_model, intermediate_size] tensor shape is (intermediate_size, d_model)
    K = k_proj.weight.detach()  # [d_model, intermediate_size] tensor shape is (intermediate_size, d_model)
    V = v_proj.weight.detach()  # [d_model, intermediate_size] tensor shape is (intermediate_size, d_model)
    O = attn_out.weight.detach()  # [intermediate_size, d_model] tensor shape is (d_model, intermediate_size)
    head_dim = d_model // n_heads
    scores = []
    for h in range(n_heads):
        c0 = h * head_dim
        c1 = (h + 1) * head_dim
        # Frobenius norm of input column group
        scores.append(Q[c0:c1,:].norm(p=p)+K[c0:c1,:].norm(p=p)+V[c0:c1,:].norm(p=p)+O[:,c0:c1].norm(p=p))
    return torch.stack(scores)  # [num_heads]

###############################################
# FFN Manual Prune
###############################################
@torch.no_grad()
def prune_ffn_in_block(block, keep_ratio: float, norm: int = 2):
    """
    block: LLaDALlamaBlock or LLaDASequentialBlock
    - LLaDALlamaBlock: ff_proj + up_proj + ff_out
    - LLaDASequentialBlock: ff_proj(+), ff_out, (up_proj 없을 수 있음)
    """
    if not hasattr(block, "ff_proj") or not hasattr(block, "ff_out"):
        return 0, 0  # nothing

    ff_proj: nn.Linear = block.ff_proj
    ff_out: nn.Linear = block.ff_out
    up_proj: nn.Linear | None = getattr(block, "up_proj", None)

    H, D = ff_proj.weight.shape  # [hidden, d_model]
    if H <= 4:
        return 0, H

    keep = max(4, int(H * keep_ratio))
    if keep >= H:
        return 0, H

    # importance based channel selection
    scores = ffn_channel_scores(ff_proj, up_proj, norm=norm)
    prune_k = H - keep
    prune_idx = torch.argsort(scores)[:prune_k]
    keep_mask = torch.ones(H, dtype=torch.bool)
    keep_mask[prune_idx] = False

    # actually slicing
    new_hidden_size = int(keep_mask.sum().item())
    ff_proj.weight = nn.Parameter(ff_proj.weight[keep_mask, :].contiguous())
    ff_proj.out_features = new_hidden_size  # update out_features
    if ff_proj.bias is not None:
        ff_proj.bias = nn.Parameter(ff_proj.bias[keep_mask].contiguous())

    if up_proj is not None:
        up_proj.weight = nn.Parameter(up_proj.weight[keep_mask, :].contiguous())
        up_proj.out_features = new_hidden_size  # update out_features
        if up_proj.bias is not None:
            up_proj.bias = nn.Parameter(up_proj.bias[keep_mask].contiguous())

    ff_out.weight = nn.Parameter(ff_out.weight[:, keep_mask].contiguous())
    ff_out.in_features = new_hidden_size  # update in_features
    # ff_out.bias has the same out dim

    # update hidden_size meta (if exists)
    if hasattr(block, "hidden_size"):
        try:
            block.hidden_size = int(keep_mask.sum().item())
        except Exception:
            pass

    return prune_k, H

@torch.no_grad()
def manual_prune_ffn(model: LlavaLladaForMaskedDiffusion, keep_ratio: float, norm: int = 2):
    tr = model.model.transformer
    if not hasattr(tr, "blocks"):
        raise NotImplementedError("Block groups path is not implemented.")
    total_pruned, total_before = 0, 0
    for i, block in enumerate(tr.blocks):
        pruned, before = prune_ffn_in_block(block, keep_ratio, norm=norm)
        total_pruned += pruned
        total_before += before
        if pruned > 0:
            print(f"[FFN][Layer {i:02d}] {before} -> {before - pruned}  (-{pruned})")
        else:
            print(f"[FFN][Layer {i:02d}] no change ({before})")
    print(f"\n[FFN Pruning] total pruned channels: {total_pruned} / {total_before}")
    
    # 更新config中的mlp_hidden_size
    if len(tr.blocks) > 0 and hasattr(tr.blocks[0], "ff_proj"):
        new_hidden_size = tr.blocks[0].ff_proj.weight.shape[0]
        model.config.mlp_hidden_size = new_hidden_size

###############################################
# Attention Head Prune
###############################################
def _prune_cols_linear(linear: nn.Linear, col_mask: torch.Tensor):
    linear.weight = nn.Parameter(linear.weight[:, col_mask].contiguous())
    # update in_features attribute
    linear.in_features = int(col_mask.sum().item())

def _prune_rows_linear(linear: nn.Linear, row_mask: torch.Tensor):
    linear.weight = nn.Parameter(linear.weight[row_mask, :].contiguous())
    # update out_features attribute
    linear.out_features = int(row_mask.sum().item())
    if linear.bias is not None:
        linear.bias = nn.Parameter(linear.bias[row_mask].contiguous())

@torch.no_grad()
def manual_prune_attention_heads(model: LlavaLladaForMaskedDiffusion, keep_ratio_heads: float, norm: int = 2):
    """
    LaViDa is not GQA, so q/k/v head numbers are the same structure.
    → only n_heads can be pruned.
    """
    tr = model.model.transformer
    if not hasattr(tr, "blocks"):
        raise NotImplementedError("Block groups path is not implemented.")

    cfg = model.config
    d_model = cfg.d_model
    n_heads_before = cfg.n_heads

    assert d_model % n_heads_before == 0, "d_model must be divisible by n_heads."
    
    # LaViDa is not GQA, so q/k/v head numbers should be the same
    # we need to prune k/v heads to keep consistency

    # calculate importance score based on attn_out of the first block
    sample_block = tr.blocks[0]
    attn_out: nn.Linear = sample_block.attn_out# [intermediate_size, d_model] # tensor shape is (d_model, intermediate_size)
    q_proj: nn.Linear = sample_block.q_proj # [d_model, intermediate_size] # tensor shape is (intermediate_size, d_model)
    k_proj: nn.Linear = sample_block.k_proj # [d_model, intermediate_size] # tensor shape is (intermediate_size, d_model)
    v_proj: nn.Linear = sample_block.v_proj # [d_model, intermediate_size] # tensor shape is (intermediate_size, d_model)
    scores = attn_head_scores_from_KVQO_proj(attn_out, q_proj, k_proj, v_proj, n_heads_before, d_model, p=norm)

    # calculate keep heads
    keep_heads = max(1, int(round(n_heads_before * keep_ratio_heads)))
    keep_heads = min(keep_heads, n_heads_before)
    
    # important: ensure d_model can be divided by keep_heads (because d_model is kept unchanged)
    # adjust keep_heads down to the nearest value that can divide d_model
    original_keep_heads = keep_heads
    while keep_heads > 0 and d_model % keep_heads != 0:
        keep_heads -= 1
    if keep_heads == 0:
        keep_heads = 1  # at least keep 1 head
    if keep_heads != original_keep_heads:
        print(f"[ATTN] Adjusted keep_heads: {original_keep_heads} -> {keep_heads} (to ensure d_model={d_model} % n_heads == 0)")

    if keep_heads == n_heads_before:
        print(f"[ATTN] no head pruning (keep={keep_heads}/{n_heads_before})")
        return

    # importance based keep head selection
    keep_idx = torch.argsort(scores, descending=True)[:keep_heads]
    keep_idx, _ = torch.sort(keep_idx)

    print(f"[ATTN] Keep heads: {keep_heads} / {n_heads_before}")
    print(f"[ATTN] head indices kept: {keep_idx.tolist()}")

    # create head unit row/column mask
    head_dim = d_model // n_heads_before
    q_row_mask = torch.zeros(n_heads_before * head_dim, dtype=torch.bool)
    o_col_mask = torch.zeros_like(q_row_mask)
    for h in keep_idx.tolist():
        q_row_mask[h * head_dim : (h + 1) * head_dim] = True
        o_col_mask[h * head_dim : (h + 1) * head_dim] = True

    # apply to all blocks equally
    for i, block in enumerate(tr.blocks):

        if hasattr(block, "q_proj") and hasattr(block, "k_proj") and hasattr(block, "v_proj"):
            # LLaDALlamaBlock shape
            q_proj: nn.Linear = block.q_proj
            k_proj: nn.Linear = block.k_proj
            v_proj: nn.Linear = block.v_proj
            attn_out: nn.Linear = block.attn_out
            # q_proj out_rows prune
            _prune_rows_linear(q_proj, q_row_mask)
            # k_proj out_rows prune
            _prune_rows_linear(k_proj, q_row_mask)
            # v_proj out_rows prune
            _prune_rows_linear(v_proj, q_row_mask)

            # attn_out in_cols prune
            _prune_cols_linear(attn_out, o_col_mask)

        else:
            raise NotImplementedError("Unknown block type for attention head pruning.")

        print(f"[ATTN][Layer {i:02d}] heads {n_heads_before} -> {keep_heads}")

    # update config
    cfg.n_heads = keep_heads
    cfg.n_kv_heads = keep_heads
    cfg.head_dim = head_dim # Keep the old head_dim
    cfg.prune_model = True
    # d_model is kept unchanged, only n_heads is updated


###############################################
# MAIN
###############################################
def main():
    cfg = PruneConfig()

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)

    # Load config & model
    print("[Load] GitHub LaViDa LlavaLladaForMaskedDiffusion...")
    with open(os.path.join(cfg.model_name, "config.json"), "r") as f:
        config_dict = json.load(f)
    config = LlavaLladaConfig.from_dict(config_dict)
    config.head_dim = config.d_model//config.n_heads
    config.head_dim = config.d_model//config.n_heads
    # create model on CPU → safe
    model = LlavaLladaForMaskedDiffusion(config)
    to_cpu(model)
    model.eval()

    # load weights
    load_llada_weights(model, cfg.model_name)

    # activation checkpointing off
    disable_activation_checkpointing(model)

    # extract components to preserve (before pruning)
    print("\n[Extract] Preserving non-pruned components...")
    preserved_components = extract_preserved_components(model)

    base_params = estimate_param_count(model)
    base_h = human(base_params)
    print(f"[Params] Base: {base_h}")
    
    # clear GPU cache (if any)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("[Memory] GPU cache cleared")

    # === Target pruning ===
    targets_list = list(cfg.targets.items())
    for idx, (tag, _targetN) in enumerate(targets_list):
        print(f"\n=== Target {tag} (ignore targetN; use manual ratios) ===")

        # memory optimization strategy:
        # 1. if the last target, can directly prune on the original model, no need to copy
        # 2. otherwise use state_dict copy (more memory efficient than reloading file)
        is_last_target = (idx == len(targets_list) - 1)
        
        if is_last_target and len(targets_list) == 1:
            # only one target, directly prune on the original model
            print("[Memory] Using original model directly (single target)")
            model_p = model
        else:
            # multiple targets, need to copy model
            print("[Memory] Creating model copy using state_dict...")
            model_p = LlavaLladaForMaskedDiffusion(LlavaLladaConfig.from_dict(config_dict))
            to_cpu(model_p)
            model_p.eval()
            
            print("[Memory] Copying weights from original model...")
            model_p.load_state_dict(model.state_dict(), strict=False)
            
            # clear GPU cache (if any)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # note: here we don't replace vision_tower, because we will restore original weights later
        disable_activation_checkpointing(model_p)

        # 1) Attention heads (Q-head only)
        attn_keep_ratio = getattr(cfg, "attn_head_keep_ratio", 1.0)  # ex) 0.75
        attn_keep_ratio = float(attn_keep_ratio)
        norm = getattr(cfg, "norm", 2)  # norm parameter for importance scoring
        if attn_keep_ratio < 1.0:
            manual_prune_attention_heads(model_p, attn_keep_ratio, norm=norm)
        else:
            print("[ATTN] skip (keep_ratio=1.0)")

        # 2) FFN channels
        ffn_keep_ratio = float(getattr(cfg, "ffn_manual_ratio", 1.0))  # ex) 0.30
        print(f"[Plan] attn_head_keep={attn_keep_ratio:.3f}, ffn_keep={ffn_keep_ratio:.3f}")
        if ffn_keep_ratio < 1.0:
            manual_prune_ffn(model_p, ffn_keep_ratio, norm=norm)
        else:
            print("[FFN] skip (keep_ratio=1.0)")

        # restore preserved components to pruned model
        if preserved_components:
            print("\n[Restore] Restoring preserved components...")
            restore_preserved_components(model_p, preserved_components)

        final_params = estimate_param_count(model_p)
        final_h = human(final_params)
        print(f"[Params] Final ≈ {final_h}")

        # save folder name: automatically named by actual parameter number
        slug = safe_slug_from_human(final_h)  # "6.12B" -> "6p12B"
        save_dir = os.path.join(cfg.out_dir, f"lavida_llada_pruned_{slug}-norm{cfg.norm}-f_r{cfg.ffn_manual_ratio}-a_r{cfg.attn_head_keep_ratio}")
        os.makedirs(save_dir, exist_ok=True)

        # save to HuggingFace format
        print(f"\n[Save] Saving to HuggingFace format: {save_dir}")
        
        # 1. save config.json (pruned config, including updated n_heads, etc.)
        model_p.config.save_pretrained(save_dir)
        
        # 2. save tokenizer
        tokenizer.save_pretrained(save_dir)
        
        # 3. save model weights to safetensors format (HuggingFace standard)
        state_dict = model_p.state_dict()
        
        # calculate size of each shard (default max 5GB per shard)
        max_shard_size = 5 * 1024 * 1024 * 1024  # 5GB in bytes
        
        # step 1: calculate how many shards are needed
        shard_list = []
        current_shard = {}
        current_size = 0
        
        for key, tensor in state_dict.items():
            tensor_size = tensor.numel() * tensor.element_size()
            
            if current_size + tensor_size > max_shard_size and current_shard:
                # save current shard to list
                shard_list.append(current_shard)
                current_shard = {}
                current_size = 0
            
            current_shard[key] = tensor
            current_size += tensor_size
        
        # add last shard
        if current_shard:
            shard_list.append(current_shard)
        
        total_shards = len(shard_list)
        
        # step 2: save each shard and create weight_map
        weight_map = {}
        total_size = 0
        
        for idx, shard_dict in enumerate(shard_list):
            if total_shards == 1:
                shard_name = "model.safetensors"
            else:
                shard_name = f"model-{idx+1:05d}-of-{total_shards:05d}.safetensors"
            
            shard_path = os.path.join(save_dir, shard_name)
            save_file(shard_dict, shard_path)
            
            # update weight_map
            for key in shard_dict.keys():
                weight_map[key] = shard_name
            
            # calculate total size
            for tensor in shard_dict.values():
                total_size += tensor.numel() * tensor.element_size()
            
            print(f"[Save] Saved shard {idx+1}/{total_shards}: {shard_name}")
        
        # 4. create model.safetensors.index.json (if there are multiple shards)
        if total_shards > 1:
            index = {
                "metadata": {"total_size": total_size},
                "weight_map": weight_map
            }
            index_path = os.path.join(save_dir, "model.safetensors.index.json")
            with open(index_path, "w") as f:
                json.dump(index, f, indent=2)
            print(f"[Save] Created index file: model.safetensors.index.json")
        
        print(f"[Saved] Model saved in HuggingFace format: {save_dir}")
        print(f"[Saved] Total shards: {total_shards}")
        
        # clear memory: if not the last target, delete pruned model to free memory
        # note: is_last_target is defined in the loop, so we need to recalculate here
        is_last = (idx == len(targets_list) - 1)
        if not is_last:
            del model_p
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            import gc
            gc.collect()
            print("[Memory] Pruned model deleted, memory freed")

if __name__ == "__main__":
    main()
