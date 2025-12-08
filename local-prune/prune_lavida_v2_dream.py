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

from config_v2_dream import PruneConfig
from utils_ratio import estimate_param_count, human

###############################################
# 1) GitHub LaViDa model import
###############################################
import sys
sys.path.append("/root/autodl-tmp/LaViDa")

# from llava.model.language_model.llada.modeling_llada import LLaDAModelLM
# from llava.model.language_model.llada.configuration_llada import LLaDAConfig

# from llava.model.language_model.dream.modeling_dream import DreamConfig,DreamModel
from llava.model.language_model.llava_dream import LlavaDreamForMaskedDiffusion,LlavaDreamModel,LlavaDreamConfig
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
#  Load Dream weights (safetensors shards)
###############################################
def load_Dream_weights(model, model_dir):
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    with open(index_path, "r") as f:
        index = json.load(f)
    weight_map = index["weight_map"]
    
    # get target dtype of the model (usually bfloat16, can save half memory)
    target_dtype = getattr(model.config, "torch_dtype", None)
    if target_dtype is None or isinstance(target_dtype, str):
        if target_dtype == "bfloat16":
            target_dtype = torch.bfloat16
        elif target_dtype == "float16":
            target_dtype = torch.float16
        else:
            target_dtype = torch.bfloat16  # default use bfloat16
    print(f"[Memory] Loading weights with dtype: {target_dtype} (saves ~50% memory vs float32)")

    for shard_file in sorted(set(weight_map.values())):
        shard_path = os.path.join(model_dir, shard_file)
        print(f"[Load] shard: {shard_path}")
        # use torch.no_grad() to avoid creating computation graph
        with torch.no_grad():
            shard_tensor = load_file(shard_path)
            # immediately convert to target dtype (usually bfloat16), save half memory
            # if the file is float32, convert to bfloat16 to save 50% memory
            shard_tensor = {k: v.to(target_dtype) if v.dtype.is_floating_point else v 
                           for k, v in shard_tensor.items()}
            # strict=False : allow some modules to be missing/added
            # use assign=True to avoid creating temporary copies (PyTorch 2.0+), if not available, fallback
            try:
                model.load_state_dict(shard_tensor, strict=False, assign=True)
            except TypeError:
                # PyTorch < 2.0 does not support assign parameter
                model.load_state_dict(shard_tensor, strict=False)
        # immediately release shard_tensor to save memory
        del shard_tensor
        import gc
        gc.collect()
    print("[Load] Completed all shards")

###############################################
# Utilities
###############################################
def to_cpu(model: nn.Module):
    model.to("cpu")
    torch.cuda.empty_cache()

def disable_activation_checkpointing(model: LlavaDreamForMaskedDiffusion):
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
    
    # check and extract vision_tower
    # vision_tower may be in model.model, or get by get_vision_tower() method
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
    
    # 检查并提取 mm_projector
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

    # check and extract lm_head.weight
    if hasattr(model, "lm_head"):
        lm_head = model.lm_head
        if lm_head is not None:
            preserved["model.lm_head"] = lm_head.state_dict()
            print("[Preserve] model.lm_head extracted")

    # check and extract embed_tokens.weight
    if hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
        embed_tokens = model.model.embed_tokens
        if embed_tokens is not None:
            preserved["model.embed_tokens"] = embed_tokens.state_dict()
            print("[Preserve] model.embed_tokens extracted")

    # check and extract model.norm.weight
    if hasattr(model, "model") and hasattr(model.model, "norm"):
        norm = model.model.norm
        if norm is not None:
            preserved["model.norm"] = norm.state_dict()
            print("[Preserve] model.norm extracted")

    return preserved

def restore_preserved_components(model, preserved_dict):
    """
    restore preserved components to model
    """
    for key, state_dict in preserved_dict.items():
        # special handling image_newline (it is nn.Parameter, not Module)
        if key.endswith("image_newline"):
            # find the corresponding Parameter by the key path
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
                    # restore the value of the Parameter
                    param_name = list(state_dict.keys())[0]  # usually "image_newline"
                    if param_name in state_dict:
                        # check if the dimension matches
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
            
        # restore normal modules (vision_tower, mm_projector, etc.)
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
def ffn_channel_scores(gate_proj: nn.Linear, up_proj: nn.Linear | None, down_proj: nn.Linear | None, p: int = 2):
    # importance score of each FFN channel (row): L2 norm sum
    s = gate_proj.weight.detach().norm(dim=1,p=p) + up_proj.weight.detach().norm(dim=1,p=p) + down_proj.weight.detach().norm(dim=0,p=p)
    # SwishGLU activation function score
    return s  # [intermediate size]

@torch.no_grad()
def attn_head_scores_from_KVQO(layer: nn.Module, num_key_value_heads: int, num_attention_heads: int, hidden_size: int, p: int = 2):
    """
    Prune corresponding Key and Value heads together with the Query heads according to the KV head importance score.
    """
    v_proj = layer.self_attn.v_proj.weight.detach()  # [hidden_size, num_key_value_heads * head_dim] tensor shape is (num_key_value_heads * head_dim, hidden_size)
    k_proj = layer.self_attn.k_proj.weight.detach()  # [hidden_size, num_key_value_heads * head_dim] tensor shape is (num_key_value_heads * head_dim, hidden_size)
    q_proj = layer.self_attn.q_proj.weight.detach()  # [hidden_size, num_attention_heads * head_dim] tensor shape is (num_attention_heads * head_dim, hidden_size)
    o_proj = layer.self_attn.o_proj.weight.detach()  # [num_attention_heads * head_dim, hidden_size] tensor shape is (hidden_size, num_attention_heads * head_dim)
    head_dim = hidden_size // num_attention_heads
    Group_step = num_attention_heads//num_key_value_heads
    scores = []
    for h in range(num_key_value_heads):
        c0 = h * head_dim
        c1 = (h + 1) * head_dim
        c2 = h * Group_step * head_dim
        c3 = (h + 1) * Group_step * head_dim
        # Frobenius norm of input column group
        scores.append(v_proj[c0:c1,:].norm(p=p)+k_proj[c0:c1,:].norm(p=p)+q_proj[c2:c3,:].norm(p=p)+o_proj[:,c2:c3].norm(p=p))
    return torch.stack(scores)  # [num_key_value_heads]

###############################################
# FFN Manual Prune
###############################################
@torch.no_grad()
def prune_ffn_in_layer(layer: nn.Module, keep_ratio: float, Pcfg: PruneConfig):
    """
    layer: DreamDecoderLayer
    
    def forward(self, hidden_state):
        return self.down_proj(self.act_fn(self.gate_proj(hidden_state)) * self.up_proj(hidden_state))
    """
    if not hasattr(layer.mlp, "gate_proj") or not hasattr(layer.mlp, "up_proj") or not hasattr(layer.mlp, "down_proj"):
        return 0, 0  # nothing (pruned, before)

    gate_proj: nn.Linear = layer.mlp.gate_proj
    up_proj: nn.Linear = layer.mlp.up_proj
    down_proj: nn.Linear = layer.mlp.down_proj

    H, I = gate_proj.weight.shape  # [hidden_size, intermediate_size] which is the same as the up_proj.weight.shape
    if I <= 4:
        return 0, I

    keep = max(4, int(I * keep_ratio))
    if keep >= I:
        return 0, I

    # importance based channel selection
    scores = ffn_channel_scores(gate_proj, up_proj, down_proj, Pcfg.norm)
    prune_k = I - keep
    prune_idx = torch.argsort(scores)[:prune_k]
    keep_mask = torch.ones(I, dtype=torch.bool)
    keep_mask[prune_idx] = False

    # actually slicing
    new_intermediate_size = int(keep_mask.sum().item())
    gate_proj.weight = nn.Parameter(gate_proj.weight[keep_mask,:].contiguous())
    up_proj.weight = nn.Parameter(up_proj.weight[keep_mask,:].contiguous())
    down_proj.weight = nn.Parameter(down_proj.weight[:,keep_mask].contiguous())

    gate_proj.out_features = new_intermediate_size  # update out_features
    if gate_proj.bias is not None:
        gate_proj.bias = nn.Parameter(gate_proj.bias[keep_mask].contiguous()) # keep bias

    up_proj.out_features = new_intermediate_size  # update out_features
    if up_proj.bias is not None:
        up_proj.bias = nn.Parameter(up_proj.bias[keep_mask].contiguous()) # keep bias

    down_proj.in_features = new_intermediate_size  # update in_features
    if down_proj.bias is not None:
        down_proj.bias = nn.Parameter(down_proj.bias.contiguous()) # keep bias

    return prune_k, I  # (pruned, before)

@torch.no_grad()
def manual_prune_ffn(model: LlavaDreamForMaskedDiffusion, keep_ratio: float, Pcfg: PruneConfig):
    layers = model.model.layers  # nn.ModuleList of DreamDecoderLayer
    if not isinstance(layers, nn.ModuleList) and not isinstance(layers, list):
        raise NotImplementedError("layers is not a ModuleList or list.")
    total_pruned, total_before = 0, 0
    for i, layer in enumerate(layers):
        pruned, before = prune_ffn_in_layer(layer, keep_ratio, Pcfg=Pcfg)
        total_pruned += pruned
        total_before += before
        if pruned > 0:
            print(f"[FFN][Layer {i:02d}] {before} -> {before - pruned}  (-{pruned})")
        else:
            print(f"[FFN][Layer {i:02d}] no change ({before})")
    print(f"\n[FFN Pruning] total pruned channels: {total_pruned} / {total_before}")
    
    # update config.intermediate_size
    if len(layers) > 0 and hasattr(layers[0], "mlp") and hasattr(layers[0].mlp, "gate_proj"):
        new_intermediate_size = layers[0].mlp.gate_proj.out_features
        model.config.intermediate_size = new_intermediate_size
        
@torch.no_grad()
def prune_attention_heads_in_layer(layer: nn.Module, keep_heads_idx: torch.Tensor, num_key_value_heads: int, num_attention_heads: int, hidden_size: int):
    """
    By grouping Q,K,V,O by KV, calculate the importance score of each group, and then prune according to the score.
    """
    Group_step = num_attention_heads//num_key_value_heads
    head_dim = hidden_size // num_attention_heads
    keep_heads_idx = keep_heads_idx.to(torch.long)
    # head group mask (used for column/row slicing)
    layer.self_attn.q_proj.weight = nn.Parameter(layer.self_attn.q_proj.weight[keep_heads_idx*head_dim*Group_step,:].contiguous())
    layer.self_attn.k_proj.weight = nn.Parameter(layer.self_attn.k_proj.weight[keep_heads_idx*head_dim,:].contiguous())
    layer.self_attn.v_proj.weight = nn.Parameter(layer.self_attn.v_proj.weight[keep_heads_idx*head_dim,:].contiguous())
    layer.self_attn.o_proj.weight = nn.Parameter(layer.self_attn.o_proj.weight[:,keep_heads_idx*head_dim*Group_step].contiguous())
    layer.self_attn.q_proj.out_features = keep_heads_idx.shape[0]*head_dim*Group_step
    layer.self_attn.k_proj.out_features = keep_heads_idx.shape[0]*head_dim
    layer.self_attn.v_proj.out_features = keep_heads_idx.shape[0]*head_dim
    layer.self_attn.o_proj.in_features = keep_heads_idx.shape[0]*head_dim*Group_step


@torch.no_grad()
def manual_prune_attention_heads(model: LlavaDreamForMaskedDiffusion, keep_ratio_heads: float, Pcfg: PruneConfig):
    layers = model.model.layers  # nn.ModuleList of DreamDecoderLayer
    if not isinstance(layers, nn.ModuleList) and not isinstance(layers, list):
        raise NotImplementedError("layers is not a ModuleList or list.")

    cfg = model.config
    hidden_size = cfg.hidden_size
    n_heads_before = cfg.num_attention_heads
    n_kv_heads_before = cfg.num_key_value_heads
    group_step = n_heads_before//n_kv_heads_before
    head_dim = hidden_size // n_heads_before
    assert hidden_size % n_heads_before == 0, "hidden_size must be divisible by num_attention_heads."
    
    # keep heads calculation
    keep_heads = max(1, int(round(n_heads_before * keep_ratio_heads)))
    keep_heads = min(keep_heads, n_heads_before)
    
    # important: ensure hidden_size is divisible by keep_heads (because hidden_size is kept unchanged)
    # adjust keep_heads to the nearest value that is divisible by hidden_size

    if keep_heads == n_heads_before:
        print(f"[ATTN] no head pruning (keep={keep_heads}/{n_heads_before})")
        return
    for i, layer in enumerate(layers):
        scores = attn_head_scores_from_KVQO(layer, n_kv_heads_before, n_heads_before, hidden_size, Pcfg.norm)
        # importance based keep head selection
        keep_idx = torch.argsort(scores, descending=True)[:keep_heads]
        keep_idx, _ = torch.sort(keep_idx)

        print(f"[ATTN] Layer {i:02d} Keep heads: {keep_heads} / {n_heads_before}")
        print(f"[ATTN] head indices kept: {keep_idx.tolist()}")

        prune_attention_heads_in_layer(layer, keep_idx, n_kv_heads_before, n_heads_before, hidden_size)

    # update config
    cfg.num_attention_heads = keep_heads * group_step
    cfg.num_key_value_heads = keep_heads
    cfg.prune_model = True
    cfg.head_dim = head_dim


###############################################
# MAIN
###############################################
def main():
    cfg = PruneConfig()

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)

    # Load config & model
    print("[Load] LlavaDreamForMaskedDiffusion...")
    with open(os.path.join(cfg.model_name, "config.json"), "r") as f:
        config_dict = json.load(f)
    config = LlavaDreamConfig.from_dict(config_dict)
    config.prune_model = False
    # create model on CPU for safety (use LlavaDreamForMaskedDiffusion like llada version)
    model = LlavaDreamForMaskedDiffusion(config)
    to_cpu(model)
    model.eval()

    # weights 로드
    load_Dream_weights(model, cfg.model_name)

    # activation checkpointing off
    disable_activation_checkpointing(model)

    # extract preserved components (before pruning)
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
            model_p = LlavaDreamForMaskedDiffusion(LlavaDreamConfig.from_dict(config_dict))
            to_cpu(model_p)
            model_p.eval()
            
            # copy state_dict from original model, instead of reloading file
            # this is more memory efficient than reloading file
            print("[Memory] Copying weights from original model...")
            model_p.load_state_dict(model.state_dict(), strict=False)
            
            # clear GPU cache (if any)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # note: here we don't replace vision_tower, because we will restore original weights later
        disable_activation_checkpointing(model_p)

        # 1) Attention heads (KVQO heads)
        attn_keep_ratio = getattr(cfg, "attn_head_keep_ratio", 1.0)  # ex) 0.75
        attn_keep_ratio = float(attn_keep_ratio)
        if attn_keep_ratio < 1.0:
            manual_prune_attention_heads(model_p, attn_keep_ratio, Pcfg=cfg)
        else:
            print("[ATTN] skip (keep_ratio=1.0)")

        # 2) FFN channels
        ffn_keep_ratio = float(getattr(cfg, "ffn_manual_ratio", 1.0))  # ex) 0.30
        print(f"[Plan] attn_head_keep={attn_keep_ratio:.3f}, ffn_keep={ffn_keep_ratio:.3f}")
        if ffn_keep_ratio < 1.0:
            manual_prune_ffn(model_p, ffn_keep_ratio, Pcfg=cfg)
        else:
            print("[FFN] skip (keep_ratio=1.0)")

        # restore preserved components to pruned model
        if preserved_components:
            print("\n[Restore] Restoring preserved components...")
            restore_preserved_components(model_p, preserved_components)

        final_params = estimate_param_count(model_p)
        final_h = human(final_params)
        print(f"[Params] Final ≈ {final_h}")

        # save folder name: automatically named by actual parameter count
        slug = safe_slug_from_human(final_h)  # "6.12B" -> "6p12B"
        save_dir = os.path.join(cfg.out_dir, f"lavida_dream_pruned_{slug}-norm{cfg.norm}-f_r{cfg.ffn_manual_ratio}-a_r{cfg.attn_head_keep_ratio}")
        os.makedirs(save_dir, exist_ok=True)

        # save to HuggingFace format
        print(f"\n[Save] Saving to HuggingFace format: {save_dir}")
        
        # 1. save config.json (pruned config, including updated n_heads etc.)
        model_p.config.save_pretrained(save_dir)
        
        # 2. save tokenizer
        tokenizer.save_pretrained(save_dir)
        
        # 3. save model weights to safetensors format (HuggingFace standard)
        state_dict = model_p.state_dict()
        
        # calculate the size of each shard (default maximum 5GB per shard)
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
        
        # clean up memory: if not the last target, delete pruned model to free memory
        # note: is_last_target is defined in the loop, so we need to recalculate it here
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
