# int4_quant_utils.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class QuantLinear4bit(nn.Module):
    """
    Very simple 4-bit per-output-channel weight-only quantization.
    We store:
      - qweight: int8 with values in [-8, 7] (effectively 4 bits)
      - scale: per-output-channel float32 scale
      - (optional) bias: float32

    At inference we dequantize on the fly: w ≈ qweight * scale.
    """
    def __init__(self, in_features, out_features, bias=True, bits=4):
        super().__init__()
        assert bits == 4, "QuantLinear4bit currently assumes 4 bits."
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits

        # We'll pack the 4-bit values in int8 tensors (range [-8, 7])
        self.register_buffer(
            "qweight",
            torch.empty(out_features, in_features, dtype=torch.int8)
        )
        self.register_buffer(
            "scale",
            torch.empty(out_features, dtype=torch.float32)
        )

        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, dtype=torch.float32))
        else:
            self.bias = None

    @torch.no_grad()
    def quantize_from(self, linear: nn.Linear):
        """
        Quantize weights from a full-precision nn.Linear into 4-bit qweight + scale.
        Simple symmetric per-output-channel quantization.
        """
        w = linear.weight.data
        # Move to CPU for safety (optional)
        w = w.to(torch.float32).cpu()

        # Per-output-channel max absolute value
        max_val = w.abs().amax(dim=1)  # shape: [out_features]

        bits = self.bits
        qmin = -2 ** (bits - 1)        # -8
        qmax = 2 ** (bits - 1) - 1     #  7

        # Avoid division by zero
        scale = max_val / qmax
        scale[scale == 0] = 1.0

        w_scaled = w / scale.unsqueeze(1)
        w_q = torch.round(w_scaled).clamp_(qmin, qmax).to(torch.int8)

        self.qweight.copy_(w_q)
        self.scale.copy_(scale)

        if self.bias is not None and linear.bias is not None:
            self.bias.data.copy_(linear.bias.data.to(torch.float32).cpu())
        elif self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x):
        # Dequantize weights to x's dtype on the fly
        # qweight: [out, in], scale: [out]
        w = self.qweight.to(x.device, dtype=x.dtype)
        s = self.scale.to(x.device, dtype=x.dtype).view(-1, 1)
        w_dequant = w * s
        return F.linear(x, w_dequant, self.bias.to(x.device, dtype=x.dtype) if self.bias is not None else None)


def _get_parent_module(root: nn.Module, module_name: str) -> nn.Module:
    """
    Given a root module and a dotted module_name from named_modules(),
    return the parent module object.
    E.g. "model.layers.0.mlp.fc1" -> parent is root.model.layers.0.mlp
    """
    parts = module_name.split(".")
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent


@torch.no_grad()
def quantize_model_4bit_inplace(model: nn.Module, bits: int = 4, verbose: bool = True):
    """
    Walk through all submodules, find nn.Linear layers, replace them with QuantLinear4bit,
    and quantize their weights.

    This is intentionally generic: it doesn't care that the model_type is 'llada',
    it just grabs every nn.Linear in the LLaVA+LaViDa model.
    """
    if verbose:
        print(f"[quantize_model_4bit_inplace] Starting 4-bit quantization (bits={bits}) ...")

    linear_names = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            linear_names.append(name)

    if verbose:
        print(f"[quantize_model_4bit_inplace] Found {len(linear_names)} nn.Linear layers.")

    for name in linear_names:
        # Because we'll be mutating the module tree, re-fetch the module each time
        parent = _get_parent_module(model, name)
        child_name = name.split(".")[-1]
        old_linear = getattr(parent, child_name)

        if not isinstance(old_linear, nn.Linear):
            # Might have already been replaced in a previous step
            continue

        qlin = QuantLinear4bit(
            old_linear.in_features,
            old_linear.out_features,
            bias=(old_linear.bias is not None),
            bits=bits,
        )
        # Place the quant module on same device as old weights
        qlin.to(old_linear.weight.device)
        qlin.quantize_from(old_linear)

        setattr(parent, child_name, qlin)

    if verbose:
        print("[quantize_model_4bit_inplace] Quantization complete.")
