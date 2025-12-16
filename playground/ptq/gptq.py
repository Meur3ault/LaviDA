# import fix_marlin  # Apply monkey patch for missing marlin kernels
from datasets import load_dataset
from gptqmodel import GPTQModel, QuantizeConfig
import os 

model_id = "jacklishufan/lavida-llada-v1.0-instruct"  # Using a non-gated model
quant_path = "jacklishufan/lavida-llada-v1.0-instruct-gptqmodel-4bit"

calibration_dataset = load_dataset(
    "allenai/c4",
    data_files="en/c4-train.00001-of-01024.json.gz",
    split="train",
).select(range(1024))["text"]

quant_config = QuantizeConfig(bits=4, group_size=128)

model = GPTQModel.load(model_id, quant_config)

# increase `batch_size` to match gpu/vram specs to speed up quantization
model.quantize(calibration_dataset, batch_size=2)

model.save(quant_path)