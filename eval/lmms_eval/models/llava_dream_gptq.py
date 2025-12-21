import copy
import json
import logging
import math
import re
import warnings
import sys
from datetime import timedelta
from typing import List, Optional, Tuple, Union

import numpy as np
import os
import PIL
import torch
import accelerate
import transformers
from accelerate import Accelerator, DistributedType, InitProcessGroupKwargs
from accelerate.state import AcceleratorState
from decord import VideoReader, cpu
from packaging import version
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.load_video import read_video_pyav
import time

# Suppress warnings
warnings.filterwarnings("ignore")

# Configure logging
eval_logger = logging.getLogger("lmms-eval")

# Enable TF32 for CUDA
torch.backends.cuda.matmul.allow_tf32 = True
DEBUG_PRINT_OUTPUT = os.environ.get('DEBUG_PRINT_OUTPUT', False)

# Import LLaVA modules
DEBUG_LOAD_TRAINER = os.environ.get('DEBUG_LOAD_TRAINER', False)
try:
    from llava.constants import (
        DEFAULT_IM_END_TOKEN,
        DEFAULT_IM_START_TOKEN,
        DEFAULT_IMAGE_TOKEN,
        IGNORE_INDEX,
        IMAGE_TOKEN_INDEX,
    )
    from llava.conversation import SeparatorStyle, conv_templates
    from llava.mm_utils import (
        KeywordsStoppingCriteria,
        get_model_name_from_path,
        process_images,
        tokenizer_image_token,
    )
    from llava.model.builder import load_pretrained_model
except ImportError as e:
    eval_logger.debug(f"LLaVA is not installed. Please install LLaVA to use this model.\nError: {e}")

# Import AutoGPTQ modules from QDLM
QDLM_PATH = os.environ.get('QDLM_PATH', '/root/autodl-tmp/QDLM')
if QDLM_PATH not in sys.path:
    sys.path.insert(0, QDLM_PATH)
    sys.path.insert(0, os.path.join(QDLM_PATH, 'AutoGPTQ'))

try:
    from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
    AUTOGPTQ_AVAILABLE = True
except ImportError as e:
    eval_logger.warning(f"AutoGPTQ is not available. Please ensure QDLM_PATH is set correctly.\nError: {e}")
    AUTOGPTQ_AVAILABLE = False

# Determine best attention implementation
if version.parse(torch.__version__) >= version.parse("2.1.2"):
    best_fit_attn_implementation = "sdpa"
else:
    best_fit_attn_implementation = "eager"


def get_c4_calibration_data(nsamples, seed, seqlen, model_path, tokenizer):
    """Load c4 dataset for calibration (already cached in huggingface)"""
    from datasets import load_dataset
    import random
    
    eval_logger.info(f"Loading c4 dataset for calibration (nsamples={nsamples}, seqlen={seqlen})")
    
    # Load c4 dataset - will use cached version if available
    traindata = load_dataset(
        'allenai/c4', 
        data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, 
        split='train'
    )
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    calibration_data = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            text = traindata[i]['text']
            if len(text.strip()) > 0:
                encoded = tokenizer(text, return_tensors='pt', truncation=True, max_length=seqlen)
                if encoded.input_ids.shape[1] >= seqlen:
                    # Take a random slice of seqlen length
                    start_idx = random.randint(0, encoded.input_ids.shape[1] - seqlen)
                    inp = encoded.input_ids[:, start_idx:start_idx + seqlen]
                    attention_mask = torch.ones_like(inp)
                    calibration_data.append({"input_ids": inp, "attention_mask": attention_mask})
                    break
    
    eval_logger.info(f"Loaded {len(calibration_data)} calibration samples from c4 dataset")
    return calibration_data


@register_model("llava_dream_gptq")
class Llava_Dream_GPTQ(lmms):
    """
    Llava Dream Model with GPTQ quantization (only quantizes language model layers)
    Uses c4 dataset for calibration
    """

    def __init__(
        self,
        pretrained: str = "lmms-lab/llava-onevision-qwen2-7b-ov",
        truncation: Optional[bool] = True,
        device: Optional[str] = "cuda:0",
        batch_size: Optional[Union[int, str]] = 1,
        model_name: Optional[str] = None,
        attn_implementation: Optional[str] = best_fit_attn_implementation,
        device_map: Optional[str] = "cuda:0",
        conv_template: Optional[str] = "dream",
        use_cache: Optional[bool] = True,
        truncate_context: Optional[bool] = False,
        customized_config: Optional[str] = None,
        max_frames_num: Optional[int] = 32,
        mm_spatial_pool_stride: Optional[int] = 2,
        mm_spatial_pool_mode: Optional[str] = "bilinear",
        token_strategy: Optional[str] = "single",
        video_decode_backend: str = "decord",
        mc_num=16,
        # GPTQ parameters
        wbits: int = 4,
        group_size: int = 128,
        desc_act: bool = False,
        nsamples: int = 128,
        calib_dataset: str = "c4",
        **kwargs,
    ) -> None:
        super().__init__()
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        if not AUTOGPTQ_AVAILABLE:
            raise ImportError("AutoGPTQ is not available. Please set QDLM_PATH environment variable correctly.")

        accelerator_kwargs = InitProcessGroupKwargs(timeout=timedelta(weeks=52))
        accelerator = Accelerator(kwargs_handlers=[accelerator_kwargs])
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        elif accelerator.num_processes == 1 and device_map == "auto":
            self._device = torch.device(device)
            self.device_map = device_map
        else:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        
        self.mc_num = mc_num
        llava_model_args = {
            "multimodal": True,
        }
        if customized_config is not None:
            llava_model_args["customized_config"] = customized_config
        if attn_implementation is not None:
            llava_model_args["attn_implementation"] = attn_implementation
        if "use_flash_attention_2" in kwargs:
            llava_model_args["use_flash_attention_2"] = kwargs["use_flash_attention_2"]
        
        model_name = 'llava_dream'
        self.pretrained = pretrained
        self.token_strategy = token_strategy
        self.max_frames_num = max_frames_num
        self.mm_spatial_pool_stride = mm_spatial_pool_stride
        self.mm_spatial_pool_mode = mm_spatial_pool_mode
        self.video_decode_backend = video_decode_backend

        overwrite_config = {}
        overwrite_config["mm_spatial_pool_stride"] = self.mm_spatial_pool_stride
        overwrite_config["mm_spatial_pool_mode"] = self.mm_spatial_pool_mode
        cfg_pretrained = AutoConfig.from_pretrained(self.pretrained, trust_remote_code=True)

        llava_model_args["overwrite_config"] = overwrite_config
        
        if os.path.exists('/data1/jacklishufan/siglip-so400m-patch14-384'):
            vision_tower_path = "/data1/jacklishufan/siglip-so400m-patch14-384"
        else:
            vision_tower_path = "/data0/jacklishufan/siglip-so400m-patch14-384"
        
        vision_kwargs = dict(
            mm_vision_tower=os.environ.get('LLADA_VISION_ENCODER', vision_tower_path),
            mm_resampler_type=None,
            mm_projector_type='mlp2x_gelu',
            mm_hidden_size=1152,
            use_mm_proj=True
        )
            
        self._tokenizer, self._model, self._image_processor, self._max_length = load_pretrained_model(
            pretrained, None, model_name, device_map=self.device_map, 
            **llava_model_args, vision_kwargs=vision_kwargs
        )
        assert self._tokenizer is not None

        # Apply GPTQ quantization to language model layers only (not vision encoder)
        # AutoGPTQ supports "Dream" model type
        if accelerator.is_local_main_process:
            eval_logger.info("=== Starting GPTQ quantization (language model layers only) ===")
            eval_logger.info(f"Quantization config: wbits={wbits}, group_size={group_size}, desc_act={desc_act}")
        
        # Load calibration data from c4 dataset
        if accelerator.is_local_main_process:
            eval_logger.info(f"Loading calibration data from {calib_dataset} dataset (cached in huggingface)...")
        
        calibration_data = get_c4_calibration_data(
            nsamples=nsamples,
            seed=2,
            seqlen=2048,
            model_path=pretrained,
            tokenizer=self._tokenizer
        )
        
        # Prepare calibration examples in AutoGPTQ format
        calib_examples = []
        for item in calibration_data:
            example = {
                "input_ids": item["input_ids"].squeeze(0) if item["input_ids"].dim() > 1 else item["input_ids"],
            }
            if "attention_mask" in item:
                example["attention_mask"] = item["attention_mask"].squeeze(0) if item["attention_mask"].dim() > 1 else item["attention_mask"]
            calib_examples.append(example)
        
        # Create quantize config
        quantize_config = BaseQuantizeConfig(
            bits=wbits,
            group_size=group_size,
            desc_act=desc_act,
        )
        
        # Get the language model part (model.model.layers for dream)
        language_model = self._model.model
        
        # Use AutoGPTQ to quantize the language model layers
        if accelerator.is_local_main_process:
            eval_logger.info("Quantizing language model layers with AutoGPTQ...")
        
        try:
            from auto_gptq.modeling._utils import find_layers, make_quant
            from auto_gptq.quantization import GPTQ
            
            # Find all Linear layers in model.layers (for dream models)
            layers = language_model.layers if hasattr(language_model, 'layers') else [language_model]
            
            # Get layer structure from first layer
            first_layer_layers = find_layers(layers[0], layers=[torch.nn.Linear])
            
            # Filter out embedding/output layers
            layers_to_quantize = {}
            for name, layer in first_layer_layers.items():
                if not any(skip in name.lower() for skip in ['embed', 'norm', 'lm_head']):
                    layers_to_quantize[name] = layer
            
            if accelerator.is_local_main_process:
                eval_logger.info(f"Found {len(layers_to_quantize)} Linear layer types to quantize per layer")
                eval_logger.info(f"Total layers: {len(layers)}")
            
            if len(layers_to_quantize) > 0:
                # Replace Linear layers with QuantLinear in all layers
                layers_module = language_model.layers if hasattr(language_model, 'layers') else language_model
                
                # Build full layer names for all layers
                all_layer_names = []
                if hasattr(language_model, 'layers'):
                    for layer_idx in range(len(layers)):
                        for layer_name in layers_to_quantize.keys():
                            all_layer_names.append(f"{layer_idx}.{layer_name}")
                else:
                    all_layer_names = list(layers_to_quantize.keys())
                
                if accelerator.is_local_main_process:
                    eval_logger.info(f"Replacing {len(all_layer_names)} Linear layers with QuantLinear...")
                
                make_quant(
                    layers_module,
                    names=all_layer_names,
                    bits=wbits,
                    group_size=group_size,
                    use_triton=False,
                    desc_act=desc_act,
                )
                
                # Now run GPTQ quantization algorithm
                if accelerator.is_local_main_process:
                    eval_logger.info("Collecting activations and quantizing weights with GPTQ...")
                
                # Collect inputs to each layer type
                layer_inputs_dict = {name: [] for name in layers_to_quantize.keys()}
                hooks = []
                
                def make_collect_hook(layer_name):
                    def hook_fn(module, input, output):
                        if len(layer_inputs_dict[layer_name]) < nsamples:
                            if isinstance(input, tuple) and len(input) > 0:
                                inp = input[0].detach()
                                layer_inputs_dict[layer_name].append(inp)
                    return hook_fn
                
                # Register hooks on first layer
                first_layer = layers[0]
                for name, layer in layers_to_quantize.items():
                    actual_layer = dict(first_layer.named_modules())[name]
                    hook = actual_layer.register_forward_hook(make_collect_hook(name))
                    hooks.append(hook)
                
                # Run calibration data through language model to collect activations
                language_model.eval()
                with torch.no_grad():
                    for item in calibration_data[:nsamples]:
                        input_ids = item["input_ids"].to(self._device)
                        attention_mask = item.get("attention_mask", None)
                        if attention_mask is not None:
                            attention_mask = attention_mask.to(self._device)
                        
                        try:
                            # Run through language model
                            if hasattr(language_model, 'embed_tokens'):
                                hidden_states = language_model.embed_tokens(input_ids)
                            else:
                                hidden_states = language_model(input_ids, attention_mask=attention_mask)
                                continue
                            
                            # Run through layers to collect activations
                            for layer in layers[:1]:  # Only need first layer for structure
                                hidden_states = layer(hidden_states, attention_mask=attention_mask)
                        except Exception:
                            pass
                
                # Remove hooks
                for hook in hooks:
                    hook.remove()
                
                # Quantize each layer using GPTQ algorithm
                quantized_count = 0
                for layer_idx, layer in enumerate(layers):
                    for name, _ in layers_to_quantize.items():
                        try:
                            quantized_layer = dict(layer.named_modules())[name]
                            
                            # Check if it's QuantLinear
                            from auto_gptq.nn_modules.qlinear import GeneralQuantLinear
                            if not isinstance(quantized_layer, GeneralQuantLinear):
                                continue
                            
                            # Initialize GPTQ
                            gptq = GPTQ(quantized_layer)
                            
                            # Add batch data
                            inputs = layer_inputs_dict.get(name, [])
                            if len(inputs) > 0:
                                for inp in inputs[:min(len(inputs), 32)]:
                                    inp = inp.to(self._device)
                                    with torch.no_grad():
                                        out = quantized_layer(inp)
                                        gptq.add_batch(inp, out)
                                
                                # Perform GPTQ quantization
                                gptq.fasterquant(
                                    blocksize=128,
                                    percdamp=0.01,
                                    group_size=group_size,
                                    actorder=desc_act,
                                    static_groups=False,
                                )
                                quantized_count += 1
                        except Exception as e:
                            if accelerator.is_local_main_process and layer_idx == 0:
                                eval_logger.debug(f"Layer {name} quantization: {e}")
                            continue
                
                if accelerator.is_local_main_process:
                    eval_logger.info(f"GPTQ quantization complete. Quantized {quantized_count} layers across {len(layers)} transformer layers.")
            else:
                if accelerator.is_local_main_process:
                    eval_logger.warning("No Linear layers found to quantize in language model.")
                    
        except Exception as e:
            if accelerator.is_local_main_process:
                eval_logger.warning(f"GPTQ quantization failed: {e}")
                import traceback
                eval_logger.debug(traceback.format_exc())
            eval_logger.info("Model will continue without quantization.")

        self._config = self._model.config
        self.model.eval()
        self.model.requires_grad_(False)
        self.truncation = truncation
        self.batch_size_per_gpu = int(batch_size)
        self.conv_template = conv_template
        self.use_cache = use_cache
        self.truncate_context = truncate_context
        assert self.batch_size_per_gpu == 1, "Llava currently does not support batched generation."

        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [DistributedType.FSDP, DistributedType.MULTI_GPU, DistributedType.DEEPSPEED]
            if accelerator.distributed_type == DistributedType.DEEPSPEED:
                kwargs = {
                    "train_micro_batch_size_per_gpu": self.batch_size_per_gpu,
                    "train_batch_size": self.batch_size_per_gpu * accelerator.num_processes,
                }
                AcceleratorState().deepspeed_plugin.deepspeed_config_process(must_match=True, **kwargs)
                eval_logger.info("Detected that you are using DistributedType.DEEPSPEED. Make sure you run `accelerate config` and set zero stage to 0")

            self.model.to(self._device).to(torch.bfloat16)
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes

        elif accelerator.num_processes == 1 and device_map == "auto":
            eval_logger.info(f"Using {accelerator.num_processes} devices with tensor parallelism")
            self._rank = 0
            self._world_size = 1

        else:
            eval_logger.info(f"Using single device: {self._device}")
            self.model.to(self._device).to(torch.bfloat16)
            self._rank = 0
            self._world_size = 1

    @property
    def config(self):
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        else:
            return self._model

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length

    def pad_sequence(self, input_ids, batch_first, padding_value):
        if self.tokenizer.padding_side == "left":
            input_ids = [torch.flip(_input_ids, [0]) for _input_ids in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=batch_first, padding_value=padding_value)
        if self.tokenizer.padding_side == "left":
            input_ids = torch.flip(input_ids, [1])
        return input_ids

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def tok_encode(self, string: str, left_truncate_len=None, add_special_tokens=None) -> List[int]:
        add_special_tokens = False if add_special_tokens is None else add_special_tokens
        encoding = self.tokenizer.encode(string, add_special_tokens=add_special_tokens)
        if left_truncate_len:
            encoding = encoding[-left_truncate_len:]
        return encoding

    def tok_decode(self, tokens):
        try:
            return self.tokenizer.decode(tokens)
        except:
            return self.tokenizer.decode([tokens])

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        # Same as llava_dream.py
        res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

        origin_image_aspect_ratio = getattr(self._config, "image_aspect_ratio", None)

        for contexts, doc_to_target, doc_to_visual, doc_id, task, split in [reg.args for reg in requests]:
            visual = doc_to_visual(self.task_dict[task][split][doc_id])

            if origin_image_aspect_ratio is not None and self._config.image_aspect_ratio != origin_image_aspect_ratio:
                self._config.image_aspect_ratio = origin_image_aspect_ratio
                eval_logger.info(f"Resetting image aspect ratio to {origin_image_aspect_ratio}")

            if visual is None or visual == []:
                visual = None
                task_type = "text"
                image_tensor = None
            else:
                if len(visual) > 1 or "image_aspect_ratio" not in self._config.__dict__:
                    self._config.image_aspect_ratio = "pad"
                    eval_logger.info(f"In Multi-Image setting, image aspect ratio: {self._config.image_aspect_ratio}")

                if "task_type" in self.metadata and self.metadata["task_type"] == "video" and "sample_frames" in self.metadata:
                    assert type(visual) == list, "sample_frames must be specified for video task"
                    sample_indices = np.linspace(0, len(visual) - 1, self.metadata["sample_frames"], dtype=int)
                    visual = [visual[i] for i in sample_indices]
                    assert len(visual) == self.metadata["sample_frames"]

                    image_tensor = process_images(visual, self._image_processor, self._config)
                    if type(image_tensor) is list:
                        image_tensor = [_image.to(dtype=torch.bfloat16, device=self.device) for _image in image_tensor]
                    else:
                        image_tensor = image_tensor.to(dtype=torch.bfloat16, device=self.device)

                    task_type = "video"

                elif isinstance(visual[0], PIL.Image.Image):
                    image_tensor = process_images(visual, self._image_processor, self._config)
                    if type(image_tensor) is list:
                        image_tensor = [_image.to(dtype=torch.bfloat16, device=self.device) for _image in image_tensor]
                    else:
                        image_tensor = image_tensor.to(dtype=torch.bfloat16, device=self.device)

                    task_type = "image"

                elif type(visual[0]) == str:
                    image_tensor = []
                    try:
                        if self.video_decode_backend == "decord":
                            frames = self.load_video(visual, self.max_frames_num)
                        elif self.video_decode_backend == "pyav":
                            frames = read_video_pyav(visual[0], num_frm=self.max_frames_num)
                        frames = self._image_processor.preprocess(frames, return_tensors="pt")["pixel_values"].half().cuda()
                        image_tensor.append(frames)
                    except Exception as e:
                        eval_logger.error(f"Error {e} in loading video")
                        image_tensor = None

                    task_type = "video"

            if image_tensor is not None and len(image_tensor) != 0 and DEFAULT_IMAGE_TOKEN not in contexts:
                placeholder_count = len(visual) if isinstance(visual, list) else 1
                if task_type == "video":
                    placeholder_count = len(frames) if self.token_strategy == "multiple" else 1
                image_tokens = [DEFAULT_IMAGE_TOKEN] * placeholder_count
                image_tokens = " ".join(image_tokens)
                prompts_input = image_tokens + "\n" + contexts
            else:
                prompts_input = contexts

            assert self.conv_template == "dream"
            
            if "llama_3" in self.conv_template or 'llada' in self.conv_template or 'dream' in self.conv_template:
                conv = copy.deepcopy(conv_templates[self.conv_template])
            else:
                conv = conv_templates[self.conv_template].copy()

            conv.append_message(conv.roles[0], prompts_input)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self.device)

            if type(doc_to_target) == str:
                continuation = doc_to_target
            else:
                continuation = doc_to_target(self.task_dict[task][split][doc_id])

            input_prompt = conv.get_prompt()
            input_ids = tokenizer_image_token(input_prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self.device)
            answers = continuation
            answer_ids = self.tokenizer(continuation)['input_ids']
            answer_ids = torch.tensor(continuation).to(input_ids.device).unsqueeze(0) 

            kwargs = {}
            if task_type == "image":
                kwargs["image_sizes"] = [[v.size[0], v.size[1]] for v in visual] if isinstance(visual, list) else [[visual.size[0], visual.size[1]]]
            elif task_type == "video":
                kwargs["modalities"] = ["video"]
                self._config.mm_spatial_pool_stride = self.mm_spatial_pool_stride
                self._config.mm_spatial_pool_mode = self.mm_spatial_pool_mode

            torch.cuda.empty_cache()
            likelyhoods = self.model.log_likelyhood_inference(
                input_ids,
                images=image_tensor.to(torch.bfloat16) if image_tensor is not None else None,
                image_sizes=None,
                verbose=True,
                answer=answer_ids,
                mc_num=self.mc_num,
            ) 

            res.append((float(-likelyhoods.item()), False))
            pbar.update(1)

        pbar.close()
        return res

    def flatten(self, input):
        if not input or any(i is None for i in input):
            return []
        new_list = []
        for i in input:
            if i:
                for j in i:
                    new_list.append(j)
        return new_list

    def load_video(self, video_path, max_frames_num):
        if type(video_path) == str:
            vr = VideoReader(video_path, ctx=cpu(0))
        else:
            vr = VideoReader(video_path[0], ctx=cpu(0))
        total_frame_num = len(vr)
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, max_frames_num, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
        spare_frames = vr.get_batch(frame_idx).asnumpy()
        return spare_frames

    def generate_until(self, requests: List[Instance]) -> List[str]:
        # Copy implementation from llava_dream.py
        res = []

        def _collate(x):
            toks = self.tok_encode(x[0])
            return -len(toks), x[0]

        metadata = requests[0].metadata
        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        num_iters = len(requests) // self.batch_size if len(requests) % self.batch_size == 0 else len(requests) // self.batch_size + 1
        pbar = tqdm(total=num_iters, disable=(self.rank != 0), desc="Model Responding")

        origin_image_aspect_ratio = getattr(self._config, "image_aspect_ratio", None)

        for chunk in chunks:
            batched_contexts, all_gen_kwargs, batched_doc_to_visual, batched_doc_id, batched_task, batched_split = zip(*chunk)
            task = batched_task[0]
            split = batched_split[0]
            batched_visuals = [batched_doc_to_visual[0](self.task_dict[task][split][ids]) for ids in batched_doc_id]
            assert len(batched_visuals) == 1

            gen_kwargs = all_gen_kwargs[0]
            if "until" in gen_kwargs:
                gen_kwargs.pop("until")

            question_input = []
            for visual, context in zip(batched_visuals, batched_contexts):
                if origin_image_aspect_ratio is not None and self._config.image_aspect_ratio != origin_image_aspect_ratio:
                    self._config.image_aspect_ratio = origin_image_aspect_ratio
                    eval_logger.info(f"Resetting image aspect ratio to {origin_image_aspect_ratio}")

                if visual is None or visual == []:
                    visual = None
                    task_type = "text"
                    placeholder_count = 0
                    image_tensor = None
                else:
                    if len(visual) > 1 or "image_aspect_ratio" not in self._config.__dict__:
                        self._config.image_aspect_ratio = getattr(gen_kwargs, "image_aspect_ratio", "pad")
                        eval_logger.info(f"In Multi-Image setting, image aspect ratio: {self._config.image_aspect_ratio}")

                    if "task_type" in metadata and metadata["task_type"] == "video" and "sample_frames" in metadata:
                        assert type(visual) == list, "sample_frames must be specified for video task"
                        sample_indices = np.linspace(0, len(visual) - 1, metadata["sample_frames"], dtype=int)
                        visual = [visual[i] for i in sample_indices]
                        assert len(visual) == metadata["sample_frames"]

                        image_tensor = process_images(visual, self._image_processor, self._config)
                        if type(image_tensor) is list:
                            image_tensor = [_image.to(dtype=torch.bfloat16, device=self.device) for _image in image_tensor]
                        else:
                            image_tensor = image_tensor.to(dtype=torch.bfloat16, device=self.device)

                        task_type = "video"
                        placeholder_count = 1

                    elif type(visual[0]) == PIL.Image.Image:
                        image_tensor = process_images(visual, self._image_processor, self._config)
                        if type(image_tensor) is list:
                            image_tensor = [_image.to(dtype=torch.bfloat16, device=self.device) for _image in image_tensor]
                        else:
                            image_tensor = image_tensor.to(dtype=torch.bfloat16, device=self.device)

                        task_type = "image"
                        placeholder_count = len(visual) if isinstance(visual, list) else 1

                    elif type(visual[0]) == str:
                        image_tensor = []
                        try:
                            if self.video_decode_backend == "decord":
                                frames = self.load_video(visual, self.max_frames_num)
                            elif self.video_decode_backend == "pyav":
                                frames = read_video_pyav(visual[0], num_frm=self.max_frames_num)
                            frames = self._image_processor.preprocess(frames, return_tensors="pt")["pixel_values"].half().cuda()
                            image_tensor.append(frames)
                        except Exception as e:
                            eval_logger.error(f"Error {e} in loading video")
                            image_tensor = None

                        task_type = "video"
                        placeholder_count = len(frames) if self.token_strategy == "multiple" else 1

                if image_tensor is not None and len(image_tensor) != 0 and DEFAULT_IMAGE_TOKEN not in context:
                    image_tokens = [DEFAULT_IMAGE_TOKEN] * placeholder_count
                    image_tokens = " ".join(image_tokens)
                    question = image_tokens + "\n" + context
                else:
                    question = context

                if "llama_3" in self.conv_template or 'llada' in self.conv_template or 'dream' in self.conv_template:
                    conv = copy.deepcopy(conv_templates[self.conv_template])
                else:
                    conv = conv_templates[self.conv_template].copy()

                if utils.is_json(question):
                    question = json.loads(question)
                    for idx, item in enumerate(question):
                        role = conv.roles[idx % 2]
                        message = item["value"]
                        conv.append_message(role, message)

                    assert len(conv.messages) % 2 == 1
                    conv.append_message(conv.roles[1], None)
                    prompt_question = conv.get_prompt()
                    question_input.append(prompt_question)
                else:
                    conv.append_message(conv.roles[0], question)
                    conv.append_message(conv.roles[1], None)
                    prompt_question = conv.get_prompt()
                    question_input.append(prompt_question)

            if "max_new_tokens" not in gen_kwargs:
                gen_kwargs["max_new_tokens"] = 256
            
            if "temperature" not in gen_kwargs:
                gen_kwargs["temperature"] = 0
            if "do_sample" not in gen_kwargs:
                gen_kwargs["do_sample"] = False
            if "top_p" not in gen_kwargs:
                gen_kwargs["top_p"] = None
            if "num_beams" not in gen_kwargs:
                gen_kwargs["num_beams"] = 1
            schedule_kwargs = {}
            for key in list(gen_kwargs.keys()):
                if key.startswith('schedule__'):
                    value = gen_kwargs.pop(key)
                    schedule_kwargs[key.replace('schedule__','')] = value
                    
            if 'block_length' not in gen_kwargs:
                gen_kwargs['block_length'] = min(128,gen_kwargs["max_new_tokens"])
            if 'step_per_block' not in gen_kwargs:
                gen_kwargs['step_per_block'] = gen_kwargs['block_length']
            gen_kwargs["temperature"] = 0 

            input_ids_list = [tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt") for prompt in question_input]
            pad_token_ids = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
            input_ids = self.pad_sequence(input_ids_list, batch_first=True, padding_value=pad_token_ids).to(self.device)
            attention_masks = input_ids.ne(pad_token_ids).to(self.device)

            if task_type == "image":
                gen_kwargs["image_sizes"] = [batched_visuals[0][idx].size for idx in range(len(batched_visuals[0]))]
            elif task_type == "video":
                stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
                keywords = [stop_str]
                stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
                gen_kwargs["modalities"] = ["video"]
                gen_kwargs["stopping_criteria"] = [stopping_criteria]
                self._config.mm_spatial_pool_stride = self.mm_spatial_pool_stride
                self._config.mm_spatial_pool_mode = self.mm_spatial_pool_mode

            if "image_aspect_ratio" in gen_kwargs.keys():
                gen_kwargs.pop("image_aspect_ratio")
            if len(schedule_kwargs) > 0:
                gen_kwargs['schedule_kwargs'] = schedule_kwargs
            try:
                with torch.inference_mode():
                    cont = self.model.generate(input_ids, 
                                               attention_mask=attention_masks, 
                                               pad_token_id=pad_token_ids,
                                               images=image_tensor, 
                                               use_cache=self.use_cache,
                                               **gen_kwargs).sequences

                text_outputs = self.tokenizer.batch_decode(cont, skip_special_tokens=True)
                text_outputs = [text_output.lstrip('!').replace('<|im_end|>\n','') for text_output in text_outputs]
            except Exception as e:
                raise e

            text_outputs = [response.strip() for response in text_outputs]
            res.extend(text_outputs)
            self.cache_hook.add_partial("generate_until", (context, gen_kwargs), text_outputs)
            pbar.update(1)
        
        res = re_ords.get_original(res)
        pbar.close()
        return res
