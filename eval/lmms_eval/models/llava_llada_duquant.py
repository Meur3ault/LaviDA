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
from transformers import AutoConfig

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

# Import DuQuant modules from QDLM
QDLM_PATH = os.environ.get('QDLM_PATH', '/root/autodl-tmp/QDLM')
if QDLM_PATH not in sys.path:
    sys.path.insert(0, QDLM_PATH)
    sys.path.insert(0, os.path.join(QDLM_PATH, 'DuQuant'))

try:
    from quantize.duquant import duquant
    from datautils import get_loaders
    DUQUANT_AVAILABLE = True
except ImportError as e:
    eval_logger.warning(f"DuQuant is not available. Please ensure QDLM_PATH is set correctly.\nError: {e}")
    DUQUANT_AVAILABLE = False

# Determine best attention implementation
if version.parse(torch.__version__) >= version.parse("2.1.2"):
    best_fit_attn_implementation = "sdpa"
else:
    best_fit_attn_implementation = "eager"


@register_model("llava_llada_duquant")
class Llava_Llada_DuQuant(lmms):
    """
    Llava Model with DuQuant quantization (only quantizes language model transformer part)
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
        conv_template: Optional[str] = "llava_llada",
        use_cache: Optional[bool] = True,
        truncate_context: Optional[bool] = False,
        customized_config: Optional[str] = None,
        max_frames_num: Optional[int] = 32,
        mm_spatial_pool_stride: Optional[int] = 2,
        mm_spatial_pool_mode: Optional[str] = "bilinear",
        token_strategy: Optional[str] = "single",
        video_decode_backend: str = "decord",
        mc_num=16,
        # DuQuant parameters
        wbits: int = 4,
        abits: int = 16,
        calib_dataset: str = "wikitext2",
        nsamples: int = 128,
        quant_method: Optional[str] = "duquant",
        epochs: int = 0,
        let: bool = False,
        lwc: bool = False,
        smooth: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        if not DUQUANT_AVAILABLE:
            raise ImportError("DuQuant is not available. Please set QDLM_PATH environment variable correctly.")

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
        
        model_name = 'llava_llada'
        self.overwrite_image_aspect = os.environ.get("LLAVA_OVERWRITE_IMAGE_ASPECT", None)
        self.pretrained = pretrained
        self.token_strategy = token_strategy
        self.max_frames_num = max_frames_num
        self.mm_spatial_pool_stride = mm_spatial_pool_stride
        self.mm_spatial_pool_mode = mm_spatial_pool_mode
        self.video_decode_backend = video_decode_backend

        overwrite_config = {}
        overwrite_config["mm_spatial_pool_stride"] = self.mm_spatial_pool_stride
        overwrite_config["mm_spatial_pool_mode"] = self.mm_spatial_pool_mode
        cfg_pretrained = AutoConfig.from_pretrained(self.pretrained)

        llava_model_args["overwrite_config"] = overwrite_config
        
        if os.path.exists('/data1/jacklishufan/siglip-so400m-patch14-384'):
            vision_tower_path = "/data1/jacklishufan/siglip-so400m-patch14-384"
        else:
            vision_tower_path = "/data0/jacklishufan/siglip-so400m-patch14-384"
        
        vision_kwargs = dict(
            mm_vision_tower=os.environ.get('LLADA_VISION_ENCODER', vision_tower_path),
            mm_resampler_type=None,
            mm_projector_type=os.environ.get('LLADA_VISION_PROJECTOR', 'mlp2x_gelu'),
            mm_hidden_size=int(os.environ.get('LLADA_VISION_ENCODER_HIDDEN_SIZE', 1152)),
            mm_pooler_ratio=int(os.environ.get('LLADA_MM_POOLER_RATIO', 2)),
            use_mm_proj=True,
            mm_patch_merge_type='spatial_unpad',
        )
        resize_embeddings = True
        if DEBUG_LOAD_TRAINER:
            resize_embeddings = False
            
        self._tokenizer, self._model, self._image_processor, self._max_length = load_pretrained_model(
            pretrained, None, model_name, device_map=self.device_map, 
            **llava_model_args, vision_kwargs=vision_kwargs, resize_embeddings=resize_embeddings
        )
        assert self._tokenizer is not None

        self._config = self._model.config
        self.model.eval()
        self.model.model.set_activation_checkpointing(None)
        self.model.requires_grad_(False)
        self.truncation = truncation
        self.batch_size_per_gpu = int(batch_size)
        self.conv_template = conv_template
        self.use_cache = use_cache
        self.truncate_context = truncate_context
        assert self.batch_size_per_gpu == 1, "Llava currently does not support batched generation."

        # Apply DuQuant quantization to transformer only (language model part)
        if wbits < 16 or abits < 16:
            if accelerator.is_local_main_process:
                eval_logger.info("=== Starting DuQuant quantization (transformer only) ===")
            
            # Create a wrapper class for the model to work with DuQuant
            class ModelWrapper:
                def __init__(self, model, device, seqlen=2048):
                    self.model = model
                    self.device = device
                    self.seqlen = seqlen
                    self.tokenizer = None
            
            # Create args object for DuQuant
            class QuantArgs:
                def __init__(self):
                    self.wbits = wbits
                    self.abits = abits
                    self.calib_dataset = calib_dataset
                    self.nsamples = nsamples
                    self.batch_size = 1
                    self.seed = 2
                    self.net = "llada"
                    self.model = pretrained
                    self.model_family = "llada"
                    self.quant_method = quant_method
                    self.epochs = epochs
                    self.let = let
                    self.lwc = lwc
                    self.smooth = smooth
                    self.deactive_amp = (wbits < 16 and wbits >= 8) or (abits < 16 and abits >= 8)
                    self.alpha = 0.5
                    self.let_alpha = 0.8
                    self.let_lr = 5e-3
                    self.lwc_lr = 1e-2
                    self.wd = 0
                    self.smooth_epochs = 0
                    self.symmetric = False
                    self.a_dynamic_method = "per_token"
                    self.w_dynamic_method = "per_channel"
                    self.group_size = None
                    self.act_group_size = None
                    self.lac = None
                    self.swc = None
                    self.max_rotation_step = 256
                    self.permutation_times = 1
                    self.block_size = 128
                    self.act_scales = None
                    self.act_shifts = None
                    self.resume = None
                    self.save_dir = None
                    self.cache_dir = os.environ.get('DUQUANT_CACHE_DIR', './cache')
                    self.aug_loss = False
                    self.limit = -1
                    
                    # Weight quantization params
                    self.weight_quant_params = {
                        "n_bits": wbits,
                        "per_channel_axes": [0],
                        "symmetric": self.symmetric,
                        "dynamic_method": self.w_dynamic_method,
                        "group_size": self.group_size,
                        "lwc": self.lwc,
                        "swc": self.swc,
                        "quant_method": self.quant_method,
                        "block_size": self.block_size,
                        "max_rotation_step": self.max_rotation_step,
                        "permutation_times": self.permutation_times,
                    }
                    # Activation quantization params
                    self.act_quant_params = {
                        "n_bits": abits,
                        "per_channel_axes": [],
                        "symmetric": False,
                        "lac": self.lac,
                        "act_group_size": self.act_group_size,
                        "dynamic_method": self.a_dynamic_method,
                        "quant_method": self.quant_method,
                        "block_size": self.block_size,
                        "max_rotation_step": self.max_rotation_step,
                        "permutation_times": self.permutation_times,
                    }
                    # QKV quantization params
                    self.q_quant_params = copy.copy(self.act_quant_params)
                    self.k_quant_params = copy.copy(self.act_quant_params)
                    self.v_quant_params = copy.copy(self.act_quant_params)
                    self.p_quant_params = {"n_bits": 16, "metric": "fix0to1"}
                    
                    # Per-layer quantization params
                    for name in ['q', 'k', 'v', 'gate', 'up', 'down', 'o']:
                        setattr(self, f'{name}_weight_quant_params', copy.copy(self.weight_quant_params))
                        setattr(self, f'{name}_act_quant_params', copy.copy(self.act_quant_params))
            
            quant_args = QuantArgs()
            
            # Create model wrapper
            lm_wrapper = ModelWrapper(self.model, self._device)
            lm_wrapper.tokenizer = self._tokenizer
            
            # Load calibration data
            cache_dataloader = f'{quant_args.cache_dir}/dataloader_llada_{calib_dataset}_{nsamples}.cache'
            if os.path.exists(cache_dataloader):
                dataloader = torch.load(cache_dataloader)
                if accelerator.is_local_main_process:
                    eval_logger.info(f"Loaded calibration data from {cache_dataloader}")
            else:
                if accelerator.is_local_main_process:
                    eval_logger.info(f"Loading calibration dataset: {calib_dataset}")
                dataloader, _ = get_loaders(
                    calib_dataset,
                    nsamples=nsamples,
                    seed=quant_args.seed,
                    model=pretrained,
                    seqlen=lm_wrapper.seqlen,
                )
                os.makedirs(quant_args.cache_dir, exist_ok=True)
                torch.save(dataloader, cache_dataloader)
            
            # Load activation scales/shifts if smooth quantization is enabled
            act_scales = None
            act_shifts = None
            if smooth:
                base_dir = os.path.join(QDLM_PATH, 'DuQuant', 'act_scales')
                act_scales_path = os.path.join(base_dir, f'{quant_args.net}.pt')
                act_shifts_path = os.path.join(base_dir.replace('act_scales', 'act_shifts'), f'{quant_args.net}.pt')
                if os.path.exists(act_scales_path):
                    act_scales = torch.load(act_scales_path)
                if os.path.exists(act_shifts_path):
                    act_shifts = torch.load(act_shifts_path)
            
            # Apply DuQuant quantization - this will only quantize transformer.blocks
            if accelerator.is_local_main_process:
                eval_logger.info("Applying DuQuant quantization to transformer blocks only...")
            
            # Note: duquant function will automatically handle only quantizing transformer.blocks
            # for llada models (see duquant.py line 83-94)
            duquant(
                lm_wrapper,
                quant_args,
                dataloader,
                act_scales,
                act_shifts,
                logger=eval_logger if accelerator.is_local_main_process else None,
            )
            
            if accelerator.is_local_main_process:
                eval_logger.info("DuQuant quantization complete.")

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
            self._model.model.transformer = accelerator.prepare(self.model.model.transformer)
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

    # Copy the rest of the methods from llava_llada.py
    # For brevity, I'll include the key methods. The rest should be identical to llava_llada.py
    
    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
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

            if "llama_3" in self.conv_template or 'llada' in self.conv_template:
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
        return spare_frames  # (frames, height, width, channels)

    def generate_until(self, requests: List[Instance]) -> List[str]:
        # This method is identical to llava_llada.py's generate_until
        # Copy the full implementation from llava_llada.py
        res = []

        def _collate(x):
            toks = self.tok_encode(x[0])
            return -len(toks), x[0]

        metadata = requests[0].metadata
        if DEBUG_PRINT_OUTPUT:
            re_ords = utils.Collator([reg.args for reg in requests], lambda x:x[-3], grouping=True)
        else:
            re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        num_iters = len(requests) // self.batch_size if len(requests) % self.batch_size == 0 else len(requests) // self.batch_size + 1
        pbar = tqdm(total=num_iters, disable=(self.rank != 0), desc="Model Responding")

        origin_image_aspect_ratio = getattr(self._config, "image_aspect_ratio", None)
        if DEBUG_LOAD_TRAINER:
            ckpt1 = torch.load(DEBUG_LOAD_TRAINER, map_location='cpu')
            ckpt1 = {k.replace('module.model','model'):v for k,v in ckpt1.items()}
            _res = self.model.load_state_dict(ckpt1,strict=False)
            print(f"DEBUG_LOAD_TRAINER:{DEBUG_LOAD_TRAINER} {_res}")
            print("Something is broken if above line does not show all keys matched!!!")
            del ckpt1
        delta_t = 0
        num_generated = 0
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
                t0 = time.time()
                if origin_image_aspect_ratio is not None and self._config.image_aspect_ratio != origin_image_aspect_ratio:
                    self._config.image_aspect_ratio = origin_image_aspect_ratio
                    eval_logger.info(f"Resetting image aspect ratio to {origin_image_aspect_ratio}")
                if self.overwrite_image_aspect:
                    self._config.image_aspect_ratio = self.overwrite_image_aspect
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

                if "llama_3" in self.conv_template or 'llada' in self.conv_template:
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
            if len(schedule_kwargs) > 0:
                gen_kwargs['schedule_kwargs'] = schedule_kwargs
            
            if 'block_length' not in gen_kwargs:
                gen_kwargs['block_length'] = min(128,gen_kwargs["max_new_tokens"])
            if 'step_per_block' not in gen_kwargs and 'step_ratio' not in gen_kwargs:
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
            try:
                with torch.inference_mode():
                    cont = self.model.generate(input_ids, attention_mask=attention_masks, pad_token_id=pad_token_ids, images=image_tensor, use_cache=self.use_cache, **gen_kwargs)

                text_outputs = self.tokenizer.batch_decode(cont, skip_special_tokens=True)
                text_outputs = [text_output.lstrip('!') for text_output in text_outputs]
            except Exception as e:
                raise e
            
            t1 = time.time()
            delta_t += t1-t0
            num_generated +=1
            print(f"Avg Latency (of {num_generated}): {delta_t/num_generated}")
            if DEBUG_PRINT_OUTPUT:
                print(f'\n--------Start of Sample {batched_doc_id[0]}---------')
                print("Question: ",prompt_question)
                print("Answer: ",text_outputs)
                print("Answer: ",gen_kwargs)
                print('--------End---------')

            text_outputs = [response.strip() for response in text_outputs]
            res.extend(text_outputs)
            self.cache_hook.add_partial("generate_until", (context, gen_kwargs), text_outputs)
            pbar.update(1)
        
        res = re_ords.get_original(res)
        pbar.close()
        return res
