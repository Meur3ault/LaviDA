

LLADA_VISION_ENCODER="google/siglip-so400m-patch14-384"
model_path="/root/autodl-tmp/huggingface_cache/hub/models--jacklishufan--lavida-llada-v1.0-instruct"
# model_path="/root/autodl-tmp/huggingface_cache/hub/models--jacklishufan--lavida-dream-v1.0-instruct"
set -x
# TASKS=
export TASKS=${TASKS:-"pope"}
export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7,8,9
export DEBUG_PRINT_IMAGE_RES=1
echo $TASKS

accelerate launch --num_processes=8 \
    -m lmms_eval \
    --model llava_llada \
    --model_args pretrained=$model_path,conv_template=llada,model_name=llava_llada \
    --tasks $TASKS \
    --batch_size 1 \
    --gen_kwargs refix_lm=True \
    --log_samples \
    --log_samples_suffix llava_llada \
    --output_path ./logs/ --verbosity=DEBUG \
    ${@:2} \

#  export TASKS=${TASKS:-"mme,vqav2_val_lite,mmbench_en_dev_lite,textvqa_val,docvqa_val,chartqa_lite,infovqa_val_lite,scienceqa_full,ai2d,coco2017_cap_val_lite,mathverse_testmini_vision_dominant,mathvista_testmini_format"}

