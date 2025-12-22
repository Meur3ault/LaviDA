

export LLADA_VISION_ENCODER="google/siglip-so400m-patch14-384"
model_path="/root/autodl-tmp/huggingface_cache/hub/models--jacklishufan--lavida-llada-v1.0-instruct"
set -x
# TASKS=
export TASKS="coco2017_cap_val_lite"
export CUDA_VISIBLE_DEVICES=0
export DEBUG_PRINT_IMAGE_RES=1
echo $TASKS
#llava_llada_int4，llava_llada_int8，llava_llada
accelerate launch --num_processes=1 \
    -m lmms_eval \
    --model llava_llada\
    --model_args pretrained=$1,conv_template=llada,model_name=llava_llada \
    --tasks $TASKS \
    --batch_size 1 \
    --gen_kwargs prefix_lm=True,step_ratio=0.25,schedule=shift,schedule__shift=0.33 \
    --log_samples \
    --log_samples_suffix llava_llada \
    --output_path ./logs/ --verbosity=DEBUG \
    ${@:2} \
