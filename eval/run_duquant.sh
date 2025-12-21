
export LLADA_VISION_ENCODER="google/siglip-so400m-patch14-384"
export QDLM_PATH="/root/autodl-tmp/QDLM"
export DUQUANT_CACHE_DIR="./cache"

set -x
# TASKS=
export TASKS=${TASKS:-"pope"}
export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7,8,9
export DEBUG_PRINT_IMAGE_RES=1
echo $TASKS

accelerate launch --num_processes=8 \
    -m lmms_eval \
    --model llava_llada_duquant \
    --model_args pretrained=$1,conv_template=llada,model_name=llava_llada,wbits=4,abits=16,calib_dataset=wikitext2,nsamples=128,quant_method=duquant,epochs=0 \
    --tasks $TASKS \
    --batch_size 1 \
    --gen_kwargs refix_lm=True \
    --log_samples \
    --log_samples_suffix llava_llada_duquant \
    --output_path ./logs/ --verbosity=DEBUG \
    ${@:2} \

#  export TASKS=${TASKS:-"mme,vqav2_val_lite,mmbench_en_dev_lite,textvqa_val,docvqa_val,chartqa_lite,infovqa_val_lite,scienceqa_full,ai2d,coco2017_cap_val_lite,mathverse_testmini_vision_dominant,mathvista_testmini_format"}
