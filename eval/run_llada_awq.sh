export LLADA_VISION_ENCODER="google/siglip-so400m-patch14-384"
export QDLM_PATH="/root/autodl-tmp/QDLM"

model_path="/root/autodl-tmp/huggingface_cache/hub/models--jacklishufan--lavida-llada-v1.0-instruct/snapshots/814b2e364e82390f03df451bdf4e81e8ba8eab37"
# TASKS=
# export TASKS=${TASKS:-"pope"}
export TASKS="coco2017_cap_val_lite"
export CUDA_VISIBLE_DEVICES=0
export DEBUG_PRINT_IMAGE_RES=1
echo $TASKS

accelerate launch --num_processes=1 \
    -m lmms_eval \
    --model llava_llada_awq \
    --model_args pretrained=$model_path,conv_template=llada,model_name=llava_llada,w_bit=4,q_group_size=128,no_zero_point=False,q_backend=fake,nsamples=512,calib_dataset=c4 \
    --tasks $TASKS \
    --batch_size 1 \
    --gen_kwargs prefix_lm=True,step_ratio=$1,schedule=shift,schedule__shift=0.33 \
    --log_samples \
    --log_samples_suffix llava_llada_awq \
    --output_path ./logs/ --verbosity=DEBUG \
    ${@:2} \

#  export TASKS=${TASKS:-"mme,vqav2_val_lite,mmbench_en_dev_lite,textvqa_val,docvqa_val,chartqa_lite,infovqa_val_lite,scienceqa_full,ai2d,coco2017_cap_val_lite,mathverse_testmini_vision_dominant,mathvista_testmini_format"}
