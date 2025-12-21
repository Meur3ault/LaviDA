export LLADA_VISION_ENCODER="google/siglip-so400m-patch14-384"
export QDLM_PATH="/root/autodl-tmp/QDLM"

set -x
export TASKS=${TASKS:-"mme"}
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export DEBUG_PRINT_IMAGE_RES=1
# max_new_tokens
accelerate launch --num_processes=8 --main_process_port=25511 \
    -m lmms_eval \
    --model llava_dream_quarot \
    --model_args pretrained=$1,conv_template=dream,model_name=llava_dream,a_bits=16,v_bits=16,k_bits=16,w_bits=16,w_clip=True,rotate=True,w_rtn=False,nsamples=128,cal_dataset=c4,seed=2 \
    --tasks $TASKS \
    --batch_size 1 \
    --gen_kwargs alg=topk_margin,prefix_lm=True \
    --log_samples \
    --log_samples_suffix llava_dream_quarot \
    --output_path ./logs/ --verbosity=DEBUG \
    ${@:2} \
