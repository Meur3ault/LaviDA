export LLADA_VISION_ENCODER="google/siglip-so400m-patch14-384"
export QDLM_PATH="/root/autodl-tmp/QDLM"

set -x
export TASKS=${TASKS:-"mme"}
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export DEBUG_PRINT_IMAGE_RES=1
# max_new_tokens
accelerate launch --num_processes=8 --main_process_port=25511 \
    -m lmms_eval \
    --model llava_dream_awq \
    --model_args pretrained=$1,conv_template=dream,model_name=llava_dream,w_bit=4,q_group_size=128,no_zero_point=False,q_backend=fake,nsamples=128,calib_dataset=c4 \
    --tasks $TASKS \
    --batch_size 1 \
    --gen_kwargs alg=topk_margin,prefix_lm=True \
    --log_samples \
    --log_samples_suffix llava_dream_awq \
    --output_path ./logs/ --verbosity=DEBUG \
    ${@:2} \
