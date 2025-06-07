# IO Setting
MODE="new_redundant_case"  # 或 "mask" 或 "inpainting without grounding" 或 "inpainting" or "inpainting looped"
# OUTPUT_DIR="./pipeline/output/new_inpaint_without_SR"
OUTPUT_DIR="./MoldingHuman/data_aigc/video_repair/new_inpaint_without_SR"

IMAGE_DIR="./MoldingHuman/data_aigc/video_repair/new_inpaint_without_SR"

# Load Models Setting


OUTPUT_DIR="./pipeline/output/pipeline_test_0301_wzq"

IMAGE_DIR="./MoldingHuman/data_aigc/pure_ori_images"



MODEL_PATH_INPAINTING="./LLaVA/checkpoints/llava-v1.5-7b-task-train_ver_with_neg_inpainting_1721922413_0392356_2_epoch_without_eye_knee_filter.json"
REASONING_MODEL="None"

MODEL_PATH_MASK="./LLaVA/checkpoints/llava-v1.5-7b-task-train_only_mask_neg_tan_0809_1723184657.9318254.json/checkpoint-15000"


# Perception Setting
QUESTION="Are there any missing parts on the person shown in the image?"
TEXT_LIST="head,arm,leg,foot,hand,ear"
TEMPERATURE=0.2
TOP_P=None
NUM_BEAMS=1
MAX_NEW_TOKENS=512



# Re-Generation Setting
SIZE_UPPER_BOUND=100000
SIZE_LOWER_BOUND=10000
PROPORTION=0.15
MAX_LOOP=20
DEPRIVATIVE_TRESHOLD=0.50
REDUNDANT_TRESHOLD=0.4
COMP_RATIO_THRESHOLD_REDUNDANT=0.4
COMP_RATIO_THRESHOLD_DEPRIVATIVE=0.4
INPAINTING_SEED=24

CUDA_VISIBLE_DEVICES=0 python pipeline_serial_mod_inpaint_last_frame_trans_aigc1k.py \
    --mode "$MODE" \
    --output-dir "$OUTPUT_DIR" \
    --image-dir "$IMAGE_DIR" \
    --model-path "$MODEL_PATH_INPAINTING" \
    --reasoning-model "$REASONING_MODEL" \
    --size-upper-bound 100000 \
    --size-lower-bound 10000 \
    --proportion 0.15 \
    --max-loop $MAX_LOOP \
    --question "$QUESTION" \
    --text-list "$TEXT_LIST" \
    --temperature "$TEMPERATURE" \
    --num_beams "$NUM_BEAMS" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --deprivative_treshold $DEPRIVATIVE_TRESHOLD \
    --redundant_treshold $REDUNDANT_TRESHOLD \
    --comp_ratio_threshold_redundant $COMP_RATIO_THRESHOLD_REDUNDANT \
    --comp_ratio_threshold_deprivative $COMP_RATIO_THRESHOLD_DEPRIVATIVE \
    --inpainting_seed $INPAINTING_SEED 
    
