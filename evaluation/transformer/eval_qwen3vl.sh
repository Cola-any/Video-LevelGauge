NUM_FRAMES=${NUM_FRAMES:-"6"}
POS_ID=${POS_ID:-"10-05"}
OUTPUT_DIR=${OUTPUT_DIR:-"./output/qwen3vl_output"}
export PYTHONWARNINGS="ignore"
gpu_list="${CUDA_VISIBLE_DEVICES:-"0,1,2,3,4,5,6,7"}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

for IDX in $(seq 0 $((CHUNKS-1))); do
  CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python3 ./evaluation/transformer/Qwen3-VL.py \
      --model_dir Qwen/Qwen3-VL-8B-Instruct \
      --video_dir ./LevelGauge/videos \
      --gt_file ./LevelGauge/json/Pos_MCQA_300_final.json \
      --pos ${POS_ID} \
      --output_dir ${OUTPUT_DIR} \
      --output_name ${CHUNKS}_${IDX} \
      --num_chunks ${CHUNKS} \
      --chunk_idx ${IDX} \
      --num_frames ${NUM_FRAMES} &
done

wait

output_dir=${OUTPUT_DIR} 
output_file=./output/qwen3vl_output/merge_${POS_ID}.jsonl
temp_dir=./output/qwen3vl_output

# # Clear out the output file if it exists.
> "${output_file}"

# # Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ${output_dir}/${CHUNKS}_${IDX}.json >> "${output_file}"
done

# ################################# Eval ##################################

python3 ./evaluation/eval_MCQA.py \
    --pred_path ${output_file} \
    --pos ${POS_ID} \
    --output_dir ./output/qwen3vl_acc
