NUM_FRAMES=${NUM_FRAMES:-"6"}
OUTPUT_DIR=${OUTPUT_DIR:-"./output/internvl_answer"}
export PYTHONWARNINGS="ignore"

for i in $(seq -w 0 10); do
    POS_ID="10-${i}"

    echo "Running with POS_ID=${POS_ID} ..."

    # inference
    python3 ./evaluation/transformer/Internvl3.py \
          --model_dir OpenGVLab/InternVL3-8B \
          --video_dir ./LevelGauge/videos \
          --gt_file ./LevelGauge/json/Pos_MCQA_300_final.json \
          --pos ${POS_ID} \
          --output_dir ${OUTPUT_DIR} \
          --output_name ${POS_ID} \
          --num_frames ${NUM_FRAMES}

    # eval
    python3 ./evaluation/eval_MCQA.py \
        --pred_path ${OUTPUT_DIR}/${POS_ID}.json \
        --pos ${POS_ID} \
        --output_dir ./output/internvl_acc
done
