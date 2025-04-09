#!/bin/bash

# --- Configuration for Evaluation ---
# Define variables for clarity and easy modification

# Set the GPU you want to use (or leave empty/comment out for CPU)
export CUDA_VISIBLE_DEVICES=0 # Or the GPU ID you trained on/want to use

# --- Task and Mode ---
TASK_NAME="long_term_forecast"
IS_TRAINING=0 # Set to 0 for evaluation/testing

# --- Data Parameters (MUST MATCH TRAINING) ---
# Root path should point to the directory containing the data file folder
# Data path should be the specific CSV file name
ROOT_PATH="./data/Public_Health"     # CORRECTED: Removed trailing slash
DATA_FILE="US_FLURATIO_Week.csv"   # Your specific data file name
DATA_TYPE="custom"
FEATURES="S" # This is overwritten to 'S' in run.py for univariate, but set explicitly
TARGET="ratio"  # <<< IMPORTANT: VERIFY this is the correct target column name in your CSV file
FREQ="w"     # Frequency for time features (weekly for FLURATIO)

# --- Model and Checkpoint ID (MUST MATCH TRAINING) ---
# Construct the model_id based on the training script pattern:
# ${model_id_base}_${seed}_${seq_len}_${pred_len}_fullLLM_${use_fullmodel}
# model_id_base is typically derived from the basename of ROOT_PATH by run.py
SEED=2021
SEQ_LEN_TRAIN=24 # Sequence length used during *training*
PRED_LEN_EVAL=24 # Prediction length to evaluate (MUST match a trained model's pred_len)
USE_FULLMODEL_TRAIN=0 # Value used during training
MODEL_ID_BASE="Public_Health" # Used to construct the expected checkpoint folder name convention
MODEL_ID="${MODEL_ID_BASE}_${SEED}_${SEQ_LEN_TRAIN}_${PRED_LEN_EVAL}_fullLLM_${USE_FULLMODEL_TRAIN}"
MODEL_NAME="Reformer" # The specific model trained

# --- Sequence Lengths (MUST MATCH TRAINING for the specific PRED_LEN) ---
SEQ_LEN=$SEQ_LEN_TRAIN
LABEL_LEN=12    # Start token length used during training
PRED_LEN=$PRED_LEN_EVAL

# --- Core Model Hyperparameters (MUST MATCH TRAINING) ---
D_MODEL=512
N_HEADS=8
E_LAYERS=2     # Number of encoder layers used during training
D_LAYERS=1     # Number of decoder layers used during training
D_FF=2048
DROPOUT=0.1
ACTIVATION="gelu"
EMBED_TYPE="timeF"
FACTOR=1       # Attn factor
# DISTIL: Default is True, action is store_false. Omit flag if True was used in training.
# If training used --distil (meaning False), add --distil to the python command below.

# --- LLM Hyperparameters (MUST MATCH TRAINING) ---
LLM_MODEL="BERT"
LLM_LAYERS=6       # Default from run.py if not specified otherwise during training
TEXT_LEN=4         # From week_health.sh example
PROMPT_WEIGHT=0.1  # From week_health.sh example
POOL_TYPE="avg"    # From week_health.sh example
USE_FULLMODEL=$USE_FULLMODEL_TRAIN # From week_health.sh example
USE_CLOSEDLLM=0    # Assuming BERT is open source here
HUGGINGFACE_TOKEN="NA" # Not needed for BERT

# --- Run/Output Parameters ---
CHECKPOINTS_PATH="./checkpoints/"         # Path where checkpoints are loaded from
EXP_DES="Eval_${MODEL_NAME}_Health_${SEQ_LEN}_${PRED_LEN}" # Description for this evaluation run
# INVERSE: Default is False, action is store_true. Add --inverse flag below if True used in training.
SAVE_NAME="result_eval_${MODEL_NAME}_Health_BERT" # File to append evaluation metrics

# --- Execution ---
echo "Starting evaluation for Model: $MODEL_NAME, Seq Len: $SEQ_LEN, Pred Len: $PRED_LEN"
echo "Using Checkpoint ID convention: $MODEL_ID"
echo "Attempting to load checkpoint based on parameters from: $CHECKPOINTS_PATH/$TASK_NAME*${MODEL_ID}*${MODEL_NAME}*/checkpoint.pth" # Show roughly what path it looks for

python -u run.py \
  --task_name $TASK_NAME \
  --is_training $IS_TRAINING \
  --root_path "$ROOT_PATH" \
  --data_path "$DATA_FILE" \
  --model_id "$MODEL_ID" \
  --model $MODEL_NAME \
  --data $DATA_TYPE \
  --features $FEATURES \
  --target "$TARGET" \
  --freq $FREQ \
  --checkpoints "$CHECKPOINTS_PATH" \
  --seq_len $SEQ_LEN \
  --label_len $LABEL_LEN \
  --pred_len $PRED_LEN \
  --d_model $D_MODEL \
  --n_heads $N_HEADS \
  --e_layers $E_LAYERS \
  --d_layers $D_LAYERS \
  --d_ff $D_FF \
  --dropout $DROPOUT \
  --factor $FACTOR \
  --activation $ACTIVATION \
  --embed $EMBED_TYPE \
  `# --distil # Add this flag ONLY if training used --distil (i.e., distil=False)` \
  --llm_model $LLM_MODEL \
  --llm_layers $LLM_LAYERS \
  --text_len $TEXT_LEN \
  --prompt_weight $PROMPT_WEIGHT \
  --pool_type $POOL_TYPE \
  --use_fullmodel $USE_FULLMODEL \
  --use_closedllm $USE_CLOSEDLLM \
  --huggingface_token "$HUGGINGFACE_TOKEN" \
  --des "$EXP_DES" \
  `# --inverse # Add this flag ONLY if training used --inverse (i.e., inverse=True)` \
  --save_name "$SAVE_NAME" \
  --seed $SEED \
  --use_gpu True \
  --gpu 0 \
  # Add other flags used during training if necessary (e.g., --use_amp)
  # Note: --bucket_size and --n_hashes are NOT passed here as they are internal to the Reformer model code

echo "Evaluation finished for $MODEL_NAME, Pred Len $PRED_LEN."