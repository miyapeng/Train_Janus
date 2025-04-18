# Copyright 2025 PKU-Alignment Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# Initialize variables
MODEL_NAME_OR_PATH="../models/Janus-Pro-1B"
TRAIN_DATASETS="../projects/janus/example/supervised/text_image_to_image"
TRAIN_DATA_FILE="train_tokenized.pt"
OUTPUT_DIR="output/janus_sft_text_image_to_image"
JANUS_REPO_PATH="../Align_Anything_Janus"

export PYTHONPATH=$PYTHONPATH:$JANUS_REPO_PATH
export WANDB_API_KEY="5f56adc141de7e053c2d4d47afebebff6d3d816b"
export WANDB_MODE=online

# Source the setup script
source ./setup.sh
# Execute deepspeed command
deepspeed \
    --master_port ${MASTER_PORT} \
    --module align_anything.trainers.janus.sft \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --train_datasets ${TRAIN_DATASETS} \
    --train_data_files ${TRAIN_DATA_FILE} \
    --train_split train \
    --learning_rate 1e-6 \
    --epochs 3 \
    --lr_scheduler_type cosine \
    --output_dir ${OUTPUT_DIR}
