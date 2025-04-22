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
INPUT_PATH="example/supervised/text_image_to_image/train.json"
OUTPUT_PATH="example/supervised/text_image_to_image/train_tokenized.pt"
MODEL_PATH="/fs-computility/ai-shen/fanyuyu/Train_Janus/model/Janus-Pro-1B"
CACHE_DIR="cache"
JANUS_REPO_PATH="../Align_Anything_Janus"
mkdir -p $CACHE_DIR
NUM_PROCESSES=1
NUM_GPUS=1

export PYTHONPATH=$PYTHONPATH:"$JANUS_REPO_PATH"

python supervised_text_image_to_image.py \
    --input_path $INPUT_PATH \
    --output_path $OUTPUT_PATH \
    --model_path $MODEL_PATH \
    --cache_dir $CACHE_DIR \
    --num_processes $NUM_PROCESSES \
    --num_gpus $NUM_GPUS
