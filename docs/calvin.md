# CALVIN Benchmark

[CALVIN](https://github.com/mees/calvin) is a benchmark for evaluating vision-language models in robotic long-horizon manipulation tasks. 

![](imgs/calvin.png)



## Environment Setup
We follow the [RoboVLMs](https://github.com/Robot-VLAs/RoboVLMs) repository for environment setup. This setup is only for evaluation. The following steps are required to set up the environment:

```shell
# Install dependencies
cd reference/RoboVLMs

# This will install the required environment and download the calvin dataset.
bash scripts/setup_calvin.sh

# Only for rendering environment.
bash scripts/setup_calvin_vla.sh

# Check if the environment is set up correctly
python eval/calvin/env_test.py
```

## Dataset Preparation
```shell
# 1. process the dataset
python tools/process/calvin_process.py

# 2. extract the vq tokens, need to change the dataset & output path
bash scripts/tokenizer/extract_vq_emu3.sh 

# 3. pickle generation for training
python tools/pickle_gen/pickle_generation_calvin.py
```

## Model Training

### FAST Tokenizer
You can fit the FAST tokenizer on the corresponding dataset. Also, you can adjust the scale in tokenizer for more fine-grained tokenization.
```shell
python tools/action_tokenizer/fit_fast.py
```

```shell
bash scripts/simulator/calvin/train_calvin_abcd_video_i-ia_bid_mi.sh 
```

## Model Evaluation
```shell
cd reference/RoboVLMs

# 8 GPUs inference
bash scripts/calvin/run_eval_calvin_univla_i2ia_dis.sh

# above command will generate the 8 results (if use 8 GPUs) in the `results` folder, calculate the final average score
python tools/evaluation/calvin_score.py
```