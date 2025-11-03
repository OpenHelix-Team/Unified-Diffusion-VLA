# World Model

## Post-train dataset processing
```shell
# here we provide the post-train dataset processing script
cd tools/process/post_train

# take oxe toto for example, remember to change the oxe path and output path
python oxe_toto.py
```

## World Model Training
```shell
# train the world model
bash scripts/pretrain/train_video_1node.sh
```

## Inference example
```shell
python models/inference/inference_vision.py
```
