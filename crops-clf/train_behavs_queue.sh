#!/bin/bash

python train_behavs.py --model efficientnet
mv ./weights/best_model_behavs.pt ./weights/efficientnet-behavs.pt
python train_behavs.py --model mobilenet
mv ./weights/best_model_behavs.pt ./weights/mobilenet-behavs.pt
python train_behavs.py --model mnasnet
mv ./weights/best_model_behavs.pt ./weights/mnasnet-behavs.pt