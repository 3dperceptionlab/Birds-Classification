#!/bin/bash

python train.py --model efficientnet
mv ./weights/best_model.pt ./weights/efficientnet.pt
python train.py --model mobilenet
mv ./weights/best_model.pt ./weights/mobilenet.pt
python train.py --model mnasnet
mv ./weights/best_model.pt ./weights/mnasnet.pt