#!/bin/bash

docker run --gpus '"device=0"' --rm -it \
        --name birds-classification \
        --volume "/mnt/chan-twin/:/data/:ro" \
        --volume "$PWD/:/workspace/:rw" \
        --shm-size=16gb \
        --memory=24gb \
        javiro01/yolov9-birds bash
