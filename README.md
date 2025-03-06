# Detection of bird species/behaviors in RPI5

In this repository code for performing the detection and posterior classification of species/behaviors of birds is given. The task is divided in two stages, first the detection is done and separately, the classification is carried out.

This repository is a fork of [YOLOv9: Learning What You Want to Learn Using Programmable Gradient Information](https://arxiv.org/abs/2402.13616)

## Contents

1. [Environment](#environment)
2. [Detection model (YOLOv9)](#birds-detection-model-yolov9)
3. [Classification models (CNNs)](#birds-classification-models-lightweight-cnns)
4. [License](#license)
5. [Contact](#contact)

## Environment

Download the prepared docker image:

```bash
docker pull javiro01/yolov9-birds
```

And execute it with:

```bash
bash launch_docker.sh
```

Ensure that the Visual-Wetland birds dataset is mounted in the route `/mnt/chan-twin`.

## Birds detection model (YOLOv9)

To deploy the model first dependencies must be installed executing the next:
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Then, to deploy the model use the inference instructions given in [YOLOv9 GitHub](https://github.com/WongKinYiu/yolov9). As the official code was partially customized to be
deployed in our RPI5, the script `detect_rpi.py` will be used instead of the original `detect.py`. It is recommended to use as model weights the ones available in this (link)[https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-c-converted.pt]. Note that the model downloaded is a *converted* YOLOv9 model. The customized script was prepared to be used with this variant of the YOLOv9 model series as it is the one which reports higher results.

Some image samples taken from *El Hondo* natural park are available in the folder `.assets`.


## Birds classification models (Lightweight CNNs)

The classification scripts are availabe in the folder `crops-clf`. Within this folder scripts to train and make inference from models are available. Note that these models are thought to receive as input the crop extracted from YOLO, and predict the bird's species/behavior.

### Training

- `train.py`: Train classifier to predict species. It uses as dataset the script `dataset.py`, which loads the [Visual-WetlandBirds dataset](https://github.com/3dperceptionlab/Visual-WetlandBirds).
- `train_behavs.py`: Train classifier to predict behaviors. It uses as dataset the script `dataset_behavs.py`, which loads the [Visual-WetlandBirds dataset](https://github.com/3dperceptionlab/Visual-WetlandBirds).

### Inference

- `detect_rpi.py`. Make species/behavior prediction given an image crop of a bird. Weights for the species and behavior classifier should be passed as arguments to the script.

Weights obtained are available in the next [Drive folder](https://drive.google.com/file/d/1MLh-bLT3r3GsO0DbD5Z0UX2kyTK-YQ3L/view?usp=sharing).

## License

The data from this repository is released under the [GPL-3.0 license](LICENSE).

## Contact

Please contact the author if you have any questions or requests.
- Javier Rodriguez-Juan (j.rodriguezjuan@ua.es)