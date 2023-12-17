# rabetorch

## Overview
This repository is created for traininig program of pytorch.

## Envirionment
### docker build
Please make sure to change dir to root dir of this repository.
```
docker build -t rabetorch:<version>  -f ./docker/Dockerfile .
```
(example)
```
docker build -t rabetorch -f ./docker/Dockerfile .
```

### docker build for develop
```
docker build --build-arg USER_ID=$(id -u [username]) --build-arg GROUP_ID=$(id -g [username]) --build-arg YOUR_USR=$(id -u [username]) -t rabetorch:<version>  -f ./docker/DockerfileDev .
```
(example)
```
docker build --build-arg USER_ID=$(id -u rabe) --build-arg GROUP_ID=$(id -g rabe) --build-arg YOUR_USER=rabe -t rabetorch -f ./docker/DockerfileDev .
```

### docker run
```
docker run --name <name-of-container> -it rabetorch:<version> /bin/bash
```
(example)
```
docker run --name rabe-torch -it --rm -v /Users/rabe:/Users/rabe rabetorch:latest /bin/bash
```

## Train
You can run train script by command below;
```
python scripts/train.py YOUR_CONFIG_NAME OVERRIDE_CONFIG
```
here, `YOUR_CONFIG_NAME` is file name of config you want to use.
`YOUR_CONFIG_NAME` must be written as relative path from `./configs`.
Also, you can override config from argument. Please refer example for detail.

(example)
```
python scripts/train.py basic_classifier SOLVER.MAX_EPOCH 12
```

## Other
### Data main strategy
We downloaded VOC dataset as annotation data but we only support COCO format.
To use VOC dataset in this repository, we implemented VOC2COCO converter.
See detail in `./scripts/voc2coco.py`