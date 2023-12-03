# rabetorch

## Overview
This repository is created for traininig program of pytorch.
We downroaded VOC dataset as annotation data but we only support COCO format.
To use VOC dataset in this repository, we implemented VOC2COCO converter.
See detail in `./scripts/voc2coco.py`

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

## Other