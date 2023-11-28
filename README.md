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

### docker run
```
docker run --name <name-of-container> -it rabetorch:<version> /bin/bash
```
(example)
```
docker run --name rabe-torch -it --rm -v /Users/rabe:/Users/rabe rabetorch:latest /bin/bash
```

## Other