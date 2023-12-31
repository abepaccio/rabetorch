# syntax=docker/dockerfile:1

FROM ubuntu:22.04

# initiate ubuntu env
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get upgrade -y \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y tzdata \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# install apps
RUN apt-get update \
    && apt-get install -y \
        git \
        curl \
        wget \
        vim \
        python3.10 \
        python3-pip \
    && apt-get clean all

RUN ln -s /usr/bin/python3 /usr/bin/python

# create a developer user as a member of developer group
ARG USERNAME=developer
ARG GROUPNAME=developer
ARG UID=1000
ARG GID=1000
RUN groupadd -g $GID $GROUPNAME && \
    useradd -m -s /bin/bash -u $UID -g $GID $USERNAME

# create work dir and login as developer
WORKDIR /app

# install python module
COPY --chown=$USERNAME:$GROUPNAME ./requirements.txt ./requirements.txt
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt


# copy repository files
COPY --chown=$USERNAME:$GROUPNAME ./ ./

# install and initialize pre-commit as the developer user
USER $USERNAME
RUN pre-commit install && pre-commit install -t pre-push

ENV PYTHONPATH ./
ENV USER_NAME $USERNAME
