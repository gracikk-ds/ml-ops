FROM python:3.8-slim-buster
LABEL maintainer="grac20101@gmail.com"

ARG USER_ID
ARG GROUP_ID
RUN addgroup --gid $GROUP_ID mlops
RUN useradd -r --no-log-init --uid $USER_ID -g mlops mlops

WORKDIR /training
RUN mkdir ./volumes

# install dependencies
RUN apt-get update && apt-get install -y \
	autoconf \
    automake \
	libtool \
	build-essential \
	unzip \
	git \
	wget

# copy dependencies
COPY pyproject.toml /training/pyproject.toml
RUN pip install poetry
RUN poetry config virtualenvs.create false
RUN poetry install --no-dev

COPY ./src/models/train_model.py /training
COPY ./src/models/validate_model.py /training

ENTRYPOINT tail -f /dev/null