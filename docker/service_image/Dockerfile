FROM python:3.8-slim-buster
LABEL maintainer="grac20101@gmail.com"

ARG USER_ID
ARG GROUP_ID
RUN addgroup --gid $GROUP_ID mlops
RUN useradd -r --no-log-init --uid $USER_ID -g mlops mlops

WORKDIR /app

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
COPY pyproject.toml /app/pyproject.toml
COPY poetry.lock /app/poetry.lock
COPY .env /app/.env
RUN pip install poetry
RUN poetry config virtualenvs.create false
RUN poetry install --no-dev --no-interaction --no-ansi

COPY ./src/app/inference.py /app

RUN chown mlops /app

USER mlops

CMD ["uvicorn", "inference:app", "--host", "0.0.0.0", "--port", "80"]