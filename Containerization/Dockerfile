FROM python:3.8-slim-buster
LABEL maintainer="grac20101@gmail.com"

WORKDIR /starspace

## install dependencies
RUN apt-get update && apt-get install -y \
	autoconf \
    automake \
	libtool \
	build-essential \
	unzip \
	git \
	wget

# install Boost library
RUN wget https://sourceforge.net/projects/boost/files/boost/1.63.0/boost_1_63_0.zip && \
    unzip boost_1_63_0.zip && \
    mv boost_1_63_0 /usr/local/bin

ARG USER_ID
ARG GROUP_ID

RUN addgroup --gid $GROUP_ID mlops
RUN useradd -rm -d /starspace --no-log-init --uid $USER_ID -g mlops mlops
RUN chown mlops /starspace
RUN chgrp mlops /starspace
USER mlops

RUN mkdir ./volumes
RUN mkdir ./model

# build StarSpace
RUN git clone https://github.com/facebookresearch/Starspace.git && \
    cd Starspace && \
    make

WORKDIR /starspace/Starspace

CMD ./starspace train -trainFile /starspace/volumes/${input_file_name} -model /starspace/model/${output_file_name}