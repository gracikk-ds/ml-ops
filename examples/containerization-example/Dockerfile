FROM python:3.8-slim-buster
LABEL maintainer="grac20101@gmail.com"

ARG USER_ID
ARG GROUP_ID
RUN addgroup --gid $GROUP_ID mlops
RUN useradd -r --no-log-init --uid $USER_ID -g mlops mlops

WORKDIR /starspace
RUN mkdir ./volumes
RUN mkdir ./model

# install dependencies
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

# build StarSpace
RUN git clone https://github.com/facebookresearch/Starspace.git && \
    cd Starspace && \
    make

WORKDIR /starspace/Starspace

USER mlops

CMD ./starspace train -trainFile /starspace/volumes/${input_file_name} -model /starspace/model/${output_file_name}