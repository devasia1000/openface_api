FROM bamos/ubuntu-opencv-dlib-torch:ubuntu_14.04-opencv_2.4.11-dlib_19.0-torch_2016.07.12
MAINTAINER Brandon Amos <brandon.amos.cs@gmail.com>
# Modified by Devasia Manuel

RUN ln -s /root/torch/install/bin/* /usr/local/bin

RUN apt-get update && apt-get install -y \
    curl \
    git \
    graphicsmagick \
    libssl-dev \
    libffi-dev \
    python-dev \
    python-pip \
    python-numpy \
    python-nose \
    python-scipy \
    python-pandas \
    python-protobuf \
    python-openssl \
    wget \
    zip \
    && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

ADD . /root/openface_lastwall
RUN python -m pip install --upgrade --force pip
RUN cd ~/openface_lastwall && \
    ./models/get-models.sh && \
    pip2 install -r requirements.txt && \
    pip2 install git+https://github.com/cmusatyalab/openface

EXPOSE 80 8080

WORKDIR '/root/openface_lastwall'
CMD /usr/bin/python openface_server.py
