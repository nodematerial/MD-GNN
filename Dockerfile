# use default image of NVIDIA
FROM nvidia/cuda:11.7.1-devel-ubuntu22.04

SHELL ["/bin/bash", "-c"]

# install required modules
RUN apt-get update && apt-get upgrade -y && apt-get install -y \
    apt-utils \
    build-essential \
    libopenmpi-dev \
    libglib2.0-0 \
    libfontconfig \
    libgl1-mesa-dev \
    libxkbcommon-x11-0 \
    libdbus-glib-1-dev \
    git \
    wget \
    curl \
    cmake \
    pypy3 \
    python3 \
    python3-pip

# install yq https://github.com/mikefarah/yq
RUN wget https://github.com/mikefarah/yq/releases/latest/download/yq_linux_amd64 -O /usr/bin/yq &&\
    chmod +x /usr/bin/yq

# install LAMMPS https://github.com/lammps/lammps
# and build with GPU & MANYBODY module
RUN cd root && mkdir .local && cd .local && git clone https://github.com/lammps/lammps.git && cd lammps && mkdir build && cd build && \
    cmake -D BUILD_MPI=yes -D PKG_MANYBODY=yes -D PKG_GPU=on -D GPU_API=cuda -D GPU_ARCH=sm_75 \
    -DCMAKE_LIBRARY_PATH=/usr/local/cuda/lib64/stubs ../cmake && make -j $(nproc) && make install

RUN echo 'export PATH=~/.local/bin:$PATH' >> ~/.bashrc
RUN pypy3 -m pip install pyyaml numpy
RUN curl -sSL https://install.python-poetry.org | python3 -

COPY pyproject.toml poetry.lock /root/MD-GNN/

WORKDIR /root/MD-GNN
RUN /root/.local/bin/poetry install

