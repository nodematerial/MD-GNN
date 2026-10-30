FROM nvidia/cuda:13.4.3-devel-ubuntu26.04

# GPU architecture for the LAMMPS GPU package (e.g. sm_75 for Turing, sm_86 for Ampere)
ARG GPU_ARCH=sm_75

SHELL ["/bin/bash", "-c"]
ENV DEBIAN_FRONTEND=noninteractive

# libGL, libEGL, fontconfig, xkbcommon and dbus are required by the OVITO Python module
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    cmake \
    curl \
    git \
    libdbus-1-3 \
    libegl1 \
    libfontconfig1 \
    libgl1 \
    libglib2.0-0t64 \
    libopengl0 \
    libopenmpi-dev \
    libxkbcommon-x11-0 \
    pypy3 \
    pypy3-venv \
    wget \
    && rm -rf /var/lib/apt/lists/*

# install yq https://github.com/mikefarah/yq
RUN wget -q https://github.com/mikefarah/yq/releases/latest/download/yq_linux_amd64 -O /usr/bin/yq && \
    chmod +x /usr/bin/yq

# install LAMMPS https://github.com/lammps/lammps
# and build with GPU & MANYBODY module
RUN mkdir -p /root/.local && cd /root/.local && \
    git clone --depth 1 --branch stable https://github.com/lammps/lammps.git && \
    cd lammps && mkdir build && cd build && \
    cmake -D BUILD_MPI=yes -D PKG_MANYBODY=yes -D PKG_GPU=on -D GPU_API=cuda -D GPU_ARCH=${GPU_ARCH} \
    -D CMAKE_LIBRARY_PATH=/usr/local/cuda/lib64/stubs ../cmake && \
    make -j $(nproc) && make install

# makegraph.py runs on PyPy for speed
RUN pypy3 -m venv /opt/pypy && /opt/pypy/bin/pip install --no-cache-dir pyyaml numpy

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
RUN uv python install 3.14 && uv tool install --python 3.14 poetry

ENV PATH=/opt/pypy/bin:/root/.local/bin:$PATH

COPY pyproject.toml poetry.lock /root/MD-GNN/

WORKDIR /root/MD-GNN
RUN poetry env use "$(uv python find 3.14)" && poetry install
