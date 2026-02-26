# -------- Base image (Debian 12, ARM64) --------
FROM debian:12

ENV DEBIAN_FRONTEND=noninteractive

# -------- System dependencies --------
RUN apt update && apt upgrade -y && \
    apt install -y \
        wget \
        curl \
        ca-certificates \
        bzip2 \
        git \
        libgl1 \
        libglib2.0-0 \
        && rm -rf /var/lib/apt/lists/*

# -------- Install Miniforge (Conda for ARM64) --------
WORKDIR /opt

RUN wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh && \
    chmod +x Miniforge3-Linux-aarch64.sh && \
    ./Miniforge3-Linux-aarch64.sh -b -p /opt/miniforge3 && \
    rm Miniforge3-Linux-aarch64.sh

# -------- Set Conda path --------
ENV PATH="/opt/miniforge3/bin:$PATH"

# -------- Configure Conda --------
RUN conda install -y conda-libmamba-solver && \
    conda config --set solver libmamba && \
    conda init bash && \
    conda clean -afy

# -------- Create YOLOE environment --------
RUN conda create -y -n ultralytics-env python=3.11

# -------- Install Ultralytics + PyTorch (CPU) --------
RUN conda run -n ultralytics-env conda install -y -c pytorch -c conda-forge \
    pytorch torchvision cpuonly ultralytics && \
    conda run -n ultralytics-env pip install kafka-python opencv-python-headless fastapi uvicorn python-multipart && \
    conda clean -afy

# -------- App directory --------
WORKDIR /app

# -------- Copy YOLOE code --------
COPY yoloe_server.py /app/yoloe_server.py

# -------- Expose Port --------
EXPOSE 5000

# -------- Run YOLOE --------
CMD ["conda", "run", "--no-capture-output", "-n", "ultralytics-env", "python", "yoloe_server.py"]
# RUN echo "conda activate ultralytics-env" >> ~/.bashrc
# CMD ["/bin/bash"]