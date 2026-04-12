FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

# 対話型インストールを無効化
ENV DEBIAN_FRONTEND=noninteractive
# タイムゾーン設定
ENV TZ=Asia/Tokyo

RUN apt-get update && apt-get install -y \
    wget \
    git \
    python3 \
    python3-pip \
    # matplotlib GUI用ライブラリ ---
    python3-tk \
    # -----------------------------
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY requirements.txt /tmp/

RUN pip3 install -U pip
RUN pip3 install -U torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128

RUN pip3 install --no-cache-dir -r /tmp/requirements.txt

RUN rm -rf /tmp/*