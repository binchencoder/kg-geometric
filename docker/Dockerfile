# =====================================================================
# TKGL-Smallpedia 时序知识图谱链接预测 —— 运行镜像
# 仅包含代码与运行环境，数据集通过 volume 挂载。
#
# 构建（构建上下文 = 项目根目录）：
#   CPU（默认）:
#     docker build -t tkgl-smallpedia:cpu -f docker/Dockerfile .
#   GPU（需 nvidia-container-toolkit）:
#     docker build -t tkgl-smallpedia:gpu -f docker/Dockerfile . \
#       --build-arg BASE_IMAGE=pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime \
#       --build-arg INSTALL_TORCH=0
#
# 运行：
#   docker run --rm -v "$PWD/data:/app/data" -v "$PWD/trained_models:/app/trained_models" \
#     tkgl-smallpedia:cpu --mode infer
# =====================================================================

# 基础镜像：默认 CPU 版 slim；GPU 时传 pytorch CUDA 官方镜像
ARG BASE_IMAGE=python:3.10-slim
FROM ${BASE_IMAGE}

# GPU 基础镜像（pytorch/pytorch）已自带 torch + numpy，无需再装；
# CPU 基础镜像则需从 PyTorch CPU 源安装 torch。
ARG INSTALL_TORCH=1
ARG TORCH_VERSION=2.2.2

# 避免交互式安装卡住
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# 运行所需系统库（torch CPU 版依赖 libgomp 等）
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ---------- 依赖安装 ----------
# 脚本自包含（自带 TGAT 实现），实际只依赖 numpy + torch，
# 不需要 torch-geometric / elasticsearch / PyYAML。
COPY docker/requirements.docker.txt /app/requirements.docker.txt
RUN pip install --no-cache-dir -r /app/requirements.docker.txt

# 仅当基础镜像不含 torch 时才安装（CPU 场景）；GPU 镜像跳过
RUN if [ "$INSTALL_TORCH" = "1" ]; then \
      pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu \
        torch==${TORCH_VERSION}; \
    fi

# ---------- 拷贝脚本运行所需代码（不含数据 / 训练产物）----------
# 推理/训练只需 demo/ 下的脚本 + data/tkgl-smallpedia/ 的纯 Python 加载器；
# 数据集与训练产物通过 -v 挂载，不进镜像。
COPY demo/ /app/demo/
COPY data/tkgl-smallpedia/ /app/data/tkgl-smallpedia/

# 默认数据目录（容器内相对路径，与脚本默认 --data-dir 一致）。
# 运行时请用 -v 把宿主机数据集挂载到 /app/data，例如：
#   -v /path/to/kg-geometric/data:/app/data
VOLUME ["/app/data", "/app/trained_models"]

# 入口：直接运行该脚本，所有参数透传（train / infer / --head ... 等）
ENTRYPOINT ["python", "demo/tkgl_smallpedia_tkg.py"]

# 默认动作：推理（若已挂载训练好的模型）；训练时覆盖为 --mode train
CMD ["--mode", "infer"]
