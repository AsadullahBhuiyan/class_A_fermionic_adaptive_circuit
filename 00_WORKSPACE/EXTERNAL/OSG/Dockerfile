FROM hub.opensciencegrid.org/htc/rocky:9-cuda-12.6.0

RUN dnf install -y python3 python3-pip git && dnf clean all

RUN pip install --no-cache-dir \
    torch==2.8.0 \
    numpy \
    scipy \
    matplotlib \
    h5py \
    pfapack \
    tqdm \
    pandas \
    polars \
    seaborn \
    fastparquet \
    pyarrow \
    opt-einsum \
    joblib \
    cloudpickle \
    networkx \
    sympy \
    lark \
    mpmath
