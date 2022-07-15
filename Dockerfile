
FROM ghcr.io/seisscoped/container-base

LABEL maintainer="Nate Groebner"

COPY requirements.txt .

RUN pip install git+https://github.com/specufex/specufex.git \
    && pip install -r requirements.txt \
    && mamba install scikit-learn \
    && rm requirements.txt \
    && docker-clean

USER ${NB_UID}
WORKDIR "${HOME}"

EXPOSE 8888
