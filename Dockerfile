
FROM ghcr.io/seisscoped/container-base

LABEL maintainer="Nate Groebner"

RUN cd /home/scoped \
    && git clone --branch main https://github.com/specufex/specufex

RUN cd /home/scoped/specufex \
    && conda install --file requirements.txt \
    && pip install -e . \
    && docker-clean


