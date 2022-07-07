
FROM ghcr.io/seisscoped/container-base

LABEL maintainer="Nate Groebner"

RUN cd /home/scoped \
    && git clone --branch feature-geysers-tutorial https://github.com/ngroebner/specufex

RUN cd /home/scoped/specufex \
    && conda install --file requirements.txt \
    && pip install -e . \
    && docker-clean \
    && mv ./tutorials "${HOME}"

USER ${NB_UID}
WORKDIR "${HOME}"

EXPOSE 8888
