FROM ghcr.io/seisscoped/container-base

LABEL maintainer="Nate Groebner"

USER ${NB_UID}
WORKDIR "${HOME}"

RUN git clone https://github.com/specufex/specufex.git \
    && cd "${HOME}"/specufex \
    && python -m pip install . \
    && docker-clean

EXPOSE 8888
