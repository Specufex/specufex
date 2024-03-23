
FROM ghcr.io/seisscoped/container-base


LABEL maintainer="Nate Groebner"

RUN git clone https://github.com/specufex/specufex.git \
    && cd "${HOME}"/specufex \
    && pip install . \
    && pip install scikit-learn seaborn tqdm h5py numexpr \
    && docker-clean

USER ${NB_UID}

WORKDIR "${HOME}"

EXPOSE 8888