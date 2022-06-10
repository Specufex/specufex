#FROM jupyter/scipy-notebook:lab-3.4.2

# build obspy
FROM python:3.10-slim-bullseye as builder
RUN apt-get update
RUN apt-get install -y --no-install-recommends gcc build-essential

RUN pip wheel --wheel-dir=/opt/wheels --no-deps obspy

# now build the working image
# base image is a custom implementation of jupyter/scipy-notebook
FROM public.ecr.aws/c8c6r3q4/jupyterlab-slim:v0.1

COPY --from=builder /opt/wheels /opt/wheels

USER root
WORKDIR /opt
COPY . /opt/specufex
RUN pip install /opt/specufex lxml && \
    pip install --no-index --find-links=/opt/wheels obspy

USER ${NB_UID}
WORKDIR "${HOME}"

