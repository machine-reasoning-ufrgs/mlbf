FROM continuumio/miniconda3:latest

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

ARG PYTHON_VERSION=3.8

RUN apt-get update && apt-get -y upgrade && \
    apt-get install -y --no-install-recommends \
    wget \
    vim \
    sudo \
    libgomp1 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

RUN conda update conda -y && \
    adduser -u 5678 --disabled-password --gecos "SAT user" sat && \
    usermod -a -G sudo sat && \
    usermod -a -G adm sat && \
    echo "%sudo ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

USER sat
WORKDIR /home/sat

COPY environment.yml .

RUN conda env create -f environment.yml && \
    conda init bash && \
    echo "conda activate mlbf" >> ~/.bashrc && \
    . ~/.bashrc

#ENTRYPOINT ["/home/sat/.conda/envs/mlbf/bin/python"]
CMD ["/bin/bash"]
