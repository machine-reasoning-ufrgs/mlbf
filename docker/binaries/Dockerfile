FROM ubuntu:20.04 as builder

ENV DOCKER_BUILDKIT=1
ENV TZ=America/Sao_Paulo

RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone && \
    apt-get update && \
    apt-get install --no-install-recommends -y software-properties-common \
    libboost-program-options-dev \
    libm4ri-dev \
    gcc \
    g++ \
    make \
    cmake \
    git \
    zlib1g-dev \
    wget \
    make \
    libgmp-dev \
    libmpfr-dev \
    libmpc-dev && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/github

# MIS
RUN git clone https://github.com/meelgroup/mis && \
    cd mis && \
    git submodule update --init && \
    make static && \
    cd ..

# CMS
RUN wget https://github.com/msoos/cryptominisat/archive/5.8.0.tar.gz && \
    tar -xvf 5.8.0.tar.gz && \
    cd cryptominisat-5.8.0 && \
    mkdir build && cd build && \
    cmake -DSTATICCOMPILE=ON .. && \
    make -j6 && make install && \
    cd ../..

# ApproxMC
RUN wget https://github.com/meelgroup/approxmc/archive/4.0.2.tar.gz && \
    tar -xvf 4.0.2.tar.gz && \
    cd  approxmc-4.0.2 && \
    mkdir build && cd build && \
    cmake -DSTATICCOMPILE=ON .. && \
    make -j6 && make install && \
    cd ../..

# Unigen3
RUN git clone https://github.com/meelgroup/unigen.git && \
    cd unigen && \
    mkdir build && cd build && \
    cmake -DSTATICCOMPILE=ON .. && \
    make -j6 && make install

# Ganak
RUN git clone https://github.com/meelgroup/ganak.git && \
    cd ganak && \
    mkdir build && cd build && \
    cmake .. && \
    make && \
    cp ganak ../bin/ && \
    cp src/libganak.so.1.0 ../bin/ && \
    cp src/clhash/libclhash.so ../bin/ && \
    cp src/component_types/libcomponent_types.so ../bin/


FROM alpine:latest

WORKDIR /opt/binaries

COPY --from=builder /opt/github/mis/mis.py .
COPY --from=builder /opt/github/mis/togmus .
COPY --from=builder /opt/github/mis/muser2-dir/src/tools/muser2/muser2 .
COPY --from=builder /usr/local/bin/unigen ./unigen3
COPY --from=builder /opt/github/ganak/bin/arjun .
COPY --from=builder /opt/github/ganak/bin/doalarm .
COPY --from=builder /opt/github/ganak/bin/ganak .
COPY --from=builder /opt/github/ganak/bin/lib* /opt/binaries/

CMD ["/bin/sh"]
