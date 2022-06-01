FROM docker.io/continuumio/conda-ci-linux-64-python3.7:latest

USER root

RUN apt-get update && \
    apt-get -y install rsync procps && \
    wget https://sourceforge.net/projects/lmod/files/lua-5.1.4.9.tar.bz2 && \
    tar xf lua-5.1.4.9.tar.bz2 && \
    cd lua-5.1.4.9 && \
    ./configure --prefix=/opt/apps/lua/5.1.4.9  && \
    make; make install && \
    cd /opt/apps/lua; ln -s 5.1.4.9 lua && \
    ln -s /opt/apps/lua/lua/bin/lua /usr/local/bin && \
    ln -s /opt/apps/lua/lua/bin/luac /usr/local/bin && \
    cd; wget https://sourceforge.net/projects/lmod/files/Lmod-8.2.tar.bz2 && \
    tar xf Lmod-8.2.tar.bz2 && \
    cd Lmod-8.2; ./configure --prefix=/opt/apps --with-fastTCLInterp=no && \
    make install && \
    ln -s /opt/apps/lmod/lmod/init/profile /etc/profile.d/z00_lmod.sh

ENV LMOD_ROOT=/opt/apps/lmod \
    LMOD_PKG=/opt/apps/lmod/lmod \
    LMOD_VERSION=8.2 \
    LMOD_CMD=/opt/apps/lmod/lmod/libexec/lmod \
    LMOD_DIR=/opt/apps/lmod/lmod/libexec \
    BASH_ENV=/opt/apps/lmod/lmod/init/bash

COPY . /reinventcli/

WORKDIR /reinventcli

RUN conda update -n base -c defaults conda && \
    conda env update --name=base --file=reinvent.yml && \
    chmod -R "a+rx" /reinventcli
