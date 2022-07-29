FROM debian:bullseye-slim

COPY VERSION /
		
RUN apt-get update
RUN apt-get install -y libtiff5-dev libtiff-dev libdeflate-dev

ARG EXEC_DIR="/opt/executables"
ARG DATA_DIR="/data"
ARG LIB_DIR="/usr/lib/x86_64-linux-gnu"

#Create folders
RUN mkdir -p ${EXEC_DIR} \
    && mkdir -p ${DATA_DIR}/images \
    && mkdir -p ${DATA_DIR}/outputs

# Copy Library
COPY local_install/lib/libblosc.so* ${LIB_DIR}/

#Copy executable
COPY nyxus ${EXEC_DIR}/
COPY nyxushie ${EXEC_DIR}/

RUN chmod +x ${EXEC_DIR}/nyxus
RUN chmod +x ${EXEC_DIR}/nyxushie

WORKDIR ${EXEC_DIR}

ENTRYPOINT ["/opt/executables/nyxus"]
