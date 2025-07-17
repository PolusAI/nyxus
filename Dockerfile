FROM nvidia/cuda:12.9.1-runtime-ubuntu22.04

COPY VERSION /

ARG LIB_LOCATION="3rd_party_libs"
ARG EXEC_DIR="/opt/executables"
ARG DATA_DIR="/data"
ARG LIB_DIR="/usr/lib/x86_64-linux-gnu"

#Create folders
RUN mkdir -p ${EXEC_DIR} \
    && mkdir -p ${DATA_DIR}/images \
    && mkdir -p ${DATA_DIR}/outputs

# Copy Library
COPY ${LIB_LOCATION}/*.so* ${LIB_DIR}/

#Copy executable
COPY nyxus ${EXEC_DIR}/

RUN chmod +x ${EXEC_DIR}/nyxus

WORKDIR ${EXEC_DIR}

ENTRYPOINT ["/opt/executables/nyxus"]
