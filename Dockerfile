FROM nvidia/cuda:12.6.3-runtime-ubuntu22.04

COPY .version /VERSION

ARG DATA_DIR="/data"

#Create folders
RUN mkdir -p /opt/executables \
    && mkdir -p ${DATA_DIR}/images \
    && mkdir -p ${DATA_DIR}/outputs

# The executable's DT_RPATH is $ORIGIN/lib, so its libraries sit next to it and
# the base image's system libraries stay as the base image ships them
COPY docker_staging/lib/ /opt/executables/lib/

#Copy executable
COPY docker_staging/nyxus /opt/executables/

RUN chmod +x /opt/executables/nyxus

WORKDIR /opt/executables

ENTRYPOINT ["/opt/executables/nyxus"]
