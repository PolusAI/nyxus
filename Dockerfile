FROM labshare/polus-bfio-util:2.0.4-slim-buster

COPY VERSION /
		
RUN apt-get update
RUN apt-get install -y libtiff5-dev

ARG EXEC_DIR="/opt/executables"
ARG DATA_DIR="/data"

#Create folders
RUN mkdir -p ${EXEC_DIR} \
    && mkdir -p ${DATA_DIR}/images \
    && mkdir -p ${DATA_DIR}/outputs

#Copy executable
COPY build-4-linux ${EXEC_DIR}/
RUN chmod +x ${EXEC_DIR}/nyxus

WORKDIR ${EXEC_DIR}

ENTRYPOINT ["/opt/executables/nyxus"]