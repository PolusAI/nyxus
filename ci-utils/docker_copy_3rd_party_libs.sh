#/bin/bash
#
# Copy necessary libraries from build environment to Docker staging dir
#
# Usage: $bash docker_copy_3rd_party_libs.sh $LIB_LOCATION_ROOT
# for conda environment, this will be $CONDA_PREFIX
#
if [ -z "$1" ]
then
      echo "No path to the library location provided"
      echo "Usage: \$bash docker_copy_3rd_party_libs.sh \$LIB_LOCATION_ROOT "
      echo "       where \$LIB_LOCATION_ROOT points to source location of dependency libraries"
      exit
fi


mkdir -p 3rd_party_libs
cp $1/lib/libblosc*.so* ./3rd_party_libs/
cp $1/lib/./liblz4*.so* ./3rd_party_libs/
cp $1/lib/./libsnappy*.so* ./3rd_party_libs/
cp $1/lib/./libz*.so* ./3rd_party_libs/
cp $1/lib/./libzstd*.so* ./3rd_party_libs/
cp $1/lib/libtiff*.so* ./3rd_party_libs/
cp $1/lib/./libwebp*.so* ./3rd_party_libs/
cp $1/lib/./liblzma*.so* ./3rd_party_libs/
cp $1/lib/./libLerc*.so* ./3rd_party_libs/
cp $1/lib/./libjpeg*.so* ./3rd_party_libs/
cp $1/lib/./libdeflate*.so* ./3rd_party_libs/
cp $1/lib/./libdcm*.so* ./3rd_party_libs/