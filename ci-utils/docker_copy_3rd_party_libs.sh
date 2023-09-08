#!/bin/bash -e
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
cp "$1"/lib/libblosc*.so* ./3rd_party_libs/
cp "$1"/lib/libstdc++*.so* 3rd_party_libs/
cp "$1"/lib/libtiff*.so* ./3rd_party_libs/
cp "$1"/lib/libofstd*.so* ./3rd_party_libs/
cp "$1"/lib/libdcmdata*.so* ./3rd_party_libs/
cp "$1"/lib/libdcmjpeg*.so* ./3rd_party_libs/
cp "$1"/lib/libdcmjpls*.so* ./3rd_party_libs/
cp "$1"/lib/libdcmseg*.so* ./3rd_party_libs/
cp "$1"/lib/libfmjpeg2k*.so* ./3rd_party_libs/
cp "$1"/lib/libparquet*.so* ./3rd_party_libs/
cp "$1"/lib/libarrow*.so* ./3rd_party_libs/
cp "$1"/lib/./liblz4*.so* ./3rd_party_libs/
cp "$1"/lib/./libsnappy*.so* ./3rd_party_libs/
cp "$1"/lib/./libz*.so* ./3rd_party_libs/
cp "$1"/lib/./libzstd*.so* ./3rd_party_libs/
cp "$1"/lib/./libwebp*.so* ./3rd_party_libs/
cp "$1"/lib/./liblzma*.so* ./3rd_party_libs/
cp "$1"/lib/./libLerc*.so* ./3rd_party_libs/
cp "$1"/lib/./libjpeg*.so* ./3rd_party_libs/
cp "$1"/lib/./libdeflate*.so* ./3rd_party_libs/
cp "$1"/lib/./liboflog*.so* ./3rd_party_libs/
cp "$1"/lib/./libxml2*.so* ./3rd_party_libs/
cp "$1"/lib/./libijg8*.so* ./3rd_party_libs/
cp "$1"/lib/./libijg12*.so* ./3rd_party_libs/
cp "$1"/lib/./libijg16*.so* ./3rd_party_libs/
cp "$1"/lib/./libdcmimgle*.so* ./3rd_party_libs/
cp "$1"/lib/./libdcmtkcharls*.so* ./3rd_party_libs/
cp "$1"/lib/./libdcmfg*.so* ./3rd_party_libs/
cp "$1"/lib/./libdcmiod*.so* ./3rd_party_libs/
cp "$1"/lib/./libopenjp2*.so* ./3rd_party_libs/
cp "$1"/lib/./libthrift*.so* ./3rd_party_libs/
cp "$1"/lib/./libcrypto*.so* ./3rd_party_libs/
cp "$1"/lib/./libbrotlienc*.so* ./3rd_party_libs/
cp "$1"/lib/./libbrotlidec*.so* ./3rd_party_libs/
cp "$1"/lib/./liborc*.so* ./3rd_party_libs/
cp "$1"/lib/./libglog*.so* ./3rd_party_libs/
cp "$1"/lib/./libutf8proc*.so* ./3rd_party_libs/
cp "$1"/lib/./libbz2*.so* ./3rd_party_libs/
cp "$1"/lib/./libgoogle_cloud_cpp_storage*.so* ./3rd_party_libs/
cp "$1"/lib/./libaws-cpp-sdk-identity-management*.so* ./3rd_party_libs/
cp "$1"/lib/./libaws-cpp-sdk-s3*.so* ./3rd_party_libs/
cp "$1"/lib/./libaws-cpp-sdk-core*.so* ./3rd_party_libs/
cp "$1"/lib/./libre2*.so* ./3rd_party_libs/
cp "$1"/lib/./libgoogle_cloud_cpp_common*.so* ./3rd_party_libs/
cp "$1"/lib/./libabsl_time*.so* ./3rd_party_libs/
cp "$1"/lib/./libabsl_time_zone*.so* ./3rd_party_libs/
cp "$1"/lib/./libaws-crt-cpp*.so* ./3rd_party_libs/
cp "$1"/lib/././libsharpyuv*.so* ./3rd_party_libs/
cp "$1"/lib/././libiconv*.so* ./3rd_party_libs/
cp "$1"/lib/././libicui18n*.so* ./3rd_party_libs/
cp "$1"/lib/././libicuuc*.so* ./3rd_party_libs/
cp "$1"/lib/././libicudata*.so* ./3rd_party_libs/
cp "$1"/lib/././libssl*.so* ./3rd_party_libs/
cp "$1"/lib/././libbrotlicommon*.so* ./3rd_party_libs/
cp "$1"/lib/././libprotobuf*.so* ./3rd_party_libs/
cp "$1"/lib/././libgflags*.so* ./3rd_party_libs/
cp "$1"/lib/././libgoogle_cloud_cpp_rest_internal*.so* ./3rd_party_libs/
cp "$1"/lib/././libcrc32c*.so* ./3rd_party_libs/
cp "$1"/lib/././libcurl*.so* ./3rd_party_libs/
cp "$1"/lib/././libabsl_crc32c*.so* ./3rd_party_libs/
cp "$1"/lib/././libabsl_str_format_internal*.so* ./3rd_party_libs/
cp "$1"/lib/././libabsl_strings*.so* ./3rd_party_libs/
cp "$1"/lib/././libabsl_strings_internal*.so* ./3rd_party_libs/
cp "$1"/lib/././libaws-cpp-sdk-cognito-identity*.so* ./3rd_party_libs/
cp "$1"/lib/././libaws-cpp-sdk-sts*.so* ./3rd_party_libs/
cp "$1"/lib/././libaws-c-event-stream*.so* ./3rd_party_libs/
cp "$1"/lib/././libaws-checksums*.so* ./3rd_party_libs/
cp "$1"/lib/././libaws-c-common*.so* ./3rd_party_libs/
cp "$1"/lib/././libabsl_int128*.so* ./3rd_party_libs/
cp "$1"/lib/././libabsl_base*.so* ./3rd_party_libs/
cp "$1"/lib/././libabsl_raw_logging_internal*.so* ./3rd_party_libs/
cp "$1"/lib/././libaws-c-mqtt*.so* ./3rd_party_libs/
cp "$1"/lib/././libaws-c-s3*.so* ./3rd_party_libs/
cp "$1"/lib/././libaws-c-auth*.so* ./3rd_party_libs/
cp "$1"/lib/././libaws-c-http*.so* ./3rd_party_libs/
cp "$1"/lib/././libaws-c-io*.so* ./3rd_party_libs/
cp "$1"/lib/././libaws-c-cal*.so* ./3rd_party_libs/
cp "$1"/lib/././libaws-c-sdkutils*.so* ./3rd_party_libs/
cp "$1"/lib/./././libnghttp2*.so* ./3rd_party_libs/
cp "$1"/lib/./././libssh2*.so* ./3rd_party_libs/
cp "$1"/lib/./././libgssapi_krb5*.so* ./3rd_party_libs/
cp "$1"/lib/./././libabsl_crc_internal*.so* ./3rd_party_libs/
cp "$1"/lib/./././libabsl_spinlock_wait*.so* ./3rd_party_libs/
cp "$1"/lib/./././libaws-c-compression*.so* ./3rd_party_libs/
cp "$1"/lib/./././libs2n*.so* ./3rd_party_libs/
cp "$1"/lib/././././libkrb5*.so* ./3rd_party_libs/
cp "$1"/lib/././././libk5crypto*.so* ./3rd_party_libs/
cp "$1"/lib/././././libcom_err*.so* ./3rd_party_libs/
cp "$1"/lib/././././libkrb5support*.so* ./3rd_party_libs/
cp "$1"/lib/././././libkeyutils*.so* ./3rd_party_libs/
