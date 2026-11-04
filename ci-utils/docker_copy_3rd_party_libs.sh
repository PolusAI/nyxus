#!/bin/bash
#
# Stage the nyxus executable and the libraries it needs from the build
# environment into docker_staging/ for the Docker image
#
# Usage: $bash docker_copy_3rd_party_libs.sh $LIB_LOCATION_ROOT [$NYXUS_EXECUTABLE [$BASE_IMAGE_LIBS]]
# for conda environment, $LIB_LOCATION_ROOT will be $CONDA_PREFIX
#
# The library set is the transitive closure of the executable's shared-library
# dependencies (ldd) that ldd finds in $LIB_LOCATION_ROOT/lib, less the sonames
# listed in the file $BASE_IMAGE_LIBS, one per line, which the base image
# provides (the CUDA runtime of the nvidia/cuda image), which must be no older
# than the env's (the list also names the versioned files). System libraries resolve
# outside the env and come from the base image too. A library that resolves
# elsewhere in the env stops the script, since the image would not have it.
#
# docker_staging/nyxus is a copy of the executable with DT_RPATH $ORIGIN/lib,
# and docker_staging/lib holds the libraries, so the image keeps the two side by
# side and leaves the base image's system library directories as they are.
#
if [ -z "$1" ]
then
      echo "No path to the library location provided"
      echo "Usage: \$bash docker_copy_3rd_party_libs.sh \$LIB_LOCATION_ROOT [\$NYXUS_EXECUTABLE [\$BASE_IMAGE_LIBS]]"
      echo "       where \$LIB_LOCATION_ROOT points to source location of dependency libraries"
      exit 1
fi

if [ ! -d "$1/lib" ]
then
      echo "No lib directory in $1"
      exit 1
fi

ENV_ROOT=$(realpath "$1") || exit 1
LIB_ROOT=$(realpath "$1/lib") || exit 1
EXECUTABLE=${2:-./nyxus}
BASE_LIBS=${3:-}
STAGING=docker_staging

if [ ! -x "$EXECUTABLE" ]
then
      echo "Executable $EXECUTABLE not found"
      exit 1
fi

if [ -n "$BASE_LIBS" ] && [ ! -s "$BASE_LIBS" ]
then
      echo "Base image library list $BASE_LIBS is missing or empty"
      exit 1
fi

if ! command -v patchelf > /dev/null
then
      echo "patchelf not found"
      exit 1
fi

if ! LDD_OUT=$(LD_LIBRARY_PATH=$LIB_ROOT ldd "$EXECUTABLE")
then
      echo "ldd failed on $EXECUTABLE:"
      echo "$LDD_OUT"
      exit 1
fi

if echo "$LDD_OUT" | grep -q "not found"
then
      echo "Unresolved shared libraries:"
      echo "$LDD_OUT" | grep "not found"
      exit 1
fi

rm -rf "$STAGING"
mkdir -p "$STAGING/lib"

# Each line is "<soname> => <path> (<address>)"; the copy is stored under the
# soname, which is what the dynamic loader looks up inside the image. A library
# belongs to the env when the directory ldd found it in is $LIB_ROOT or one of
# its subdirectories, and is staged flat next to the others; the file
# the soname links to may live elsewhere in the env (for example targets/).
COUNT=0
while read -r SONAME LIB_PATH
do
      # A soname the base image provides is left to it, provided the base
      # image's file is no older than the one the executable was built against:
      # the CUDA runtime is compatible forwards, not backwards.
      if [ -n "$BASE_LIBS" ] && grep -qxF "$SONAME" "$BASE_LIBS"
      then
            ENV_FILE=$(basename "$(realpath "$LIB_PATH")")
            BASE_FILE=$(awk -v p="$SONAME." 'index($0, p) == 1' "$BASE_LIBS" | sort -V | tail -n 1)
            if [ -z "$BASE_FILE" ]
            then
                  echo "$SONAME: the base image list names no versioned $SONAME.* file to compare with the env's $ENV_FILE"
                  exit 1
            fi
            if [ "$(printf '%s\n%s\n' "$ENV_FILE" "$BASE_FILE" | sort -V | tail -n 1)" != "$BASE_FILE" ]
            then
                  echo "$SONAME: the base image has $BASE_FILE, older than the $ENV_FILE the executable was built against"
                  exit 1
            fi
            continue
      fi
      case "$(realpath "$(dirname "$LIB_PATH")")" in
            "$LIB_ROOT"|"$LIB_ROOT"/*)
                  cp -L "$LIB_PATH" "$STAGING/lib/$SONAME" || exit 1
                  COUNT=$((COUNT + 1))
                  ;;
            "$ENV_ROOT"/*)
                  echo "$SONAME resolves inside the env but outside $LIB_ROOT: $LIB_PATH"
                  exit 1
                  ;;
      esac
done < <(echo "$LDD_OUT" | awk '$2 == "=>" && $3 ~ /^\// { print $1, $3 }')

if [ "$COUNT" -eq 0 ]
then
      echo "No libraries of $EXECUTABLE resolve inside $LIB_ROOT"
      exit 1
fi

# DT_RPATH rather than DT_RUNPATH: LD_LIBRARY_PATH does not override it, and the
# loader also searches it for the dependencies of a library that has no
# DT_RUNPATH of its own. conda-forge libraries carry $ORIGIN-relative run paths,
# which resolve inside the same directory; the image's smoke test checks that
# every staged library is the copy the loader takes.
cp -L "$EXECUTABLE" "$STAGING/nyxus" || exit 1
patchelf --force-rpath --set-rpath '$ORIGIN/lib' "$STAGING/nyxus" || exit 1

echo "Copied $COUNT libraries from $LIB_ROOT to $STAGING/lib"
