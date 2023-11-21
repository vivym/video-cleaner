#!/bin/bash

set -e
set -u
SCRIPTROOT="$( cd "$(dirname "$0")" ; pwd -P )"
cd "${SCRIPTROOT}/.."

DOCKER_BUILDKIT=1 docker build --network host \
                               --build-arg USER=$USER \
                               --build-arg UID=$(id -u) \
                               --build-arg GID=$(id -g) \
                               --build-arg PW=$USER \
                               -t video-cleaner-dev \
                               -f docker/Dockerfile.dev \
                               .
