#!/bin/bash

set -e
set -u

docker run --gpus all -it --rm --ipc=host -v ./:/project video-cleaner-dev
