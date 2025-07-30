#!/bin/bash
docker run --rm -it -v $(pwd)/instances:/opt/instances -v $(pwd):/home/sat/github mlbf_docker /bin/bash
