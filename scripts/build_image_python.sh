#!/usr/bin/env bash
docker build --pull --rm -f "Dockerfile" -t "mlbf_docker:latest" .
