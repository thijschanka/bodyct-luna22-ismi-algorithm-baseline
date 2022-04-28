#!/usr/bin/env bash

./build.sh

docker save nodule_classifier | gzip -c > nodule_classifier.tar.gz
