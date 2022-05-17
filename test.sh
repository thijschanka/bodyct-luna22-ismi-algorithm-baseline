#!/usr/bin/env bash

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

./build.sh

VOLUME_SUFFIX=$(dd if=/dev/urandom bs=32 count=1 | md5sum | cut --delimiter=' ' --fields=1)
MEM_LIMIT="4g"  # Maximum is currently 30g, configurable in your algorithm image settings on grand challenge

docker volume create nodule_classifier-output-$VOLUME_SUFFIX

# Do not change any of the parameters to docker run, these are fixed
docker run --rm \
        --gpus=all \
        --memory="${MEM_LIMIT}" \
        --memory-swap="${MEM_LIMIT}" \
        --network="none" \
        --cap-drop="ALL" \
        --security-opt="no-new-privileges" \
        --shm-size="128m" \
        --pids-limit="256" \
        -v $SCRIPTPATH/test/:/input/ \
        -v nodule_classifier-output-$VOLUME_SUFFIX:/output/ \
        nodule_classifier

docker run --rm \
        -v nodule_classifier-output-$VOLUME_SUFFIX:/output/ \
        -v $SCRIPTPATH/test/:/input/ \
        python:3.9-slim python -c """
import json

predicted_risk = json.load(open('/output/lung-nodule-malignancy-risk.json'))
predicted_type = json.load(open('/output/lung-nodule-type.json'))

expected_outputs = json.load(open('/input/expected_outputs.json'))

assert predicted_risk == expected_outputs['malignancy_risk'], 'malignancy risk does not match; test failed!'
assert predicted_type == expected_outputs['nodule_type'], 'nodule type does not match; test failed!'

print('Tests successfully passed!')

"""

docker volume rm nodule_classifier-output-$VOLUME_SUFFIX
