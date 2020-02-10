#!/usr/bin/env bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
BASE_DIR="${SCRIPT_DIR}/.."

PYTHON_PACKAGE=purpl

# # Delete any downloaded data
# rm -rf ${BASE_DIR}/.data
# Remove the logging generated files
rm -rf ${BASE_DIR}/logs
# Remove the old results
rm -rf ${BASE_DIR}/res
# Remove the files defining the PU split
rm -rf ${BASE_DIR}/tensors
# Delete all tensorboard folders
rm -rf ${BASE_DIR}/tb
# Remove the generated models
rm -rf ${BASE_DIR}/models
# Remove the slurm logs
rm -rf ${BASE_DIR}/out
# Delete all plots
rm -rf ${BASE_DIR}/plots
# Remove the Python cache
find ${BASE_DIR} -type d -name __pycache__ -exec rm -rf {} \; > /dev/null
# Remove the vim PyRope files
rm -rf ${BASE_DIR}/.ropeproject
rm -rf ${BASE_DIR}/${PYTHON_PACKAGE}/.ropeproject
# Delete the MacOS File
rm -rf ${BASE_DIR}/.DS_Store
