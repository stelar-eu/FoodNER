#!/usr/bin/env bash

#
# A poor builder for foodNER
#
# @author: Konstantinos Pechlivanis
#

# SECTION:  Configuration

# Import poor mans logger
. ./poor.mans.logger.sh

# Define base folder
if [ -z "$BASE_PATH" ]; then
  export BASE_PATH=.
fi

# SECTION: directories
log_info "Create folder for logs: $(realpath $BASE_PATH)"
mkdir -p ${BASE_PATH}/logs
touch ${BASE_PATH}/logs/development.log
touch ${BASE_PATH}/logs/production.log

log_info "Create folders for project results"
mkdir -p ${BASE_PATH}/results/test_coverage
mkdir -p ${BASE_PATH}/results/{dataset,evaluation,model,prediction,statistical_analysis}/{spacy,pytorch,bert}
