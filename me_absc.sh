#!/bin/bash

# ===================================================================
# ME-ABSC Model Training Script
# ===================================================================
# This script automates the training of ME-ABSC across different
# domains and layer configurations. It supports both weight-based editing and non-weight-based editing
# training modes.
#
# Usage: bash me_absc.sh
# ===================================================================

# Set strict error handling
set -euo pipefail

# Configuration Variables
# =======================

# weight-based editing configuration - set to true/false as needed
readonly WEIGHT_EDITING_VALUES=(true)

# Layer configuration for feature selection and fine-tuning
# Uncomment the lines below for multiple layer combinations
#readonly FS_LAYER_VALUES=(0 5 5 10 10 15 15 20 20)
#readonly FN_LAYER_VALUES=(31 10 15 15 20 20 25 25 30)

# Current configuration - single layer combination
readonly FS_LAYER_VALUES=(10)
readonly FN_LAYER_VALUES=(15)

# Training domains for different datasets
readonly TRAIN_DOMAINS=('device' 'laptop' 'rest' 'service')

# Script paths and validation
readonly PYTHON_SCRIPT="model_editing.py"
readonly DEBUG_SCRIPT="debug_model_editing.py"

# Functions
# =========

# Print colored output for better visibility
print_info() {
    echo -e "\033[1;34m[INFO]\033[0m $1"
}

print_success() {
    echo -e "\033[1;32m[SUCCESS]\033[0m $1"
}

print_error() {
    echo -e "\033[1;31m[ERROR]\033[0m $1"
}

# Validate prerequisites before starting
validate_environment() {
    print_info "Validating environment..."

    # Check if Python script exists
    if [[ ! -f "$PYTHON_SCRIPT" ]]; then
        print_error "Python script '$PYTHON_SCRIPT' not found in current directory"
        exit 1
    fi

    # Check if Python is available
    if ! command -v python &> /dev/null; then
        print_error "Python is not installed or not in PATH"
        exit 1
    fi

    print_success "Environment validation completed"
}

# Main execution function
main() {
    print_info "Starting ME-ABSC model training pipeline..."
    print_info "Training domains: ${TRAIN_DOMAINS[*]}"
    print_info "weight-based editing modes: ${WEIGHT_EDITING_VALUES[*]}"

    local total_runs=0
    local completed_runs=0

    # Calculate total number of runs for progress tracking
    total_runs=$((${#WEIGHT_EDITING_VALUES[@]} * ${#FS_LAYER_VALUES[@]} * ${#TRAIN_DOMAINS[@]}))
    print_info "Total training runs planned: $total_runs"

    echo "========================================"

    # Main execution loops
    for is_lora in "${WEIGHT_EDITING_VALUES[@]}"; do
        print_info "Processing weight-based editing mode: $is_lora"

        # Loop through layer configurations
        for i in "${!FS_LAYER_VALUES[@]}"; do
            local fs_layer=${FS_LAYER_VALUES[$i]}
            local fn_layer=$((FN_LAYER_VALUES[$i] + 1))

            print_info "Layer configuration: FS=$fs_layer, FN=$fn_layer"
            print_info "About to start training domains loop..."
            print_info "Training domains: ${TRAIN_DOMAINS[*]}"

            # Loop through training domains
            for train_domain in "${TRAIN_DOMAINS[@]}"; do
                print_info "Entering loop for domain: $train_domain"
                completed_runs=$((completed_runs + 1))

                print_info "[$completed_runs/$total_runs] Training domain: $train_domain"
                print_info "Parameters: domain=$train_domain, fs_layer=$fs_layer, fn_layer=$fn_layer, lora=$is_lora"

                # Execute the training script with detailed logging
                print_info "Executing: python $PYTHON_SCRIPT $train_domain $fs_layer $fn_layer $is_lora"

                # Run with timeout and capture output
                python "$PYTHON_SCRIPT" "$train_domain" "$fs_layer" "$fn_layer" "$is_lora" 2>&1
                exit_code=$?

                if [ $exit_code -eq 0 ]; then
                    print_success "Completed training for $train_domain (weight-based editing: $is_lora, Layers: $fs_layer-$fn_layer)"
                else
                    print_error "Training failed with exit code $exit_code for $train_domain (weight-based editing: $is_lora, Layers: $fs_layer-$fn_layer)"
                    print_error "Check the Python script for errors or missing dependencies"
                    exit 1
                fi

                echo "----------------------------------------"
            done
        done
    done

    print_success "All training runs completed successfully!"
    print_info "Total runs: $completed_runs"
}

# Script execution
# ================

# Trap errors and provide cleanup
trap 'print_error "Script interrupted or failed at line $LINENO"' ERR

# Validate environment before starting
validate_environment

# Execute main function
main "$@"
