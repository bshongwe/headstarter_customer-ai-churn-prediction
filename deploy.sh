#!/bin/bash

# Configuration
IMAGE_NAME="username/app:latest"  # Change this to your Docker Hub username and app name
YAML_FILE="streamlit-deployment.yaml"      # Your Kubernetes deployment configuration file

# Function for logging messages with timestamps
log() {
    local message=$1
    local status=$2
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $message"
    if [ "$status" != "success" ]; then
        exit 1
    fi
}

# Function to run tests and capture results
run_tests() {
    log "Running tests..." "info"
    TEST_RESULTS=$(python -m unittest discover -s . -p "test_main.py" 2>&1)
    TEST_COUNT=$(echo "$TEST_RESULTS" | grep -c "ok")
    TEST_FAILED=$(echo "$TEST_RESULTS" | grep -c "FAIL\|ERROR")

    echo "$TEST_RESULTS"

    if [ "$TEST_FAILED" -ne 0 ]; then
        log "Tests Failed: $TEST_FAILED" "error"
    else
        log "All tests passed: $TEST_COUNT tests." "success"
    fi
}

# Run the tests before proceeding with the deployment
run_tests

# Build Docker image
log "Building Docker image..." "info"
docker build -t "$IMAGE_NAME" .
if [ $? -eq 0 ]; then
    log "Docker image built successfully." "success"
else
    log "Failed to build Docker image." "error"
fi

# Push Docker image to Docker Hub
log "Pushing Docker image to Docker Hub..." "info"
docker push "$IMAGE_NAME"
if [ $? -eq 0 ]; then
    log "Docker image pushed successfully." "success"
else
    log "Failed to push Docker image." "error"
fi

# Apply Kubernetes configuration
log "Deploying to Kubernetes with $YAML_FILE..." "info"
kubectl apply -f "$YAML_FILE"
if [ $? -eq 0 ]; then
    log "Kubernetes deployment applied successfully." "success"
else
    log "Failed to apply Kubernetes deployment." "error"
fi

log "=== Deployment Process Completed Successfully ===" "success"

