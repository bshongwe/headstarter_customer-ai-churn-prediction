#!/bin/bash

# Function for logging messages
log() {
    local message=$1
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $message"
}

# Check required environment variables
log "Checking required environment variables..."
REQUIRED_VARS=("GROQ_API_KEY" "DOCKER_HUB_USERNAME" "DOCKER_IMAGE" "K8S_NAMESPACE")
for var in "${REQUIRED_VARS[@]}"; do
    if [ -z "${!var}" ]; then
        log "Error: Environment variable $var is not set."
        exit 1
    fi
done
log "All required environment variables are set."

# Docker Health Check
log "Checking if the Docker container is running..."
if [ "$(docker ps -q -f name=your_container_name)" ]; then
    log "Docker container is running."
else
    log "Error: Docker container is not running."
    exit 1
fi

# Kubernetes Pod Status Check
log "Checking Kubernetes pod status..."
POD_STATUS=$(kubectl get pods -n "$K8S_NAMESPACE" -l app=yourapp -o jsonpath='{.items[0].status.phase}')
if [ "$POD_STATUS" == "Running" ]; then
    log "Kubernetes pod is running."
else
    log "Error: Kubernetes pod status is $POD_STATUS."
    exit 1
fi

# Library Installation Verification
log "Verifying that all required libraries are installed..."
REQUIRED_LIBRARIES=("streamlit==1.26.0" "pandas==2.1.1" "numpy==1.24.3" "scikit-learn==1.3.0" "xgboost==1.7.6" "plotly==5.15.0" "matplotlib==3.8.1" "openai==0.29.0" "requests==2.31.0" "pickle5==0.0.12")
for lib in "${REQUIRED_LIBRARIES[@]}"; do
    if ! pip show "${lib%%=*}" &> /dev/null; then
        log "Error: Required library $lib is not installed."
        exit 1
    fi
done
log "All required libraries are installed."

# Functionality Tests for Libraries
test_library_functionality() {
    local lib_name=$1
    local test_command=$2
    log "Testing basic functionality for $lib_name..."
    if python -c "$test_command" &> /dev/null; then
        log "$lib_name is functioning correctly."
    else
        log "Error: $lib_name is not functioning correctly."
        exit 1
    fi
}

# Testing required libraries
test_library_functionality "Pandas" "import pandas as pd; pd.DataFrame({'test': [1, 2, 3]})"
test_library_functionality "scikit-learn" "from sklearn.linear_model import LinearRegression; model = LinearRegression()"
test_library_functionality "XGBoost" "import xgboost as xgb; model = xgb.XGBRegressor()"

# Test main.py functions
log "Testing main.py functions..."

# Test load_data function
log "Testing load_data function..."
if python -c "import main; main.load_data('test_file.csv')" &> /dev/null; then
    log "load_data function passed."
else
    log "Error: load_data function failed."
    exit 1
fi

# Test run_model function
log "Testing run_model function..."
if python -c "import main; import pandas as pd; data = pd.DataFrame({'feature': [1, 2, 3], 'target': [1, 2, 3]}); model = main.run_model(data)" &> /dev/null; then
    log "run_model function passed."
else
    log "Error: run_model function failed."
    exit 1
fi

# Test display_data function
log "Testing display_data function..."
if python -c "import main; import pandas as pd; data = pd.DataFrame({'feature': [1, 2, 3], 'target': [1, 2, 3]}); main.display_data(data)" &> /dev/null; then
    log "display_data function passed."
else
    log "Error: display_data function failed."
    exit 1
fi

# API Endpoint Test
log "Testing API endpoint..."
API_URL="http://your-app-url/api/endpoint"
RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" "$API_URL")
if [ "$RESPONSE" -eq 200 ]; then
    log "API endpoint is reachable. Status code: $RESPONSE"
else
    log "Error: API endpoint returned status code $RESPONSE."
    exit 1
fi

# UI Tests
log "Testing Streamlit app UI..."
UI_URL="http://your-app-url"
UI_RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" "$UI_URL")
if [ "$UI_RESPONSE" -eq 200 ]; then
    log "Streamlit app UI is reachable. Status code: $UI_RESPONSE"
else
    log "Error: Streamlit app UI returned status code $UI_RESPONSE."
    exit 1
fi

log "=== Deployment Test Suite Completed Successfully ==="
