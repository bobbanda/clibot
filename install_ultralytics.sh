#!/bin/bash

# Script to install ultralytics with retry mechanism for hash mismatch issues

echo "Installing ultralytics with retry mechanism..."

# Function to install with retries
install_with_retry() {
    local max_attempts=3
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        echo "Attempt $attempt of $max_attempts..."
        
        # Clear pip cache before each attempt
        pip cache purge 2>/dev/null || true
        
        # Try to install with no cache and increased timeout
        if pip install --no-cache-dir --timeout 300 ultralytics; then
            echo "Successfully installed ultralytics!"
            return 0
        else
            echo "Installation failed on attempt $attempt"
            
            # If it's a hash mismatch error, try alternative methods
            if [ $attempt -lt $max_attempts ]; then
                echo "Waiting 5 seconds before retry..."
                sleep 5
                
                # On second attempt, try with trusted host
                if [ $attempt -eq 2 ]; then
                    echo "Trying with trusted host option..."
                    pip install --no-cache-dir --timeout 300 --trusted-host pypi.org --trusted-host files.pythonhosted.org ultralytics && return 0
                fi
            fi
        fi
        
        attempt=$((attempt + 1))
    done
    
    # If all attempts failed, try installing scipy separately first
    echo "All attempts failed. Trying to install scipy separately first..."
    pip install --no-cache-dir --timeout 300 "scipy>=1.4.1"
    
    # Then try ultralytics again
    pip install --no-cache-dir --timeout 300 ultralytics
}

# Main execution
install_with_retry