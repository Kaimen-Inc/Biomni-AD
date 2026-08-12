#!/bin/bash

# Configuration
BASE_PORT=9000
SHARED_READ_ONLY_PATH="/mnt/vm-shared-storage"
NUM_INSTANCES=$1

if [ -z "$1" ]; then
    echo "Usage: ./launch_fleet.sh [number_of_instances]"
    exit 1
fi

# --- NEW: CLEANUP SECTION ---
# This stops and removes any existing instances of this fleet
echo "Cleaning up existing fleet instances..."
docker ps -aq --filter "name=biomni-inst-" | xargs -r sudo docker rm -f
# ----------------------------

for i in $(seq 1 $NUM_INSTANCES)
do
    CURRENT_PORT=$((BASE_PORT + i))
    PROJECT_NAME="biomni-inst-$i"

    echo "Starting $PROJECT_NAME on port $CURRENT_PORT (Read-Only Mode)..."

    # Launching with a unique PROJECT_NAME allows multiple instances
    sudo -E HOST_PORT=$CURRENT_PORT \
         BIOMNI_USER_DATA_HOST_PATH="$SHARED_READ_ONLY_PATH" \
         docker-compose -p "$PROJECT_NAME" up -d
done

echo "Done! $NUM_INSTANCES instances are running."