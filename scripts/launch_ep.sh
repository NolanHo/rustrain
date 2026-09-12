#!/bin/bash
# Launch 4 server processes for EP=4 on Qwen3.6-35B-A3B
# Each process binds to one GPU via LOCAL_RANK

set -e

cd /mnt/workspace/rustrain

# Clean up any old NCCL ID file
rm -f /tmp/rustrain-nccl/nccl-persistent-id.bin 2>/dev/null
mkdir -p /tmp/rustrain-nccl

export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:/mnt/workspace/rustrain-env/lib/python3.12/site-packages/torch/lib:/mnt/workspace/rustrain-env/lib/python3.12-site-packages/nvidia/cudnn/lib:/mnt/workspace/rustrain-env/lib/python3.12-site-packages/nvidia/nccl/lib:/mnt/workspace/rustrain-env/lib/python3.12-site-packages/nvidia/cu13/lib:/usr/local/cuda/lib64"

export RUST_BACKTRACE=1
export QWEN36_SUBCKPT=1

# Kill any existing rustrain servers
pkill -9 rustrain 2>/dev/null || true
sleep 3

WORLD_SIZE=4

# Launch 4 processes, each on a different GPU + port
for RANK in 0 1 2 3; do
    PORT=$((8080 + RANK))
    GRPC_PORT=$((50051 + RANK))
    RANK=$RANK WORLD_SIZE=$WORLD_SIZE LOCAL_RANK=$RANK \
    ./target/release/rustrain server \
        --http-port $PORT --grpc-port $GRPC_PORT \
        --metrics-dir /tmp/rustrain-ep${RANK} \
        > /tmp/server_ep${RANK}.log 2>&1 &
    echo "Launched EP rank $RANK on port $PORT (GPU $RANK), PID=$!"
done

# Wait for all to start
sleep 10

# Check all servers are alive
for RANK in 0 1 2 3; do
    PORT=$((8080 + RANK))
    if curl -s http://localhost:$PORT/v1/sessions -X POST -H "Content-Type: application/json" -d "{\"session_id\":\"ep_test\"}" 2>/dev/null | grep -q "session_id"; then
        echo "EP rank $RANK: OK"
    else
        echo "EP rank $RANK: FAILED"
        cat /tmp/server_ep${RANK}.log | tail -5
    fi
done
