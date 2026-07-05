#!/bin/bash

# ==============================================================================
# Herd Parallel Load & Performance Benchmark Test Script
# Run manually: chmod +x tests/scripts/test_benchmark.sh && ./tests/scripts/test_benchmark.sh
# ==============================================================================

# Terminal colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

API_PORT=11434
BASE_URL="http://127.0.0.1:${API_PORT}"
MODEL_NAME="LiquidAI/LFM2.5-VL-450M-GGUF:LFM2.5-VL-450M-BF16.gguf"

echo -e "${CYAN}=== Starting Herd Parallel Load & Performance Benchmark ===${NC}"

# 1. Ensure gateway is online and model is loaded
if ! curl -s "${BASE_URL}/health" &> /dev/null; then
    echo -e "${YELLOW}Gateway is offline. Starting herd gateway in background...${NC}"
    herd serve &
    GATEWAY_PID=$!
    sleep 3
fi

echo -e "Pre-loading model: ${YELLOW}${MODEL_NAME}${NC}..."
curl -s -X POST "${BASE_URL}/v1/models/load" \
  -H "Content-Type: application/json" \
  -d "{\"model\": \"${MODEL_NAME}\"}" > /dev/null

# 2. Trigger concurrent requests in parallel
CONCURRENCY=4
echo -e "\n${CYAN}Triggering ${CONCURRENCY} parallel chat completions requests...${NC}"
start_time=$(date +%s.%N)

declare -a pids
for i in $(seq 1 $CONCURRENCY); do
    echo -e "  -> Launching request #$i..."
    curl -s -o /dev/null -w "%{http_code}" -X POST "${BASE_URL}/v1/chat/completions" \
      -H "Content-Type: application/json" \
      -d "{\"model\": \"${MODEL_NAME}\", \"messages\": [{\"role\": \"user\", \"content\": \"Write a single line about physics.\"}], \"stream\": false}" &
    pids[$i]=$!
done

# Wait for all background requests to finish
failures=0
for i in $(seq 1 $CONCURRENCY); do
    wait ${pids[$i]}
    exit_code=$?
    if [ $exit_code -ne 0 ]; then
        failures=$((failures + 1))
    fi
done

end_time=$(date +%s.%N)
duration=$(echo "$end_time - $start_time" | bc)
avg_latency=$(echo "$duration / $CONCURRENCY" | bc -l)

echo -e "\n${CYAN}=== Performance Summary ===${NC}"
echo -e "  Total Requests:  [bold white]${CONCURRENCY}[/bold white]"
echo -e "  Failed Requests: [bold red]${failures}[/bold red]"
echo -e "  Total Time:      [bold white]$(printf "%.3f" $duration) seconds[/bold white]"
echo -e "  Avg Latency/Req: [bold green]$(printf "%.3f" $avg_latency) seconds[/bold green]"

# 3. Clean up
curl -s -X POST "${BASE_URL}/v1/models/unload" -H "Content-Type: application/json" -d "{\"model\": \"${MODEL_NAME}\"}" > /dev/null

if [ ! -z "$GATEWAY_PID" ]; then
    kill "$GATEWAY_PID"
    wait "$GATEWAY_PID" 2>/dev/null
fi

if [ $failures -eq 0 ]; then
    echo -e "\n${GREEN}Benchmark test passed with 100% success rate!${NC}"
else
    echo -e "\n${RED}Benchmark test finished with ${failures} request failures.${NC}"
    exit 1
fi
