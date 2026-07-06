#!/bin/bash

# ==============================================================================
# Herd Runtime Operations Integration Test Script
# Tests: ps, stop, top, stats, logs
# Run manually: chmod +x tests/scripts/test_runtime_operations.sh && ./tests/scripts/test_runtime_operations.sh
# ==============================================================================

# Terminal colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

API_PORT=11434
BASE_URL="http://127.0.0.1:${API_PORT}"
TEST_MODEL="second-state/All-MiniLM-L6-v2-Embedding-GGUF:all-MiniLM-L6-v2-Q8_0.gguf"

echo -e "${CYAN}=== Starting Herd Runtime Operations Tests ===${NC}"

# 1. Ensure gateway is online
started_by_us=false
echo -e "${CYAN}[Step 1/7] Verifying gateway server status on port ${API_PORT}...${NC}"
if ! curl -s "${BASE_URL}/health" &>/dev/null; then
    echo -e "${YELLOW}Gateway is offline. Starting herd gateway in background...${NC}"
    herd serve &
    GATEWAY_PID=$!
    started_by_us=true
    sleep 3
fi

# Ping check
if curl -s -f "${BASE_URL}/health" | grep -q "ok"; then
    echo -e "${GREEN}Gateway server is online.${NC}\n"
else
    echo -e "${RED}Error: Gateway failed to start.${NC}"
    exit 1
fi

# 2. Pull model to guarantee it is locally cached
echo -e "${CYAN}[Step 2/7] Ensuring test model is cached: ${YELLOW}${TEST_MODEL}${NC}..."
herd pull "$TEST_MODEL"
if [ $? -ne 0 ]; then
    echo -e "${RED}Error: Failed to verify/pull test model.${NC}"
    exit 1
fi
echo -e "${GREEN}Test model cached.${NC}\n"

# 3. Load model in gateway (using --embedding to exit instantly instead of entering REPL)
echo -e "${CYAN}[Step 3/7] Loading test model in background...${NC}"
herd run "$TEST_MODEL" --embedding
if [ $? -ne 0 ]; then
    echo -e "${RED}Error: Failed to load model.${NC}"
    exit 1
fi
sleep 2 # Let the model server startup fully
echo -e "${GREEN}Model loaded successfully.${NC}\n"

# 4. Test "herd ps" (list running)
echo -e "${CYAN}[Step 4/7] Testing 'herd ps'...${NC}"
ps_res=$(herd ps)
echo -e "${YELLOW}Active process output:${NC}"
echo "$ps_res"
echo -e "\n"

if [[ "$ps_res" == *"$TEST_MODEL"* ]]; then
    echo -e "${GREEN}Success! 'herd ps' listed the active running model.${NC}\n"
else
    echo -e "${RED}Failure! 'herd ps' did not list the loaded model.${NC}"
    exit 1
fi

# 5. Test "herd stats" (model statistics)
echo -e "${CYAN}[Step 5/7] Testing 'herd stats'...${NC}"
stats_res=$(herd stats)
echo -e "${YELLOW}Model stats output:${NC}"
echo "$stats_res"
echo -e "\n"

if [[ "$stats_res" == *"Requests"* || "$stats_res" == *"Tokens"* || "$stats_res" == *"Model"* ]]; then
    echo -e "${GREEN}Success! 'herd stats' retrieved metric summaries.${NC}\n"
else
    echo -e "${RED}Failure! 'herd stats' returned invalid output.${NC}"
    exit 1
fi

# 6. Test "herd logs" (server logs)
echo -e "${CYAN}[Step 6/7] Testing 'herd logs' for model...${NC}"
logs_res=$(herd logs "$TEST_MODEL" --lines 15)
echo -e "${YELLOW}Tail logs output:${NC}"
echo "$logs_res"
echo -e "\n"

if [[ "$logs_res" == *"llama"* || "$logs_res" == *"server"* || "$logs_res" == *"Tailing"* ]]; then
    echo -e "${GREEN}Success! 'herd logs' retrieved model log output.${NC}\n"
else
    echo -e "${RED}Failure! 'herd logs' failed to view logs.${NC}"
    exit 1
fi

# 7. Test "herd top" (terminal live resource monitor in sandboxed timeout)
echo -e "${CYAN}[Step 7/7] Testing 'herd top' live terminal monitor...${NC}"
# Run herd top with a timeout of 3s to let the live display render.
# Since it is a blocking live loop, timeout will terminate it with exit code 124.
# If it crashes on startup, it will exit with code 1 or similar.
timeout 3 herd top > /dev/null 2>&1
top_exit=$?

if [ $top_exit -eq 124 ]; then
    echo -e "${GREEN}Success! 'herd top' TUI started and ran successfully in sandboxed loop.${NC}\n"
else
    echo -e "${RED}Failure! 'herd top' crashed on startup with exit code: ${top_exit}${NC}"
    exit 1
fi

# 8. Teardown: Test "herd stop" (unload running model)
echo -e "${CYAN}[Cleanup] Unloading running model using 'herd stop'...${NC}"
stop_res=$(herd stop "$TEST_MODEL")
echo "$stop_res"

# Verify that the model is no longer active
ps_check=$(herd ps)
if [[ "$ps_check" == *"No models are currently running"* ]]; then
    echo -e "${GREEN}Success! Model stopped and unloaded cleanly.${NC}\n"
else
    echo -e "${RED}Failure! Model process is still active after stop command.${NC}"
    exit 1
fi

# Stop gateway if started by us
if [ "$started_by_us" = true ]; then
    echo -e "${YELLOW}Stopping background gateway process (PID ${GATEWAY_PID})...${NC}"
    kill "$GATEWAY_PID"
    wait "$GATEWAY_PID" 2>/dev/null
fi

echo -e "${GREEN}=== All Runtime Operations tests passed successfully! ===${NC}"
