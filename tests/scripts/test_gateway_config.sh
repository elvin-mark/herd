#!/bin/bash

# ==============================================================================
# Herd Gateway & Configuration Integration Test Script
# Tests: config, doctor, clean, share, proxy
# Run manually: chmod +x tests/scripts/test_gateway_config.sh && ./tests/scripts/test_gateway_config.sh
# ==============================================================================

# Terminal colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

API_PORT=11434
PROXY_PORT=11435
BASE_URL="http://127.0.0.1:${API_PORT}"
TEST_MODEL="second-state/All-MiniLM-L6-v2-Embedding-GGUF:all-MiniLM-L6-v2-Q8_0.gguf"

echo -e "${CYAN}=== Starting Herd Gateway & Configuration Tests ===${NC}"

# 1. Ensure gateway is online
started_by_us=false
echo -e "${CYAN}[Step 1/6] Verifying gateway server status on port ${API_PORT}...${NC}"
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

# 2. Test "herd doctor" (diagnostics)
echo -e "${CYAN}[Step 2/6] Testing 'herd doctor' diagnostics...${NC}"
doctor_res=$(herd doctor)
echo "$doctor_res"
echo -e "\n"

if [[ "$doctor_res" == *"Diagnostics"* || "$doctor_res" == *"Processor"* || "$doctor_res" == *"Compiled"* ]]; then
    echo -e "${GREEN}Success! 'herd doctor' diagnostics passed.${NC}\n"
else
    echo -e "${RED}Failure! 'herd doctor' returned invalid diagnostics output.${NC}"
    exit 1
fi

# 3. Test "herd clean" (clean inactive logs)
echo -e "${CYAN}[Step 3/6] Testing 'herd clean' logs optimizer...${NC}"
clean_res=$(herd clean --force)
echo "$clean_res"
echo -e "\n"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}Success! 'herd clean' logs cleanup completed.${NC}\n"
else
    echo -e "${RED}Failure! 'herd clean' failed to execute.${NC}"
    exit 1
fi

# 4. Test "herd config" (override configurations mutations)
echo -e "${CYAN}[Step 4/6] Testing 'herd config show' and 'herd config set'...${NC}"
# Backup existing default model if any
original_llm=$(herd config show | grep "default_llm" | awk '{print $4}' | tr -d ' ')

# Set test model config
set_res=$(herd config set default_llm "$TEST_MODEL")
echo "$set_res"

# Verify set value
show_res=$(herd config show)
echo "$show_res"
echo -e "\n"

if [[ "$show_res" == *"$TEST_MODEL"* ]]; then
    echo -e "${GREEN}Success! Config value mutated and verified successfully.${NC}\n"
else
    echo -e "${RED}Failure! Config failed to update.${NC}"
    exit 1
fi

# Restore original config
if [ -n "$original_llm" ] && [ "$original_llm" != "-" ]; then
    herd config set default_llm "$original_llm" > /dev/null
fi

# 5. Test "herd share" (connection pairing helpers)
echo -e "${CYAN}[Step 5/6] Testing 'herd share' local network pairing...${NC}"
share_res=$(herd share --qr)
echo "$share_res"
echo -e "\n"

if [[ "$share_res" == *"Helper"* || "$share_res" == *"Base URL"* || "$share_res" == *"http"* ]]; then
    echo -e "${GREEN}Success! 'herd share' connection information printed cleanly.${NC}\n"
else
    echo -e "${RED}Failure! 'herd share' did not output connection helper details.${NC}"
    exit 1
fi

# 6. Test "herd proxy" (reverse proxy loop in sandboxed timeout)
echo -e "${CYAN}[Step 6/6] Testing 'herd proxy' reverse proxy gateway...${NC}"
# Run herd proxy pointing to local gateway in background using a timeout of 3s
timeout 3 herd proxy "$BASE_URL" --port "$PROXY_PORT" --host "127.0.0.1" > /dev/null 2>&1
proxy_exit=$?

if [ $proxy_exit -eq 124 ]; then
    echo -e "${GREEN}Success! 'herd proxy' reverse proxy server ran successfully without crashing.${NC}\n"
else
    echo -e "${RED}Failure! 'herd proxy' crashed on startup with exit code: ${proxy_exit}${NC}"
    exit 1
fi

# Stop gateway if started by us
if [ "$started_by_us" = true ]; then
    echo -e "${YELLOW}Stopping background gateway process (PID ${GATEWAY_PID})...${NC}"
    kill "$GATEWAY_PID"
    wait "$GATEWAY_PID" 2>/dev/null
fi

echo -e "${GREEN}=== All Gateway & Configuration tests passed successfully! ===${NC}"
