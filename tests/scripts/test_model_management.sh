#!/bin/bash

# ==============================================================================
# Herd Model Management Integration Test Script
# Tests: list, pull, search, suggest
# Run manually: chmod +x tests/scripts/test_model_management.sh && ./tests/scripts/test_model_management.sh
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

echo -e "${CYAN}=== Starting Herd Model Management Tests ===${NC}"

# 1. Ensure gateway is online
started_by_us=false
echo -e "${CYAN}[Step 1/4] Verifying gateway server status on port ${API_PORT}...${NC}"
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

# 2. Test "herd list" (list local downloaded models)
echo -e "${CYAN}[Step 2/4] Testing 'herd list'...${NC}"
list_res=$(herd list)
echo "$list_res"
echo -e "\n"

if [[ "$list_res" == *"Local Models"* || "$list_res" == *"File Name"* ]]; then
    echo -e "${GREEN}Success! 'herd list' checked local models catalog successfully.${NC}\n"
else
    echo -e "${RED}Failure! 'herd list' returned invalid catalog layout.${NC}"
    exit 1
fi

# 3. Test "herd search" (Hugging Face registry search)
echo -e "${CYAN}[Step 3/4] Testing 'herd search' on Hugging Face...${NC}"
search_res=$(herd search "All-MiniLM-L6-v2")
echo "$search_res"
echo -e "\n"

if [[ "$search_res" == *"Hugging Face"* || "$search_res" == *"second-state"* || "$search_res" == *"Repository"* ]]; then
    echo -e "${GREEN}Success! 'herd search' found remote Hugging Face GGUF repository matches.${NC}\n"
else
    echo -e "${RED}Failure! 'herd search' failed to find matching GGUF targets.${NC}"
    exit 1
fi


# 4. Test "herd pull" (GGUF model downloader checks)
echo -e "${CYAN}[Step 4/4] Testing 'herd pull' model validator...${NC}"
pull_res=$(herd pull "$TEST_MODEL")
echo "$pull_res"
echo -e "\n"

if [[ "$pull_res" == *"already exists"* || "$pull_res" == *"Successfully downloaded"* ]]; then
    echo -e "${GREEN}Success! 'herd pull' resolved and verified the model locally.${NC}\n"
else
    echo -e "${RED}Failure! 'herd pull' failed to download or verify the local model.${NC}"
    exit 1
fi

# Stop gateway if started by us
if [ "$started_by_us" = true ]; then
    echo -e "${YELLOW}Stopping background gateway process (PID ${GATEWAY_PID})...${NC}"
    kill "$GATEWAY_PID"
    wait "$GATEWAY_PID" 2>/dev/null
fi

echo -e "${GREEN}=== All Model Management tests passed successfully! ===${NC}"
