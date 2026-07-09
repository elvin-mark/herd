#!/bin/bash

# ==============================================================================
# Herd Autonomous Agent REPL Interactive Integration Test Script
# Run manually: chmod +x tests/scripts/test_agent_repl.sh && ./tests/scripts/test_agent_repl.sh
# ==============================================================================

# Terminal colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Ensure local herd package is loaded from the repository root
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
herd() {
    PYTHONPATH="$REPO_ROOT" python3 -m herd.cli "$@"
}

echo -e "${CYAN}=== Starting Herd Agent REPL Interactive Tests ===${NC}"

# 1. Test clean exit on 'exit' input
echo -e "${CYAN}[Step 1/3] Testing clean REPL exit on 'exit' command...${NC}"
echo "exit" | herd agent
if [ $? -eq 0 ]; then
    echo -e "${GREEN}Success! Agent exited cleanly with status code 0.${NC}\n"
else
    echo -e "${RED}Failure! Agent exited with non-zero status code on 'exit'.${NC}"
    exit 1
fi

# 2. Test clean exit on EOF (empty stdin)
echo -e "${CYAN}[Step 2/3] Testing clean REPL exit on EOF (empty stdin)...${NC}"
echo -n "" | herd agent
if [ $? -eq 0 ]; then
    echo -e "${GREEN}Success! Agent exited cleanly on EOF with status code 0.${NC}\n"
else
    echo -e "${RED}Failure! Agent failed to exit cleanly on EOF.${NC}"
    exit 1
fi

# 3. Test objective execution + REPL exit combination
echo -e "${CYAN}[Step 3/3] Testing task execution followed by interactive exit...${NC}"
# Use a simple objective that doesn't require actual LLM reasoning to test pipeline flow
(echo "list the files in the current folder"; echo "exit") | herd agent --max-turns 1
if [ $? -eq 0 ]; then
    echo -e "${GREEN}Success! Agent processed conversation and exited cleanly.${NC}\n"
else
    echo -e "${RED}Failure! Agent failed interactive conversation pipeline.${NC}"
    exit 1
fi

echo -e "${GREEN}=== All Agent REPL Integration Tests Completed Successfully! ===${NC}"
