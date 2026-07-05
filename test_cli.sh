#!/bin/bash

# ==============================================================================
# Herd CLI Functionality Integration Test Script
# Model: LiquidAI/LFM2.5-VL-450m:LFM2.5-VL-450M.F16
# Run manually: chmod +x test_cli.sh && ./test_cli.sh
# ==============================================================================

# Terminal colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

MODEL_NAME="LiquidAI/LFM2.5-VL-450M-GGUF:LFM2.5-VL-450M-BF16.gguf"

echo -e "${CYAN}=== Starting Herd CLI Functionality Integration Tests ===${NC}"
echo -e "Target Model: ${YELLOW}${MODEL_NAME}${NC}\n"

# 1. Verify 'herd' command is installed and executable
echo -e "${CYAN}[Step 1/7] Verifying Herd CLI installation...${NC}"
if ! command -v herd &> /dev/null; then
    echo -e "${RED}Error: 'herd' CLI command is not installed or not in PATH.${NC}"
    echo -e "Please install it in editable mode first: ${YELLOW}pip install -e .${NC}"
    exit 1
fi
echo -e "${GREEN}Herd CLI is installed successfully.${NC}\n"

# 2. Pull the target lightweight test model
echo -e "${CYAN}[Step 2/7] Pulling test model: ${YELLOW}${MODEL_NAME}${NC}..."
herd pull "$MODEL_NAME"
if [ $? -ne 0 ]; then
    echo -e "${RED}Error: Failed to pull the model.${NC}"
    exit 1
fi
echo -e "${GREEN}Model pulled successfully.${NC}\n"

# 3. List downloaded models and verify presence
echo -e "${CYAN}[Step 3/7] Verifying model presence in local catalog...${NC}"
herd list
if [ $? -ne 0 ]; then
    echo -e "${RED}Error: Failed to fetch local model library.${NC}"
    exit 1
fi
echo -e "\n"

# 4. Index local source files into the vector database
echo -e "${CYAN}[Step 4/7] Recursively indexing local source files (herd/core) into RAG DB...${NC}"
herd index ./herd/core --model "$MODEL_NAME"
if [ $? -ne 0 ]; then
    echo -e "${RED}Error: RAG indexing failed.${NC}"
    exit 1
fi
echo -e "${GREEN}Local codebase directory indexed successfully.${NC}\n"

# 5. Run semantic search query against the index
echo -e "${CYAN}[Step 5/7] Executing semantic search against indexed files...${NC}"
herd ask "What are the default host and port configs?" --model "$MODEL_NAME"
if [ $? -ne 0 ]; then
    echo -e "${RED}Error: Semantic RAG search failed.${NC}"
    exit 1
fi
echo -e "\n"

# 6. Check running processes and metrics statistics
echo -e "${CYAN}[Step 6/7] Auditing gateway process lists and statistics...${NC}"
echo -e "${YELLOW}--- Active Gateway Models (herd ps) ---${NC}"
herd ps
echo -e "\n${YELLOW}--- Gateway Metrics Stats (herd stats) ---${NC}"
herd stats
echo -e "\n"

# 7. Unload model to release VRAM and verify termination
echo -e "${CYAN}[Step 7/7] Stopping and unloading model to release system memory...${NC}"
herd stop "$MODEL_NAME"
if [ $? -ne 0 ]; then
    echo -e "${RED}Error: Failed to unload the model.${NC}"
    exit 1
fi
echo -e "\n${YELLOW}--- Active Gateway Models After Stop (herd ps) ---${NC}"
herd ps

echo -e "\n${GREEN}=== All Herd CLI functionality checks finished successfully! ===${NC}"
