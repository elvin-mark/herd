#!/bin/bash

# ==============================================================================
# Herd CLI Functionality Integration Test Script
# Model: LiquidAI/LFM2.5-VL-450M-GGUF:LFM2.5-VL-450M-BF16.gguf
# Embedding: second-state/All-MiniLM-L6-v2-Embedding-GGUF:all-MiniLM-L6-v2-Q8_0.gguf
# Run manually: chmod +x tests/scripts/test_cli.sh && ./tests/scripts/test_cli.sh
# ==============================================================================

# Terminal colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

MODEL_NAME="LiquidAI/LFM2.5-VL-450M-GGUF:LFM2.5-VL-450M-BF16.gguf"
EMBED_MODEL="second-state/All-MiniLM-L6-v2-Embedding-GGUF:all-MiniLM-L6-v2-Q8_0.gguf"

echo -e "${CYAN}=== Starting Herd CLI Functionality Integration Tests ===${NC}"
echo -e "LLM Model:       ${YELLOW}${MODEL_NAME}${NC}"
echo -e "Embedding Model: ${YELLOW}${EMBED_MODEL}${NC}\n"

# 1. Verify 'herd' command is installed and executable
echo -e "${CYAN}[Step 1/8] Verifying Herd CLI installation...${NC}"
if ! command -v herd &> /dev/null; then
    echo -e "${RED}Error: 'herd' CLI command is not installed or not in PATH.${NC}"
    echo -e "Please install it in editable mode first: ${YELLOW}pip install -e .${NC}"
    exit 1
fi
echo -e "${GREEN}Herd CLI is installed successfully.${NC}\n"

# 2. Run System Doctor diagnostics
echo -e "${CYAN}[Step 2/8] Auditing hardware capability and environment configurations...${NC}"
herd doctor
if [ $? -ne 0 ]; then
    echo -e "${RED}Warning: 'herd doctor' reported system diagnostic checks failed.${NC}"
fi
echo -e "\n"

# 3. Pull the target test models
echo -e "${CYAN}[Step 3/8] Pulling test models...${NC}"
echo -e "Pulling LLM: ${YELLOW}${MODEL_NAME}${NC}"
herd pull "$MODEL_NAME"
if [ $? -ne 0 ]; then
    echo -e "${RED}Error: Failed to pull LLM model.${NC}"
    exit 1
fi
echo -e "Pulling Embedding Model: ${YELLOW}${EMBED_MODEL}${NC}"
herd pull "$EMBED_MODEL"
if [ $? -ne 0 ]; then
    echo -e "${RED}Error: Failed to pull embedding model.${NC}"
    exit 1
fi
echo -e "${GREEN}Models pulled successfully.${NC}\n"

# 4. List downloaded models and verify presence
echo -e "${CYAN}[Step 4/8] Verifying model presence in local catalog...${NC}"
herd list
if [ $? -ne 0 ]; then
    echo -e "${RED}Error: Failed to fetch local model library.${NC}"
    exit 1
fi
echo -e "\n"

# 5. Index local source files into the vector database using embedding model
echo -e "${CYAN}[Step 5/8] Indexing local files (src/herd/core) using ${YELLOW}${EMBED_MODEL}${NC}..."
herd db index ./src/herd/core --model "$EMBED_MODEL"
if [ $? -ne 0 ]; then
    echo -e "${RED}Error: RAG indexing failed.${NC}"
    exit 1
fi
echo -e "${GREEN}Local codebase directory indexed successfully.${NC}\n"

# 6. Run semantic search query against the index
echo -e "${CYAN}[Step 6/8] Executing semantic search against indexed files...${NC}"
herd db ask "What are the default host and port configs?" "$MODEL_NAME" --model "$EMBED_MODEL"
if [ $? -ne 0 ]; then
    echo -e "${RED}Error: Semantic RAG search failed.${NC}"
    exit 1
fi
echo -e "\n"

# 7. Check running processes and metrics statistics
echo -e "${CYAN}[Step 7/8] Auditing gateway process lists and statistics...${NC}"
echo -e "${YELLOW}--- Active Gateway Models (herd ps) ---${NC}"
herd ps
echo -e "\n${YELLOW}--- Gateway Metrics Stats (herd stats) ---${NC}"
herd stats
echo -e "\n"

# 8. Unload models to release VRAM and verify termination
echo -e "${CYAN}[Step 8/8] Stopping and unloading models to release system memory...${NC}"
herd stop "$MODEL_NAME"
if [ $? -ne 0 ]; then
    echo -e "${RED}Error: Failed to unload LLM model.${NC}"
    exit 1
fi
herd stop "$EMBED_MODEL"
if [ $? -ne 0 ]; then
    echo -e "${RED}Error: Failed to unload embedding model.${NC}"
    exit 1
fi
echo -e "\n${YELLOW}--- Active Gateway Models After Stop (herd ps) ---${NC}"
herd ps

echo -e "\n${GREEN}=== All Herd CLI functionality checks finished successfully! ===${NC}"
