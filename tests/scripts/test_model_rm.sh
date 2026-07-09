#!/bin/bash

# ==============================================================================
# Herd Model Removal (rm) Integration Test Script
# Run manually: chmod +x tests/scripts/test_model_rm.sh && ./tests/scripts/test_model_rm.sh
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

echo -e "${CYAN}=== Starting Herd Model Removal (rm) Tests ===${NC}"

# Define paths
HERD_MODELS_DIR="${HOME}/.herd/models"
TEST_AUTHOR="test-author-rm"
TEST_REPO="test-repo-rm"
TEST_DIR="${HERD_MODELS_DIR}/huggingface/${TEST_AUTHOR}/${TEST_REPO}"

# Clean up any leftover test dirs
rm -rf "${HERD_MODELS_DIR}/huggingface/${TEST_AUTHOR}"

# 1. Test removing a specific tagged GGUF file
echo -e "${CYAN}[Step 1/3] Testing specific file removal (tag-based)...${NC}"
mkdir -p "${TEST_DIR}"
touch "${TEST_DIR}/dummy-model-Q4_K_M.gguf"
touch "${TEST_DIR}/dummy-model-Q8_0.gguf"

# Run herd rm for the Q4_K_M tag
herd rm "${TEST_AUTHOR}/${TEST_REPO}:Q4_K_M" -y
if [ $? -eq 0 ] && [ ! -f "${TEST_DIR}/dummy-model-Q4_K_M.gguf" ] && [ -f "${TEST_DIR}/dummy-model-Q8_0.gguf" ]; then
    echo -e "${GREEN}Success! Only the tagged model file was deleted.${NC}\n"
else
    echo -e "${RED}Failure! Tagged model file removal failed.${NC}"
    rm -rf "${HERD_MODELS_DIR}/huggingface/${TEST_AUTHOR}"
    exit 1
fi

# 2. Test directory cleanup when the last file is deleted
echo -e "${CYAN}[Step 2/3] Testing directory cleanup on deleting the final file...${NC}"
herd rm "${TEST_AUTHOR}/${TEST_REPO}:Q8_0" -y
if [ $? -eq 0 ] && [ ! -d "${HERD_MODELS_DIR}/huggingface/${TEST_AUTHOR}" ]; then
    echo -e "${GREEN}Success! Empty parent author and repo directories were cleaned up.${NC}\n"
else
    echo -e "${RED}Failure! Empty directories were not cleaned up.${NC}"
    rm -rf "${HERD_MODELS_DIR}/huggingface/${TEST_AUTHOR}"
    exit 1
fi

# 3. Test removing an entire model repository
echo -e "${CYAN}[Step 3/3] Testing complete repository directory removal...${NC}"
mkdir -p "${TEST_DIR}"
touch "${TEST_DIR}/dummy-model-Q4_K_M.gguf"
touch "${TEST_DIR}/dummy-model-Q8_0.gguf"

# Run herd rm for the entire repository
herd rm "${TEST_AUTHOR}/${TEST_REPO}" -y
if [ $? -eq 0 ] && [ ! -d "${HERD_MODELS_DIR}/huggingface/${TEST_AUTHOR}" ]; then
    echo -e "${GREEN}Success! Entire repository directory was deleted and cleaned up.${NC}\n"
else
    echo -e "${RED}Failure! Complete repository removal failed.${NC}"
    rm -rf "${HERD_MODELS_DIR}/huggingface/${TEST_AUTHOR}"
    exit 1
fi

echo -e "${GREEN}=== All Model Removal (rm) Integration Tests Completed Successfully! ===${NC}"
