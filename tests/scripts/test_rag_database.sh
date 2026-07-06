#!/bin/bash

# ==============================================================================
# Herd RAG Database Integrity & Semantic Retrieval Test Script
# Run manually: chmod +x tests/scripts/test_rag_database.sh && ./tests/scripts/test_rag_database.sh
# ==============================================================================

# Terminal colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

MODEL_NAME="LiquidAI/LFM2.5-VL-450M-GGUF:LFM2.5-VL-450M-F16.gguf"
EMBED_MODEL="second-state/All-MiniLM-L6-v2-Embedding-GGUF:all-MiniLM-L6-v2-Q8_0.gguf"

# Setup temporary directory for testing
TEMP_DIR="/tmp/herd_rag_test"
echo -e "${CYAN}=== Starting Herd RAG Database & Semantic Retrieval Tests ===${NC}"
echo -e "Creating temporary workspace at: ${YELLOW}${TEMP_DIR}${NC}"
rm -rf "$TEMP_DIR"
mkdir -p "$TEMP_DIR"

# 1. Populate workspace with known semantic document content
echo -e "${CYAN}[Step 1/8] Creating test documents with unique keyword keys...${NC}"
cat << 'EOF' > "$TEMP_DIR/doc1.txt"
The secret code phrase for accessing the secure research vault is: AlphaOmega42.
Make sure to keep this code confidential and do not share it with external vendors.
EOF

cat << 'EOF' > "$TEMP_DIR/doc2.txt"
Herd is designed by the Antigravity team. It runs local GGUF models on system CPU or CUDA GPU hardware.
The default configuration file is stored in ~/.herd/config.json.
EOF
echo -e "${GREEN}Created doc1.txt and doc2.txt.${NC}\n"

# 2. Ensure gateway is online and pull embedding model
echo -e "${CYAN}[Step 2/8] Pulling/checking embedding model: ${YELLOW}${EMBED_MODEL}${NC}..."
herd pull "$EMBED_MODEL" > /dev/null
herd pull "$MODEL_NAME" > /dev/null
echo -e "${GREEN}Model pull verified.${NC}\n"

# 3. Index the temporary directory
echo -e "${CYAN}[Step 3/8] Indexing temporary workspace...${NC}"
herd index "$TEMP_DIR" --model "$EMBED_MODEL"
if [ $? -ne 0 ]; then
    echo -e "${RED}Error: RAG indexing failed.${NC}"
    rm -rf "$TEMP_DIR"
    exit 1
fi
echo -e "${GREEN}Indexing completed successfully.${NC}\n"

# 4. Test "herd db list" (list indexed files)
echo -e "${CYAN}[Step 4/8] Testing 'herd db list' command...${NC}"
db_list_res=$(herd db list)
echo -e "${YELLOW}Database list output:${NC}"
echo "$db_list_res"
echo -e "\n"

if [[ "$db_list_res" == *"doc1.txt"* && "$db_list_res" == *"doc2.txt"* ]]; then
    echo -e "${GREEN}Success! 'herd db list' contains indexed files.${NC}\n"
else
    echo -e "${RED}Failure! 'herd db list' is missing indexed file records.${NC}"
    rm -rf "$TEMP_DIR"
    exit 1
fi

# 5. Test "herd db search" (direct semantic similarity query)
echo -e "${CYAN}[Step 5/8] Testing 'herd db search' semantic vector queries...${NC}"
db_search_res=$(herd db search "secret vault code" --model "$EMBED_MODEL" --limit 2)
echo -e "${YELLOW}Database search output:${NC}"
echo "$db_search_res"
echo -e "\n"

if [[ "$db_search_res" == *"AlphaOmega42"* && "$db_search_res" == *"doc1.txt"* ]]; then
    echo -e "${GREEN}Success! 'herd db search' found semantic vector matches.${NC}\n"
else
    echo -e "${RED}Failure! 'herd db search' did not locate the secret code snippet.${NC}"
    rm -rf "$TEMP_DIR"
    exit 1
fi

# 6. Query SQLite database metadata directly
echo -e "${CYAN}[Step 6/8] Querying SQLite vector database metadata...${NC}"
sqlite_db="$HOME/.herd/embeddings.db"
if [ ! -f "$sqlite_db" ]; then
    echo -e "${RED}Error: SQLite database file not found at ${sqlite_db}${NC}"
    rm -rf "$TEMP_DIR"
    exit 1
fi

echo -e "${YELLOW}Reading database files table:${NC}"
sqlite3 "$sqlite_db" "SELECT file_path, model_name, COUNT(*) FROM chunks WHERE file_path LIKE '%herd_rag_test%' GROUP BY file_path, model_name;"
if [ $? -ne 0 ]; then
    echo -e "${RED}Error: SQLite query failed.${NC}"
    rm -rf "$TEMP_DIR"
    exit 1
fi
echo -e "\n"

# 7. Run Semantic Ask Query and assert keyword presence in output
echo -e "${CYAN}[Step 7/8] Querying semantic retrieve-and-generate (herd ask)...${NC}"
ask_res=$(herd ask "What is the secret code phrase for accessing the research vault?" "$MODEL_NAME" --model "$EMBED_MODEL")
echo -e "${YELLOW}Response output:${NC}"
echo "$ask_res"
echo -e "\n"

# Assert keyword presence
if [[ "$ask_res" == *"AlphaOmega42"* ]]; then
    echo -e "${GREEN}Success! LLM retrieved context and correctly answered 'AlphaOmega42'.${NC}\n"
else
    echo -e "${RED}Failure! LLM failed to retrieve vault credentials context.${NC}"
    rm -rf "$TEMP_DIR"
    exit 1
fi

# 8. Database cleanups and verification of partial removal
echo -e "${CYAN}[Step 8/8] Cleaning up RAG databases and verifying 'herd db remove'...${NC}"
herd db remove "$TEMP_DIR/doc1.txt" > /dev/null
echo -e "${YELLOW}List after removing doc1.txt:${NC}"
herd db list

list_res_1=$(herd db list)
if [[ "$list_res_1" == *"doc2.txt"* && "$list_res_1" != *"doc1.txt"* ]]; then
    echo -e "${GREEN}Success! Partial removal of doc1.txt verified.${NC}\n"
else
    echo -e "${RED}Failure! Partial removal of doc1.txt failed.${NC}"
    rm -rf "$TEMP_DIR"
    exit 1
fi

herd db remove "$TEMP_DIR/doc2.txt" > /dev/null
rm -rf "$TEMP_DIR"

echo -e "${YELLOW}Verifying SQLite clean status:${NC}"
rem_check=$(sqlite3 "$sqlite_db" "SELECT COUNT(*) FROM chunks WHERE file_path LIKE '%herd_rag_test%';")
if [ "$rem_check" -eq 0 ]; then
    echo -e "${GREEN}Success! All RAG entries cleared from database.${NC}\n"
else
    echo -e "${RED}Warning: Clean failed, RAG residue remains in database (Count: ${rem_check}).${NC}"
fi

echo -e "${GREEN}=== All RAG Integrity Tests Completed Successfully! ===${NC}"
