#!/usr/bin/env bash
# Call RecGPT Predict via grpcurl. Run from repo root. Server must be up: mix recgpt.serve
# Usage: ./scripts/grpcurl_recommend.sh [context_item_ids...]   e.g.  ./scripts/grpcurl_recommend.sh 0 1 2
set -e
CONTEXT_IDS="${*:-0}"
MAX_RESULTS="${MAX_RESULTS:-5}"
PORT="${RECGPT_GRPC_PORT:-50051}"
PROTO="${PROTO:-priv/proto/recgpt/v1/recommendation.proto}"

# Build JSON array from space-separated IDs
IDS_JSON="[$(echo "$CONTEXT_IDS" | sed 's/ /, /g')]"
BODY="{\"context_item_ids\": $IDS_JSON, \"max_results\": $MAX_RESULTS}"

grpcurl -plaintext \
  -proto "$PROTO" \
  -d "$BODY" \
  "localhost:$PORT" \
  recgpt.v1.PredictionService/Predict
