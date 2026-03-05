# Scripts Directory

This directory contains utility scripts for the RecGPT project.

## Figgie Arbitrage Scripts

- `generate_figgie_data.exs` - Generate training data by simulating Figgie games
- `test_bayesian.exs` - Test Bayesian probability calculations for Figgie

## Development & Debugging Scripts

- `debug_decode_scoring.exs` - Debug decode scoring logic
- `embed_one.py` - Test single embedding generation
- `fix_freeze_doc.exs` - Fix documentation freezing issues
- `fix_parity.exs` - Fix parity testing issues
- `profile_predict.exs` - Profile prediction performance

## Data Processing Scripts

- `dump_canonical_to_sqlite.py` - Export canonical data to SQLite
- `reproduce_embeddings.py` - Reproduce embedding calculations

## Infrastructure Scripts

- `compile_torchx_cuda.cmd` - Compile TorchX with CUDA support
- `run_with_exla_env.sh` - Run with EXLA environment variables
- `setup_exla_libs.sh` - Setup EXLA libraries

## API Testing Scripts

- `grpcurl_predict_request.json` - Sample gRPC predict request
- `grpcurl_recommend.sh` - Test gRPC recommendation endpoint

## Usage

Run Elixir scripts with: `elixir scripts/script_name.exs`
Run Python scripts with: `python scripts/script_name.py`
Run shell scripts with: `./scripts/script_name.sh`