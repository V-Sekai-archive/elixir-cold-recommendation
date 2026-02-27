# Architecture Summary

RecGPT in this repo: text → embeddings (Bumblebee/MPNet) and FSQ → token sequences; trie and beam search; inference in-process with Nx. API: unified gRPC+REST; REST subset implemented. [13](13_grpc_rest_api.md), [14](14_api_schemas.md). Pipeline and modules: [08](08_pipeline_reference.md), [00](00_recgpt_library.md).

**References:** [17_architecture_references.md](17_architecture_references.md).
