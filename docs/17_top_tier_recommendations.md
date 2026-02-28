# Top-tier recommendations

Sub-proposal of the [documentation index](README.md). Elevate the library to production-grade quality: typespecs, Dialyzer, integration test, health endpoint, property tests, benchmarks, release and Docker. Checklist: [18 Quality assurance](18_quality_assurance.md).

---

## Goal

Make the codebase production-ready with static typing, linting, tests, health checks, and deployable artifacts.

---

## Recommended improvements

- [x] **Typespecs and Dialyzer** - Add @spec to public functions; run mix dialyzer.
- [x] **Integration test** - Full flow: load_state to predict (see prediction_service_test).
- [x] **Health / readiness** - HTTP endpoint (e.g. port 50052) for K8s probes.
- [x] **Property-based tests** - StreamData for trie and other invariants.
- [x] **Benchmarks** - Benchee for Serve.recommend/3; run mix run bench/recgpt_serve_bench.exs.
- [x] **Release and Docker** - mix release; RecGPT.ReleaseTasks.serve(); Dockerfile.

---

## See also

- [18 Quality assurance](18_quality_assurance.md) - QA checklist and CI.
- [04 RecGPT library](04_recgpt_library.md) - Module reference.