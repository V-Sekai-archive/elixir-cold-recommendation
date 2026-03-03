# Top-tier recommendations

Sub-proposal of the [documentation index](README.md). Elevate the library to production-grade quality: typespecs, Dialyzer, integration test, health endpoint, property tests, benchmarks, release and Docker. Checklist: [23 Quality assurance](23_quality_assurance.md).

---

## Problem or limitation

The library must be production-ready: static typing, linting, tests, health checks, and deployable artifacts. Without a defined checklist, quality and deployment vary.

---

## Proposed improvement

Make the codebase production-ready with the following improvements. Checklist: [23 Quality assurance](23_quality_assurance.md).

---

## Recommended improvements

- [x] **Typespecs and Dialyzer** - Add @spec to public functions; run mix dialyzer.
- [x] **Integration test** - Full flow: load_state to predict (see prediction_service_test).
- [x] **Health / readiness** - HTTP endpoint (e.g. port 50052) for K8s probes.
- [x] **Property-based tests** - StreamData for trie and other invariants.
- [x] **Benchmarks** - Recommendation latency via `mix recgpt.trace_predict --runs 20` or gRPC Predict (see docs/42_latency_and_performance.md).
- [x] **Release and Docker** - mix release; RecGPT.ReleaseTasks.serve(); Dockerfile.

---

## See also

- [23 Quality assurance](23_quality_assurance.md) - QA checklist and CI.
- [04 RecGPT library](04_recgpt_library.md) - Module reference.
