# Proposal 45: Extract RecGPT as Hex.pm Package

**Status:** Proposed
**Date:** 2026-03-05
**Author:** RecGPT Team

## Overview

Reorganize and extract RecGPT as a production-ready Hex.pm package with clear public API boundaries, modular architecture, and comprehensive documentation. This enables external developers to use RecGPT components as dependencies in their own projects.

## Problem Statement

Currently, RecGPT is a monolithic application. To publish as a Hex.pm package, we need:

- **Clear Public API:** Distinguish between public exports and internal implementation details
- **Modular Design:** Allow users to adopt individual components (inference, embeddings, tokenization)
- **Stable Interfaces:** Design APIs that won't break between minor versions
- **Documentation:** Comprehensive module docs and usage guides for external consumers
- **Licensing & Legal:** Proper metadata, licenses, and attribution
- **Minimal Dependencies:** Keep core modules lightweight (current deps: Nx, Exla, Bumblebee, gRPC, Ecto)

## Proposed Structure for Hex.pm

Organize `lib/recgpt/` by domain and functionality, following Elixir conventions (like Ecto, Phoenix, Plug):

```
lib/recgpt/
├── recgpt.ex                # Main package entry point
├── inference/               # Inference domain
│   ├── inference.ex         # Public: Core inference API
│   ├── engine.ex            # Internal: @doc false - computation engine
│   ├── params.ex            # Internal: @doc false - parameter loading
│   ├── fuxi_linear/         # Internal: FuXi-Linear specific
│   └── examples/            # Public: Example usage patterns
├── embedding/               # Embedding domain
│   ├── embedding.ex         # Public: Core embedding operations
│   ├── cache.ex             # Public: Embedding caching layer
│   └── normalizer.ex        # Internal: @doc false - normalization helpers
├── tokenization/            # Tokenization domain
│   ├── fsq.ex              # Public: FSQ tokenizer API
│   └── encoder.ex          # Internal: @doc false - token encoding
├── checkpoint/              # Checkpoint domain
│   ├── loader.ex            # Public: Load trained models
│   ├── formats.ex           # Public: Supported checkpoint formats
│   └── import.ex            # Internal: @doc false - format converters
├── training/                # Training domain (internal)
│   ├── batching.ex          # Internal: @doc false - batch building
│   ├── loss.ex              # Internal: @doc false - loss computation
│   └── optimization.ex      # Internal: @doc false - optimizer utilities
├── core/                    # Core utilities
│   ├── artifact.ex          # Public: Model artifact types
│   ├── resource.ex          # Public: Resource validation
│   └── inspect.ex           # Public: Safe introspection utilities
├── health/                  # Health checks
│   └── check.ex             # Public: Health check API
├── grpc/                    # gRPC services (internal)
│   ├── service.ex           # Internal: @doc false - gRPC service
│   └── interceptors.ex      # Internal: @doc false - request handling
└── integration/             # Legacy/specific integrations
    ├── figgie/              # Figgie-specific data handling
    ├── clickstream.ex       # Clickstream tracking
    └── ...
```

**Public Modules** (no `@doc false`):
- Stable API contracts
- Well-documented with examples
- Backward compatibility across minor versions
- Examples embedded in module docs

**Internal Modules** (`@doc false` or `:internal` tag):
- May change between versions
- Not included in hex.pm documentation
- Documented as internal for maintainers only
- Subject to refactoring/deprecation


## Benefits

1. **Reusable Components:** External developers can depend on individual RecGPT modules
2. **Clear Contracts:** Public APIs (no `@doc false`) are stable, internal code evolves freely
3. **Elixir Convention:** Follows patterns used by Ecto, Phoenix, Plug - familiar to community
4. **Reduced Coupling:** Core modules organize by domain, not implementation details
5. **Community Contribution:** Open-source library attracts external contributors
6. **Industry Standard:** Distributed via Hex.pm with semantic versioning
7. **Documentation Excellence:** Hex.pm documentation auto-hides `@doc false` modules
8. **Simplified Structure:** No artificial `impl/` directories - organization mirrors functionality

## Implementation Phases
- Define public API surface (which modules are stable, which are internal)
- Mark internal modules with `@doc false` - no separate directories needed
- Design module deprecation strategy for legacy code
- Add @doc and @moduledoc annotations to all public modules
- Create CHANGELOG.md following keepachangelog.org format

### Phase 1: Architecture & API Design

### Phase 1: Architecture & API Design
- Define public API surface (which modules are stable, which are internal)
- Mark internal modules with `@doc false` - no separate directories needed
- Design module deprecation strategy for legacy code
- Add @doc and @moduledoc annotations to all public modules
- Create CHANGELOG.md following keepachangelog.org format

### Elixir Visibility Pattern

Unlike languages with access modifiers (public/private), Elixir uses documentation to declare intent:

```elixir
# Public API - documented, included in hex.pm docs
defmodule RecGPT.Inference do
  @doc """
  Run inference on a batch of items.
  ...
  """
  def predict(model, items), do: ...
end

# Internal implementation - not documented, excluded from hex.pm docs
defmodule RecGPT.Inference.Engine do
  @doc false
  # Users can still call this if they really need to, but:
  # - It's not documented
  # - Hex.pm doesn't show it
  # - No stability guarantees
  def compute_graph(params), do: ...
end
```

Key point: **No separate `internal/` or `impl/` directories** - visibility is declared via `@doc false`, not directory structure.

### Phase 2: Restructure Code
- Reorganize by domain (Inference, Embedding, Tokenization, etc.)
- Mark internal modules with `@doc false`
- Keep all code in the same domain folder (no internal/ subdirectory)
- Create public wrapper APIs where internal logic is complex
- Update internal module names to clarify relationships (e.g., `Inference.Engine`, `Tokenization.Encoder`)
- Update all imports to use domain-based paths
- Verify no breaking changes to documented public modules

### Phase 3: Dependency Minimization
- Profile external dependencies used by core modules
- Evaluate which dependencies are truly required in core vs supporting (gRPC, Ecto, Waffle, Explorer)
- Consider optional dependencies for heavy ML libraries (Bumblebee, Axon) if used only for specific features
- Update mix.exs with optional dependency markers
- Minimize core package dependencies while allowing users to opt-in to full functionality

### Phase 4: Documentation & Examples
- Add comprehensive module documentation with usage examples
- Create example projects in docs/ showing common use cases
- Document the stability guarantees for each module
- Write "Stability Matrix" showing which modules are stable

### Phase 5: Hex.pm Preparation
- Create hex.pm account/organization if needed
- Update mix.exs with proper package metadata
- Add CONTRIBUTING.md for external contributors
- Set up CI/CD for automated testing on merge
- Create v1.0.0 release with changelog

### Phase 6: Publication & Maintenance
- Publish to hex.pm with proper version tagging
- Set up security policy and reporting mechanisms
- Establish release schedule (e.g., monthly releases)
- Monitor hex.pm statistics and download patterns

## Public API Surface

### Stable (v1.0.0 guarantee)
- `RecGPT.Inference` - Core inference API
- `RecGPT.Embedding` - Core embedding operations
- `RecGPT.Tokenization.FSQ` - FSQ tokenizer
- `RecGPT.Checkpoint.Loader` - Model loading
- `RecGPT.Health` - Health checks

### Planned for Future Stability
- `RecGPT.Training` - Training utilities
- `RecGPT.Eval` - Evaluation metrics

### Internal (subject to change)
- All modules marked with `@doc false`
- Internal submodules (e.g., `Inference.Engine`, `Tokenization.Encoder`)
- Legacy integration modules (figgie, clickstream, grpc)
- Training and optimization utilities

## Version Management

For external packages depending on RecGPT:

```elixir
# mix.exs example
{:recgpt, "~> 1.0", only: [:runtime]}

# Will match 1.x.y but not 2.0.0 (breaking changes)
```

## Semantic Versioning

- **MAJOR** (1.0.0 → 2.0.0): Breaking API changes to public modules
- **MINOR** (1.0.0 → 1.1.0): New public features (backward compatible)
- **PATCH** (1.0.0 → 1.0.1): Bug fixes and internal improvements

## Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| Heavy ML dependencies deter adoption | Mark Bumblebee, Axon as optional dependencies |
| Public API changes break users | Semantic versioning + 3-month deprecation notice |
| Incomplete documentation | Document all public modules before v1.0 release |
| Security vulnerabilities in deps | Regular `mix audit` checks in CI/CD |
| License conflicts | Verify all dependencies compatible with chosen license |
| Inactive maintenance perception | Establish bi-weekly release cadence |
| Onboarding complexity | Provide 2-3 end-to-end example projects |

## Publishing Checklist

- [ ] Define public API surface (which modules/functions are stable)
- [ ] Add @doc to 100% of public functions
- [ ] Add @moduledoc to 100% of public modules  
- [ ] Create CHANGELOG.md with all changes documented
- [ ] Update mix.exs with package metadata (description, licenses, links)
- [ ] Create README with installation, quick start, example
- [ ] Set up GitHub release tags for version control
- [ ] Pass full test suite with >90% code coverage
- [ ] Run hexpm validation: `mix hex.build`
- [ ] Create Hex.pm account and publish initial version
- [ ] Add package to hex.pm Organizations (if applicable)
- [ ] Set up hex.pm documentation site with examples

## Decision

This proposal recommends preparing RecGPT as a production-grade Hex.pm package. By investing in clear API boundaries, minimal dependencies, and excellent documentation now, we enable:

1. **External Adoption:** Companies can use RecGPT components in production
2. **Community Growth:** Open-source model attracts contributions and feedback
3. **Credibility:** Published on Hex.pm signals production-readiness to the Elixir community
4. **Long-term Maintenance:** Public versioning ensures API stability for users

**Recommendation:** Begin with Phase 1 (API Design) to establish which modules form the stable v1.0 surface, then proceed sequentially through remaining phases before initial publication.
