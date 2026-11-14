# ADR 0001: Keep Streamlit as a Backup Interface

## Status

Accepted

## Decision

The Agentic RAG Knowledge Assistant will be developed as a product with a UI-independent core. The agent orchestration, retrieval, tools, context assembly, verification, and answer generation must live in reusable application services exposed through a local API.

Streamlit will be retained as a backup interface for rapid internal testing, demonstrations, and future use. It will not be allowed to become a hidden dependency of the core workflow.

## Consequences

- Core business logic must not be implemented only inside Streamlit callbacks.
- API contracts will be designed before building the primary user interface.
- A future web UI can replace or coexist with Streamlit.
- Streamlit remains useful for quick experiments and troubleshooting.
- UI-specific code should be isolated under a clearly named UI boundary.
- Tests should primarily exercise services and APIs rather than only visual behavior.

## Product-development rule

Every feature must first be expressible through an application service or API. A UI may call that capability, but it must not own the capability.
