# Project Scope

## Objective

Build a local-first Agentic RAG Knowledge Assistant that searches trusted documents, selects appropriate tools, verifies evidence, and returns grounded answers with citations.

## MVP user experience

The user submits a natural-language question through a local API or simple web interface. The assistant investigates the question using available knowledge sources and returns:

- A concise answer
- Source citations
- The tools used
- Verification status
- An uncertainty/refusal response when reliable evidence is unavailable

## MVP capabilities

### In scope

- Ingest local PDF, Markdown, and text documents
- Preserve document, page, section, and chunk metadata
- Create and store embeddings locally
- Perform semantic document search
- Support keyword plus semantic retrieval where useful
- Use an agent orchestrator to plan and select tools
- Support document search and calculator tools initially
- Assemble and rank retrieved evidence
- Generate answers from supplied evidence only
- Validate citations against retrieved source metadata
- Detect unsupported answers and refuse safely
- Expose a local API
- Provide a basic local chat interface
- Add structured logs and request tracing
- Add unit, integration, and evaluation tests

### Deferred

- Cloud deployment
- Production hosting and domains
- Multi-user account management
- Billing and usage management
- Complex external enterprise integrations
- Autonomous write actions against external systems
- Long-running background research jobs

## Non-functional requirements

- Local development must work with a documented setup.
- Tool calls must have input validation, timeouts, and bounded retries.
- Retrieved document content must be treated as evidence, not instructions.
- Every citation must refer to preserved source metadata.
- Secrets must remain outside version control.
- Unsupported questions must not be answered confidently.
- The main workflow must be observable and testable.

## Initial tool contract

```text
document_search(query, filters, top_k) -> ranked evidence with citations
calculator(expression) -> validated calculation result
```

Web search and external APIs will be added only after the local document workflow is reliable.

## Definition of done for the MVP

The MVP is complete when a local user can ask a question about the supplied documents and receive a verified, cited answer; the system can explain when evidence is missing; the API and UI work locally; automated tests pass; and an evaluation dataset produces measurable results.
