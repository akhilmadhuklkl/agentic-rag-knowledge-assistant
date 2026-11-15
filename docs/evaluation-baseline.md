# Evaluation baseline

Audit date: 2026-11-15

This baseline was run through the real local pipeline: fixture ingestion, vector indexing, hybrid retrieval, proximity reranking, orchestration, grounded generation, and verification.

## Multi-document benchmark

| Measure | Result |
|---|---:|
| Total cases | 9 |
| Answerable cases | 8 |
| Answerable retrieval | 100% |
| Answerable citation accuracy | 100% |
| Answer quality | 100% |
| Refusal-case accuracy | 100% |

Category results:

- Access policy: 3/3 retrieval, 3/3 answer matches
- Onboarding: 2/2 retrieval, 2/2 answer matches
- Support: 3/3 retrieval, 3/3 answer matches
- Refusal: 1/1 refusal correct; retrieval is intentionally 0/1 because unsupported questions should not cite a source

## Interpretation

The current checked-in benchmark is passing cleanly, including the newly added support domain. This is a strong local regression baseline, not proof of production-level retrieval quality: the fixture set is small, controlled, and written to be unambiguous.

The next evaluation improvement should add paraphrases, distractor documents, multi-hop questions, conflicting statements, and longer documents. Those cases will better test whether reranking and refusal behavior generalize beyond exact fixture wording.
