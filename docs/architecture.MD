# System Architecture

The Cortex Monorepo is built on three interconnected modules.

## Workflow

```mermaid
flowchart LR
    A[Streaming Data] -->|Kafka/WebSocket| B[Semantic Analyzer]
    B -->|Embeddings + Entities| C[Models Hub]
    C -->|Exported Models| D[Cortex Engine]
    D --> E[Decisions/Insights]
