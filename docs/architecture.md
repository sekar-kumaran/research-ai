# Research AI Architecture

## Runtime topology

```mermaid
flowchart TB
    subgraph Cluster[Kubernetes cluster or Docker host]
        Ingress[Ingress or port 7860]
        Backend[research-ai-backend\nFastAPI + static frontend]
        Postgres[(Postgres 16\nPVC: postgres-data)]
        LLM[Configured LLM API\nexternal or optional Ollama]
        Data[(App data\nPVC: research-ai-data)]
        Artifacts[(ML artifacts\nPVC: research-ai-artifacts)]
        Ingress --> Backend
        Backend --> Postgres
        Backend --> LLM
        Backend --> Data
        Backend --> Artifacts
    end
    External[arXiv / Crossref / OpenAlex] --> Backend
    Browser[Browser] --> Ingress
```

## Deployment responsibilities

| Component | Responsibility | Kubernetes resource |
| --- | --- | --- |
| Backend | API, frontend, agents, retrieval, persistence integration | Deployment + ClusterIP Service |
| Postgres | Users, sessions, conversations, papers, graphs | Deployment + PVC + Service |
| LLM provider | Chat completion for planning and synthesis; external by default | Provider API configured through Secret/ConfigMap; Ollama is optional |
| Artifacts | FAISS index, metadata, classifiers, clustering files | PVC or separately provisioned runtime files |
| Ingress | Optional external HTTP entrypoint | Ingress |

## Health model

- `/health` returns component and artifact status and is used for startup/liveness checks.
- `/ready` reports whether the database and at least one paper search provider are configured.
- A ready pod can still report degraded optional ML features when large artifacts are unavailable.