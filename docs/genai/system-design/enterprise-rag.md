# Enterprise RAG System Design

> **Design production-grade knowledge base systems with document processing, access control, and scale**

---

## Learning Objectives

By the end of this module, you will be able to:

- **Architect enterprise RAG systems** with multi-tenant support and access control
- **Design document processing pipelines** for diverse document types and formats
- **Implement hybrid search strategies** combining vector, keyword, and structured search
- **Build scalable retrieval systems** handling millions of documents
- **Address enterprise concerns** including security, compliance, and auditability

---

## Enterprise RAG Architecture

### High-Level System Architecture

```mermaid
graph TB
    subgraph "Ingestion Layer"
        Sources[Document Sources]
        Connectors[Source Connectors]
        Processing[Document Processing]
        Chunking[Chunking Service]
        Embedding[Embedding Service]
    end

    subgraph "Storage Layer"
        DocStore[(Document Store)]
        VectorDB[(Vector Database)]
        MetadataDB[(Metadata Store)]
        GraphDB[(Knowledge Graph)]
    end

    subgraph "Query Layer"
        QueryAPI[Query API]
        QueryProcess[Query Processor]
        Retriever[Hybrid Retriever]
        Reranker[Reranker]
        ContextBuilder[Context Builder]
    end

    subgraph "Generation Layer"
        PromptMgr[Prompt Manager]
        LLMGateway[LLM Gateway]
        CitationGen[Citation Generator]
        Safety[Safety Service]
    end

    subgraph "Access Control"
        AuthN[Authentication]
        AuthZ[Authorization]
        ACL[ACL Engine]
        Audit[Audit Logger]
    end

    subgraph "Operations"
        Monitoring[Monitoring]
        Feedback[Feedback Loop]
        Evaluation[Evaluation]
    end

    Sources --> Connectors --> Processing --> Chunking --> Embedding
    Embedding --> VectorDB
    Processing --> DocStore
    Chunking --> MetadataDB
    Processing --> GraphDB

    QueryAPI --> AuthN --> AuthZ --> QueryProcess
    QueryProcess --> Retriever
    Retriever --> VectorDB
    Retriever --> MetadataDB
    Retriever --> GraphDB
    Retriever --> ACL --> Reranker --> ContextBuilder

    ContextBuilder --> PromptMgr --> LLMGateway --> CitationGen --> Safety

    LLMGateway --> Monitoring
    Safety --> Audit
    CitationGen --> Feedback
```

### Core Components

| Component | Purpose | Enterprise Considerations |
|-----------|---------|--------------------------|
| **Source Connectors** | Ingest from various systems | SharePoint, Confluence, S3, databases |
| **Document Processing** | Extract and structure content | OCR, table extraction, metadata |
| **Chunking Service** | Split documents optimally | Semantic boundaries, overlap |
| **Embedding Service** | Generate vector representations | Batch processing, model versioning |
| **Hybrid Retriever** | Multi-strategy search | Vector + keyword + structured |
| **ACL Engine** | Access control enforcement | Real-time permission checking |
| **Citation Generator** | Source attribution | Compliance, trustworthiness |

---

## Document Processing Pipeline

### Processing Architecture

```mermaid
graph TB
    subgraph "Document Intake"
        Upload[Upload API]
        Crawler[Web Crawler]
        Connector[System Connectors]
        Stream[Event Stream]
    end

    subgraph "Pre-Processing"
        TypeDetect[Type Detection]
        Extraction[Content Extraction]
        OCR[OCR Service]
        TableExtract[Table Extraction]
    end

    subgraph "Content Processing"
        Clean[Text Cleaning]
        Language[Language Detection]
        Entity[Entity Extraction]
        Structure[Structure Analysis]
    end

    subgraph "Chunking"
        Strategy[Strategy Selection]
        Chunk[Chunking Engine]
        Overlap[Overlap Manager]
        Metadata[Metadata Enrichment]
    end

    subgraph "Vectorization"
        Batch[Batch Manager]
        Embed[Embedding Model]
        Index[Index Builder]
    end

    subgraph "Storage"
        DocDB[(Documents)]
        VecDB[(Vectors)]
        MetaDB[(Metadata)]
    end

    Upload --> TypeDetect
    Crawler --> TypeDetect
    Connector --> TypeDetect
    Stream --> TypeDetect

    TypeDetect --> Extraction
    TypeDetect -->|Scanned| OCR
    OCR --> Extraction
    Extraction --> TableExtract --> Clean

    Clean --> Language --> Entity --> Structure

    Structure --> Strategy --> Chunk --> Overlap --> Metadata

    Metadata --> Batch --> Embed --> Index

    Index --> VecDB
    Metadata --> MetaDB
    Extraction --> DocDB
```

### Document Type Handling

| Document Type | Extraction Method | Chunking Strategy | Considerations |
|---------------|-------------------|-------------------|----------------|
| **PDF** | PyMuPDF, pdfplumber | Page-aware, heading-based | Tables, images, OCR fallback |
| **Word/DOCX** | python-docx | Section/heading-based | Styles, tables, embedded objects |
| **HTML** | BeautifulSoup | DOM-aware, semantic tags | Navigation removal, content extraction |
| **Markdown** | Unified parser | Header-based hierarchy | Code blocks, links |
| **PowerPoint** | python-pptx | Slide-based | Speaker notes, visual context |
| **Excel** | openpyxl, pandas | Sheet/table-based | Formula context, named ranges |
| **Email** | email parser | Thread-aware | Attachments, conversation threading |

### Chunking Strategies

```mermaid
graph TB
    subgraph "Fixed-Size Chunking"
        FS[Fixed Size]
        FS --> FS1[500 tokens]
        FS --> FS2[1000 tokens]
        FS --> FS3[With overlap]
    end

    subgraph "Semantic Chunking"
        SC[Semantic]
        SC --> SC1[Sentence boundaries]
        SC --> SC2[Paragraph boundaries]
        SC --> SC3[Section boundaries]
    end

    subgraph "Recursive Chunking"
        RC[Recursive]
        RC --> RC1[Split by headers]
        RC --> RC2[Then paragraphs]
        RC --> RC3[Then sentences]
    end

    subgraph "Document-Aware"
        DA[Document-Aware]
        DA --> DA1[Preserve tables]
        DA --> DA2[Keep code blocks]
        DA --> DA3[Maintain hierarchy]
    end
```

::: info Chunking Best Practices
- **Chunk size**: 256-512 tokens for precise retrieval, 512-1024 for context-rich
- **Overlap**: 10-20% overlap to maintain context continuity
- **Semantic boundaries**: Prefer splitting at natural breaks (paragraphs, sections)
- **Metadata preservation**: Attach source, page, section info to each chunk
:::

### Chunking Parameters Comparison

| Strategy | Chunk Size | Overlap | Use Case | Trade-offs |
|----------|------------|---------|----------|------------|
| **Small chunks** | 256 tokens | 50 tokens | Precise Q&A | May miss context |
| **Medium chunks** | 512 tokens | 100 tokens | General RAG | Balanced |
| **Large chunks** | 1024 tokens | 200 tokens | Summarization | Lower precision |
| **Semantic** | Variable | Natural | Technical docs | Complex implementation |

---

## Hybrid Search Architecture

### Multi-Strategy Retrieval

```mermaid
graph TB
    Query[User Query] --> QueryAnalyzer[Query Analyzer]

    QueryAnalyzer --> VectorSearch[Vector Search]
    QueryAnalyzer --> KeywordSearch[Keyword Search]
    QueryAnalyzer --> StructuredSearch[Structured Search]
    QueryAnalyzer --> GraphSearch[Graph Search]

    subgraph "Vector Search"
        VectorSearch --> Embed[Embed Query]
        Embed --> ANN[ANN Search]
        ANN --> VecResults[Vector Results]
    end

    subgraph "Keyword Search"
        KeywordSearch --> BM25[BM25 Scoring]
        BM25 --> KeyResults[Keyword Results]
    end

    subgraph "Structured Search"
        StructuredSearch --> Filters[Metadata Filters]
        Filters --> SQL[SQL Query]
        SQL --> StructResults[Structured Results]
    end

    subgraph "Graph Search"
        GraphSearch --> EntityMatch[Entity Matching]
        EntityMatch --> GraphTraverse[Graph Traversal]
        GraphTraverse --> GraphResults[Graph Results]
    end

    VecResults --> Fusion[Result Fusion]
    KeyResults --> Fusion
    StructResults --> Fusion
    GraphResults --> Fusion

    Fusion --> Rerank[Reranking]
    Rerank --> TopK[Top-K Results]
```

### Fusion Strategies

| Strategy | Description | Pros | Cons |
|----------|-------------|------|------|
| **Reciprocal Rank Fusion (RRF)** | Combine by reciprocal ranks | Simple, effective | Ignores score magnitude |
| **Weighted Average** | Weight scores by source | Tunable | Requires score normalization |
| **Learn-to-Rank** | ML model for fusion | Optimal | Training data needed |
| **Cascade** | Sequential filtering | Efficient | May miss good results |

### RRF Implementation

```python
def reciprocal_rank_fusion(result_lists, k=60):
    """
    Combine multiple ranked lists using RRF.
    k: smoothing constant (typically 60)
    """
    fused_scores = {}

    for result_list in result_lists:
        for rank, doc_id in enumerate(result_list):
            if doc_id not in fused_scores:
                fused_scores[doc_id] = 0
            fused_scores[doc_id] += 1 / (k + rank + 1)

    # Sort by fused score
    return sorted(fused_scores.items(),
                  key=lambda x: x[1],
                  reverse=True)
```

---

## Access Control System

### Multi-Tenant Architecture

```mermaid
graph TB
    subgraph "Identity Layer"
        SSO[SSO/SAML]
        OAuth[OAuth 2.0]
        API[API Keys]
    end

    subgraph "Authorization Layer"
        RBAC[Role-Based AC]
        ABAC[Attribute-Based AC]
        PolicyEngine[Policy Engine]
    end

    subgraph "Document Layer"
        DocACL[Document ACLs]
        Inheritance[Permission Inheritance]
        Groups[Group Membership]
    end

    subgraph "Query Layer"
        QueryFilter[Query-Time Filtering]
        ResultFilter[Result Filtering]
        Redaction[Content Redaction]
    end

    SSO --> PolicyEngine
    OAuth --> PolicyEngine
    API --> PolicyEngine

    RBAC --> PolicyEngine
    ABAC --> PolicyEngine

    PolicyEngine --> QueryFilter
    DocACL --> QueryFilter
    Inheritance --> QueryFilter
    Groups --> QueryFilter

    QueryFilter --> ResultFilter --> Redaction
```

### Permission Models

| Model | Description | Use Case | Complexity |
|-------|-------------|----------|------------|
| **RBAC** | Role-based access | Simple org structures | Low |
| **ABAC** | Attribute-based | Complex policies | Medium |
| **ACL** | Per-document permissions | Fine-grained control | High |
| **Hierarchical** | Folder inheritance | File systems | Medium |
| **Hybrid** | Combination | Enterprise | High |

### Query-Time Access Control

::: info Performance Consideration
Access control filtering at query time can significantly impact latency. Pre-compute permission sets where possible, and use efficient filtering strategies.
:::

```mermaid
sequenceDiagram
    participant U as User
    participant Q as Query Service
    participant A as ACL Service
    participant V as Vector DB
    participant R as Reranker

    U->>Q: Query + Auth Token
    Q->>A: Get User Permissions
    A->>A: Resolve Groups
    A->>Q: Permission Set

    Q->>V: Vector Search + ACL Filter
    Note over V: Pre-filter by<br/>permitted documents

    V->>Q: Filtered Results
    Q->>R: Rerank Results
    R->>Q: Final Results
    Q->>U: Response with Citations
```

### ACL Implementation Approaches

| Approach | Description | Pros | Cons |
|----------|-------------|------|------|
| **Pre-filter** | Filter before search | Fast queries | Index per permission set |
| **Post-filter** | Filter after search | Simple index | Over-fetch needed |
| **Hybrid** | Coarse pre + fine post | Balanced | Implementation complexity |
| **Materialized views** | Pre-computed per user/group | Fastest queries | Storage cost, staleness |

---

## Scaling Strategies

### Horizontal Scaling Architecture

```mermaid
graph TB
    subgraph "Load Balancing"
        LB[Load Balancer]
        LB --> Q1[Query Node 1]
        LB --> Q2[Query Node 2]
        LB --> QN[Query Node N]
    end

    subgraph "Sharded Vector Store"
        Q1 --> S1[Shard 1]
        Q1 --> S2[Shard 2]
        Q2 --> S1
        Q2 --> S2
        QN --> SN[Shard N]
    end

    subgraph "Ingestion Pipeline"
        Kafka[Kafka]
        Kafka --> W1[Worker 1]
        Kafka --> W2[Worker 2]
        Kafka --> WN[Worker N]
        W1 --> S1
        W2 --> S2
        WN --> SN
    end

    subgraph "Coordination"
        ZK[Zookeeper/etcd]
        ZK --> S1
        ZK --> S2
        ZK --> SN
    end
```

### Scaling Dimensions

| Dimension | Strategy | Implementation |
|-----------|----------|----------------|
| **Documents** | Horizontal sharding | Shard by tenant, date, or hash |
| **Queries** | Read replicas | Multiple query nodes per shard |
| **Embeddings** | Batch processing | Async ingestion queue |
| **LLM Calls** | Rate limiting + caching | Semantic cache, request coalescing |
| **Tenants** | Isolation | Dedicated shards or namespaces |

### Caching Strategy

```mermaid
graph TB
    Query[Query] --> QCache{Query Cache}
    QCache -->|Hit| Response[Response]
    QCache -->|Miss| SemCache{Semantic Cache}

    SemCache -->|Hit| Response
    SemCache -->|Miss| Process[Full Processing]

    Process --> EmbedCache{Embedding Cache}
    EmbedCache -->|Hit| Retrieve[Retrieval]
    EmbedCache -->|Miss| Embed[Embed Query]
    Embed --> EmbedCache
    Embed --> Retrieve

    Retrieve --> LLMCache{LLM Cache}
    LLMCache -->|Hit| Response
    LLMCache -->|Miss| LLM[LLM Generation]
    LLM --> LLMCache
    LLM --> Response
```

| Cache Level | Hit Rate | Latency Savings | Freshness |
|-------------|----------|-----------------|-----------|
| **Query (exact)** | 10-20% | 95% | Immediate |
| **Semantic** | 30-50% | 90% | Configurable |
| **Embedding** | 80%+ | 50ms | Stable |
| **LLM Response** | 20-40% | 80% | Configurable |

---

## Enterprise Considerations

### Compliance and Audit

```mermaid
graph TB
    subgraph "Audit Trail"
        Query[Query] --> Log[Audit Logger]
        Log --> Who[Who: User ID]
        Log --> What[What: Query, Results]
        Log --> When[When: Timestamp]
        Log --> Why[Why: Purpose/Context]
    end

    subgraph "Data Governance"
        Retention[Retention Policies]
        Classification[Data Classification]
        Lineage[Data Lineage]
        PII[PII Detection]
    end

    subgraph "Compliance"
        GDPR[GDPR]
        HIPAA[HIPAA]
        SOC2[SOC2]
        SOX[SOX]
    end

    Log --> Retention
    Log --> Classification
    Classification --> GDPR
    Classification --> HIPAA
    Retention --> SOC2
    Lineage --> SOX
```

### Enterprise Features Checklist

| Feature | Description | Implementation |
|---------|-------------|----------------|
| **SSO Integration** | Enterprise identity | SAML, OIDC |
| **Audit Logging** | Query and access logs | Structured logging, retention |
| **Data Classification** | Sensitivity levels | Metadata tagging |
| **Retention Policies** | Automatic deletion | TTL, archival |
| **Encryption** | At rest and in transit | TLS, AES-256 |
| **Backup/Recovery** | Disaster recovery | Regular snapshots |
| **SLA Monitoring** | Uptime and performance | Prometheus, alerts |

---

## Interview Q&A

### Q1: How do you handle documents with different access levels in RAG?

**Answer:**
"I implement a multi-layer access control strategy:

1. **Index-time tagging**: Each document chunk gets ACL metadata (permitted users, groups, roles)

2. **Query-time filtering**: Two approaches:
   - **Pre-filter**: Include ACL predicates in vector search (e.g., `permitted_groups IN user.groups`)
   - **Post-filter**: Over-fetch results, then filter by permissions

3. **Hybrid approach** for scale:
   - Coarse pre-filtering by department/tenant (reduces search space)
   - Fine post-filtering for specific document permissions

4. **Caching**: Cache permission sets per user with short TTL to handle permission changes

5. **Audit**: Log all queries with user context for compliance

For very sensitive environments, I'd consider separate indexes per permission level to ensure complete isolation."

### Q2: How do you optimize retrieval quality in enterprise RAG?

**Answer:**
"I use a multi-stage retrieval pipeline:

**Stage 1: Query Enhancement**
- Query expansion using synonyms and related terms
- Query decomposition for complex questions
- Intent classification to route to appropriate sources

**Stage 2: Hybrid Retrieval**
- Vector search for semantic similarity
- BM25 for keyword matching
- Metadata filters for structured constraints
- Fusion using Reciprocal Rank Fusion

**Stage 3: Reranking**
- Cross-encoder reranking (e.g., Cohere, BGE)
- Diversity injection to avoid redundant results
- Recency weighting for time-sensitive queries

**Stage 4: Context Optimization**
- Parent-child retrieval: retrieve chunks, expand to parent context
- Lost-in-the-middle mitigation: order by relevance, not position
- Compression: summarize verbose contexts

I measure quality using retrieval metrics (MRR, NDCG) and end-to-end evaluation with LLM-as-judge."

### Q3: How would you scale a RAG system to handle millions of documents?

**Answer:**
"I design for horizontal scalability at each layer:

**Ingestion**:
- Async processing via message queue (Kafka)
- Parallel workers for embedding generation
- Batch embedding (reduce API calls)
- Incremental indexing (update only changed documents)

**Storage**:
- Sharded vector database (by tenant or document hash)
- Index partitioning for large collections
- Tiered storage (hot/warm/cold) based on access patterns

**Query**:
- Query routing to relevant shards
- Caching at multiple levels (query, embedding, LLM)
- Read replicas for high-traffic tenants
- Rate limiting per tenant

**Specific numbers**:
- 1M documents: Single node Qdrant/Pinecone
- 10M documents: Sharded deployment, 3-5 shards
- 100M+ documents: Distributed cluster, tenant isolation

Key optimizations:
- Quantization (reduce vector size by 4x)
- HNSW parameter tuning (ef_construction, M)
- Product quantization for memory efficiency"

### Q4: How do you ensure answer quality and prevent hallucinations?

**Answer:**
"I implement multiple layers of quality assurance:

**Retrieval Quality**:
- Relevance thresholds (reject low-scoring retrievals)
- Source diversity requirements
- Citation extraction and verification

**Generation Quality**:
- Grounded generation prompts ('only answer based on provided context')
- Temperature 0 for factual responses
- Structured output formats where possible

**Post-Generation Checks**:
- Citation verification (does the answer match sources?)
- Consistency checking (multiple generations agree?)
- Fact-checking against structured data

**Confidence Signaling**:
- 'I don't have information about X'
- 'Based on available documents...'
- Confidence scores exposed to users

**Continuous Improvement**:
- User feedback collection
- Regular human evaluation sampling
- Automated regression testing on known Q&A pairs

For high-stakes domains (legal, medical), I add human-in-the-loop review for uncertain responses."

---

## Trade-off Analysis

| Decision | Option A | Option B | Recommendation |
|----------|----------|----------|----------------|
| **Chunking Size** | Small (precise) | Large (contextual) | Medium with overlap |
| **Search Strategy** | Vector only (semantic) | Hybrid (comprehensive) | Hybrid for enterprise |
| **Access Control** | Pre-filter (fast) | Post-filter (flexible) | Hybrid approach |
| **Embedding Model** | Small (fast) | Large (accurate) | Large for quality-critical |
| **Caching** | Aggressive (fast) | Conservative (fresh) | Semantic cache with TTL |
| **Reranking** | Skip (fast) | Cross-encoder (accurate) | Always for enterprise |

---

## Navigation

| Previous | Next |
|----------|------|
| [Chatbot Design](./chatbot-design) | [Code Assistant](./code-assistant) |

---

## Sources

- Lewis, P. et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*. https://arxiv.org/abs/2005.11401
- Gao, L. et al. (2023). *Retrieval-Augmented Generation for Large Language Models: A Survey*. https://arxiv.org/abs/2312.10997
- Liu, N. et al. (2023). *Lost in the Middle: How Language Models Use Long Contexts*. https://arxiv.org/abs/2307.03172
- Pinecone. (2024). *Vector Database Scaling Guide*. https://docs.pinecone.io/guides/
- Weaviate. (2024). *Hybrid Search Documentation*. https://weaviate.io/developers/weaviate
- LlamaIndex. (2024). *Advanced RAG Techniques*. https://docs.llamaindex.ai/en/stable/optimizing/
