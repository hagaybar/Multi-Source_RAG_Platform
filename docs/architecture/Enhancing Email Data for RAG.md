

# **Architecting the Optimal Email RAG System: A Comprehensive Technical Report (2025)**

## **1\. Introduction: The Evolution of Retrieval-Augmented Generation in Enterprise Communication**

The integration of corporate email archives into Retrieval-Augmented Generation (RAG) architectures represents a definitive frontier in the field of Unstructured Data Management. As organizations transition from static knowledge bases to dynamic, conversational intelligence, the "State of the Art" in 2025 has fundamentally shifted. We have moved beyond the initial paradigms of simple vector-based retrieval toward sophisticated, hybrid architectures known as **Agentic GraphRAG**. This evolution addresses the specific, intractable challenges posed by email data: its high noise-to-signal ratio, its intricate nested structures, and its deep reliance on temporal and social context.1

This report provides an exhaustive technical analysis of the architectural requirements for building an optimal Email Data Module for a multipurpose RAG platform. It synthesizes current research to detail advanced preprocessing pipelines, thread-aware chunking strategies, hybrid retrieval algorithms, and the emerging dominance of Knowledge Graph integration.

### **1.1 The Unique Physics of Email Data**

Email data is distinct from the clean, prose-based documentation that early RAG systems were designed to handle. It functions less like a static library and more like a stream of consciousness, heavily dependent on metadata and interpersonal relationships. A query such as, "What was the decision regarding the Q3 marketing budget?" cannot be answered by semantic similarity alone. The system must understand the entity "Q3," the concept of "marketing budget," and, crucially, the directional nature of the communication—who asked, who answered, and when the decision was finalized.

Standard "naive RAG" approaches, which treat emails as isolated documents, invariably fail in this domain. They suffer from **context fragmentation**, where the answer to a question is separated from the question itself by the chunking process, and **semantic drift**, where the inclusion of signatures, disclaimers, and repetitive reply headers dilutes the vector representation of the core message.3

The industry consensus for 2025 emphasizes that a production-grade Email RAG system must act as an "Agentic" orchestrator. It must possess the capability to plan retrieval strategies, utilizing "GraphRAG" to navigate the entity relationships of the organization while employing vector search for semantic nuance. This report will rigorously dismantle the "naive" approach and reconstruct a high-performance architecture layer by layer.5

---

## **2\. Data Engineering and Ingestion Pipelines**

The integrity of a RAG system is deterministic based on the quality of its input data. For email, the ingestion layer must function as a sophisticated ETL (Extract, Transform, Load) pipeline capable of normalizing the arcane complexities of internet mail standards.

### **2.1 Advanced MIME Parsing and Normalization**

The raw format of email, defined by MIME (Multipurpose Internet Mail Extensions) standards, is a hierarchical tree structure that mixes plain text, HTML, and binary attachments. Navigating this tree requires recursive parsing strategies that go beyond simple text extraction.

#### **2.1.1 Recursive Tree Traversal and Content Prioritization**

Standard parsing libraries, such as Python's email package, provide access to the MIME structure but require significant logic to identify the "true" content payload.7 A robust pipeline must implement a priority queue for content extraction. While text/plain parts offer semantic clarity and lower token counts, they often lack the structural cues—such as bolding, lists, and headers—that Large Language Models (LLMs) utilize to understand emphasis. Conversely, text/html parts contain this structure but are laden with markup noise.

The optimal strategy involves a **hybrid parsing approach**:

1. **Traversal:** The parser walks the MIME tree to identify all content parts.  
2. **Selection:** If a text/html part exists, it is selected as the primary source.  
3. **Transformation:** The HTML is not stripped to plain text; rather, it is converted to Markdown using sophisticated tools like html2text or BeautifulSoup.9 Markdown retains the hierarchical signals (e.g., \# Heading, \*emphasis\*) that are native to LLM training data, thereby preserving the semantic weight of the content without the token overhead of HTML tags.10

#### **2.1.2 Encoding Normalization and Corruption Handling**

A frequent failure mode in email ingestion is the mishandling of character encodings. Emails frequently contain a mix of utf-8, iso-8859-1, and legacy windows-1252 encodings, often within the same thread. Furthermore, the Content-Transfer-Encoding header dictates how bytes are mapped to characters (e.g., quoted-printable or base64).

Blindly decoding text without strictly adhering to these headers results in artifact generation (e.g., \=20 for spaces), which degrades embedding quality. Modern pipelines leverage libraries like unstructured.io, which provide partition\_email functions. These functions abstract the complexity of normalization, automatically detecting encoding schemes and separating metadata (headers) from the narrative content.11 This ensures that the vector database receives clean, normalized Unicode text, free from transport-layer artifacts.

### **2.2 Structure-Aware Noise Reduction**

Once the payload is extracted, it must be cleaned. Email is arguably the "noisiest" data source in the enterprise, with a significant percentage of tokens dedicated to non-semantic content: signatures, legal disclaimers, and reply headers.

#### **2.2.1 The Signature and Disclaimer Problem**

If an organization employs a standard 200-word legal disclaimer, and this disclaimer is indexed for every email, the vector space becomes warped. Queries will cluster based on the disclaimer text rather than the unique message content. This phenomenon, known as **Vector Collapse**, results in retrieval systems returning hundreds of irrelevant emails simply because they all share the same high-density boilerplate text.13

Traditional regex-based filtering (looking for delimiters like \-- or Best regards) is computationally efficient but brittle, failing across different languages and email clients.14

#### **2.2.2 Machine Learning Classification for Segmentation**

The 2025 standard for cleaning involves **Segment Classification**. Instead of relying on heuristics, the ingestion pipeline employs lightweight transformer models (e.g., fine-tuned DistilBERT) to classify text segments.

| Method | Pros | Cons | Recommended Use Case |
| :---- | :---- | :---- | :---- |
| **Regex / Heuristics** | Extremely fast; Zero inference cost. | Brittle; Fails on non-standard formats. | First-pass filtering of obvious delimiters. |
| **Line Classification (ML)** | High accuracy (\>95%); Adaptable. | Higher computational cost; Needs training data. | Second-pass for identifying complex signatures/disclaimers. |
| **Visual Layout Analysis** | Can detect signatures via spatial layout. | Very slow; Requires rendering email to image. | Specialized use cases (e.g., scanned PDFs). |

Models trained on datasets like the Signature Detection corpus on Hugging Face can identify signature blocks based on semantic and structural features, distinct from the body text.16 This allows for the surgical removal of noise while preserving the integrity of the message.

#### **2.2.3 Handling Quoted Replies**

Perhaps the most difficult noise source is the "Quoted Reply"—the accumulation of previous messages at the bottom of an email. Indexing this creates massive duplication; a thread with 10 emails might index the first message 10 times.

Advanced pipelines utilize **Text Overlap Analysis**. Algorithms such as **MinHash** or **Levenshtein Distance** are used to fingerprint the body of the current email and compare it against the In-Reply-To parent. If a block of text in the current email shares high similarity with the parent's body, it is identified as a quote and stripped.18 This "Deduplication at Ingestion" strategy ensures that each email in the vector index represents only its *unique contribution* to the conversation.19

---

## **3\. Thread Reconstruction and Context Preservation**

A fundamental error in early email RAG systems was the treatment of emails as atomic, independent documents. In reality, an email is a node in a directed graph (the thread). The semantic meaning of "I approve this" is entirely dependent on the parent node asking "Can we proceed?"

### **3.1 Algorithmic Threading**

To preserve context, the ingestion pipeline must reconstruct threads before chunking.

1. **Header-Based Reconstruction:** The primary method involves tracing the Message-ID, In-Reply-To, and References headers. These form a linked list or tree structure that defines the conversation history.20  
2. **Subject-Based Normalization:** As a fallback, systems often group by normalized subject lines (stripping prefixes like Re:, Fwd:). While less precise, this captures threads where headers have been stripped by aggressive forwarding clients.22

### **3.2 The "Parent-Child" Indexing Pattern**

The optimal architecture for serving this threaded data is the **Parent-Child (or Small-to-Big) Retrieval Pattern**.

* **The Parent Document:** The system stores the *entire thread* (or a significantly large window of it) as a single document in a document store. This serves as the "Context Source."  
* **The Child Chunks:** The system splits the individual emails within that thread into small, granular chunks (e.g., single paragraphs or sentences). These are embedded and stored in the Vector Database.  
* **The Link:** Each child chunk retains a metadata reference to its Parent ID.

**Retrieval Mechanic:** When a user's query matches a specific Child Chunk (e.g., a specific sentence about "budget cuts"), the system does not return just that sentence. Instead, it uses the Parent ID to retrieve the *full conversation window* surrounding that sentence.23 This provides the LLM with the necessary "before and after" context to generate a coherent response, resolving the fragmentation issue inherent in standard RAG.24

---

## **4\. Advanced Chunking Strategies: Beyond Fixed-Size Splitting**

Once the thread is reconstructed and cleaned, it must be divided into manageable segments for embedding. The choice of chunking strategy is one of the most significant variables impacting retrieval performance.25

### **4.1 The Failure of Fixed-Size Chunking**

Standard "Fixed-Size Chunking" (e.g., splitting every 512 tokens with a 50-token overlap) is generally unsuited for email. It blindly cuts through sentences, separates questions from their answers, and fails to respect the natural boundaries of the communication. In a conversational dataset, a split that occurs in the middle of a logical thought results in a "hallucination hazard," where the LLM attempts to complete the thought without the actual data.25

### **4.2 Semantic Chunking**

For 2025, **Semantic Chunking** is the preferred methodology. This approach leverages the embedding model itself to determine boundaries. The algorithm processes the text sentence by sentence, calculating the cosine similarity between the current sentence and the next.

* **Mechanism:** If the similarity score remains high, the sentences are grouped into the same chunk. If the similarity drops below a predefined threshold (indicating a shift in topic), a breakpoint is introduced, and a new chunk begins.

This ensures that a single email covering multiple topics (e.g., "Project Status," "HR Complaint," "Lunch Plans") is split into three distinct, semantically coherent chunks, rather than arbitrary blocks of text.26

### **4.3 Hierarchical and Recursive Chunking**

For complex, lengthy threads, **Hierarchical Chunking** is employed. This strategy uses a multi-level delimiters approach:

1. **Level 1:** Split by Thread (Conversation).  
2. **Level 2:** Split by Message (Sender/Time).  
3. **Level 3:** Split by Paragraph (\\n\\n).  
4. **Level 4:** Split by Sentence (.).

Tools like LangChain's RecursiveCharacterTextSplitter implement this logic, attempting to keep the largest semantic units (paragraphs) intact before resorting to smaller splits.28 This preserves the logical flow of the email, ensuring that embeddings represent complete thoughts.

### **4.4 Sliding Window and Time-Based Grouping**

To handle extremely long histories, **Sliding Window** chunking is essential. By creating overlapping chunks (e.g., 500 tokens with a 20% overlap), the system ensures that context at the boundaries is not lost. Furthermore, **Time-Based Grouping** can be applied, where chunks are defined not by size but by temporal windows (e.g., "All correspondence from Week 42"). This supports temporal queries ("How did the negotiation evolve last week?") which are common in enterprise settings.30

---

## **5\. Embeddings and Vector Space: The Semantic Core**

The "brain" of the RAG system is the embedding model, which translates text into high-dimensional vectors. The specific characteristics of email—informal, multilingual, and domain-specific—demand careful model selection.

### **5.1 State-of-the-Art Embedding Models (2025)**

The MTEB (Massive Text Embedding Benchmark) leaderboard provides the empirical basis for model selection.

* **OpenAI text-embedding-3 (Small/Large):** These models are the industry standard for general-purpose retrieval. Their support for **variable dimensions** allows architects to trade off storage costs against precision. For example, storing 256-dimension vectors for rapid candidate generation and 3072-dimension vectors for re-ranking.32  
* **Cohere Embed v3:** This model is specifically trained for RAG tasks. It distinguishes between "Query" embeddings (questions) and "Document" embeddings (facts). This asymmetry is crucial for email, where the user's question ("Why did we delay?") often looks semantically different from the answer ("The supplier had a shortage").32  
* **Open Source Leaders (BGE-M3, E5):** Models like **BGE-M3** (BAAI General Embedding) offer exceptional performance, particularly for multilingual corpora. BGE-M3 supports **Sparse Retrieval** natively alongside dense vectors, enabling a single model to handle both semantic concepts and exact keyword matching.35

### **5.2 Comparison of Embedding Strategies**

| Model Family | Strengths | Weaknesses | Best For |
| :---- | :---- | :---- | :---- |
| **OpenAI (ada-002 / embedding-3)** | Ease of use; Variable dimensions; 8k context. | Cost at scale; "Black box" (no fine-tuning). | General enterprise use; Rapid prototyping. |
| **Cohere Embed v3** | Asymmetric retrieval (Query vs. Doc); Compression. | API latency; Cost. | High-precision QA tasks; Multi-lingual RAG. |
| **Open Source (BGE / E5)** | Free (self-hosted); Fine-tunable; Privacy. | Infrastructure management cost; Hardware requirements. | High-security environments; Domain-specific jargon. |

### **5.3 Domain Adaptation and Fine-Tuning**

Generic models often fail on company-specific jargon (e.g., internal project codenames like "Project Apollo"). To address this, **Matryoshka Representation Learning** and **GPL (Generative Pseudo Labeling)** can be used to fine-tune embeddings.

* **GPL:** This technique uses an LLM to generate synthetic queries for your specific email corpus. The embedding model is then fine-tuned on these pairs, teaching it that "Apollo" refers to "The Q3 Migration Project" within this specific vector space.33 This domain adaptation can yield performance improvements of 10-20% in retrieval accuracy.

---

## **6\. Retrieval Architectures: Moving to Hybrid Search**

Relying solely on dense vector search is insufficient for email. Users frequently search for exact identifiers—Invoice Numbers, Ticket IDs, or specific Error Codes—which semantic models tend to abstract into generic concepts (e.g., "financial document" or "technical error").

### **6.1 The Hybrid Search Paradigm**

The optimal architecture implements **Hybrid Search**, combining the strengths of two distinct retrieval methodologies.

1. **Dense Vector Search (Semantic):** Retrieves documents based on conceptual similarity. Good for: "Why is the project delayed?"  
2. **Sparse Vector Search (Keyword):** Retrieves documents based on lexical matching (e.g., BM25 or SPLADE). Good for: "Error 503" or "Invoice \#992".

Reciprocal Rank Fusion (RRF):  
To combine these disparate results, the RRF algorithm is employed. RRF assigns a score based on the rank of a document in both lists.

$$Score(d) \= \\sum\_{r \\in R} \\frac{1}{k \+ r(d)}$$

Where $r(d)$ is the rank of document $d$ in list $R$, and $k$ is a constant (typically 60). This method boosts documents that appear in both retrieval lists, ensuring that the final result set balances semantic relevance with keyword precision.4

### **6.2 Reranking: The Precision Layer**

Retrieving the top 50 or 100 chunks via Hybrid Search is only the first step. To select the final 5-10 chunks for the LLM's context window, a **Cross-Encoder Reranker** is required.

* **ColBERT (Contextualized Late Interaction over BERT):** This architecture represents the current state-of-the-art for efficient reranking. Unlike standard cross-encoders that process the full query-document pair (computationally expensive), ColBERT interacts at the token level ("Late Interaction"). It allows for the pre-computation of document embeddings while still capturing the fine-grained interaction between query and document terms. This provides the accuracy of a cross-encoder with speeds approaching that of a bi-encoder.37  
* **Commercial Rerankers:** Services like **Cohere Rerank** serve as a "black box" optimization layer. Benchmarks consistently show that adding a reranking step can double the MRR (Mean Reciprocal Rank) of the system, making it the single most effective upgrade for a RAG pipeline.34

---

## **7\. GraphRAG: The Structural Revolution**

While vector search excels at finding similar *text*, it fails at *reasoning* about the relationships between entities. It cannot easily answer questions like, "Who are the key stakeholders in Project X?" or "How did the communication sentiment change after the incident?" This limitation drives the shift toward **GraphRAG**.1

### **7.1 Knowledge Graph Construction (KGC)**

GraphRAG involves constructing a Knowledge Graph (KG) where emails, users, and concepts are nodes, and their interactions are edges.

Schema Design for Email:  
A robust ontology is critical for effective GraphRAG. The recommended schema includes:

* **Nodes:**  
  * Person (Properties: Name, Email, Role, Department)  
  * Email (Properties: ID, Timestamp, Subject, Embedding)  
  * Entity (Properties: Extracted Topic, e.g., "Budget", "Project Alpha")  
  * Attachment (Properties: Type, Content Vector)  
* **Relationships:**  
  * (:Person)--\>(:Email)  
  * (:Email)--\>(:Person)  
  * (:Email)--\>(:Email)  
  * (:Email)--\>(:Entity)  
  * (:Person)--\>(:Community)

This graph structure allows for "Multi-hop" reasoning. A query can traverse the graph: *Find all emails sent by Bob (Node) \-\> that mention 'Project Alpha' (Relationship) \-\> and return the attachments (Node)*.41

### **7.2 Community Detection and Summarization**

Microsoft's open-source GraphRAG framework (released late 2024\) introduces Community Detection (using the Leiden algorithm). The system clusters nodes into "communities" based on the density of their connections (e.g., a cluster of people and emails discussing "Q3 Budget").  
The system then uses an LLM to generate a summary for each community. When a user asks a high-level global question ("What is the team working on?"), the system retrieves these pre-computed community summaries rather than thousands of individual emails. This capability—answering "global" questions—is the primary advantage of GraphRAG over Vector RAG.5

### **7.3 Hybrid Graph+Vector Architecture**

The optimal architecture is a **Hybrid System**:

1. **Vector Index:** Handles unstructured semantic search ("Find emails about 'latency'").  
2. **Knowledge Graph:** Handles structured/relational queries ("Who did Bob email?").  
3. **Router:** An agent determines which index to query based on the user's intent.  
4. **Graph-Enhanced Vector Search:** Graph algorithms (like PageRank) can calculate a "centrality score" for each email or user. This score can be stored as metadata in the Vector DB and used to boost the ranking of important emails during vector retrieval.44

---

## **8\. Agentic Orchestration and Workflows**

The static "Query \-\> Retrieve \-\> Answer" loop is being replaced by **Agentic Workflows**. An "Agent" is a system that uses an LLM to reason about the steps required to answer a question, utilizing tools and loops to refine its output.

### **8.1 The Router/Triage Agent**

The entry point of the system is a **Router Agent**. This agent classifies the incoming query to determine the appropriate retrieval strategy.

* **Intent Classification:**  
  * *Factual Retrieval:* "What is the invoice number?" \-\> Route to **Vector Store**.  
  * *Relational Query:* "Who is working on the API?" \-\> Route to **Knowledge Graph**.  
  * *Summarization:* "Summarize the last month." \-\> Route to **GraphRAG Community Summaries**.  
  * *Action:* "Draft a reply." \-\> Route to **Generation Module**.

Frameworks like **LangGraph** and **CrewAI** allow developers to define these distinct paths as a state machine. The agent can transition between states; for example, if a Vector Search yields low-confidence results, the agent can transition to a "Query Reformulation" state and try again—a pattern known as **Corrective RAG**.6

### **8.2 Query Transformation and HyDE**

Users rarely formulate perfect queries. Agentic workflows employ **Query Transformation** to bridge the gap between user intent and data representation.

* **HyDE (Hypothetical Document Embeddings):** The Agent generates a *hypothetical* ideal email that answers the user's query. This hypothetical text is embedded and used for retrieval. This is particularly effective for finding emails based on their *implied* content rather than just keywords.1  
* **Multi-Query Expansion:** For complex questions ("Compare the marketing plans of 2023 and 2024"), the Agent breaks the query into sub-queries ("2023 marketing plan", "2024 marketing plan"), executes them in parallel, and synthesizes the results.36

---

## **9\. Multimodal Capabilities: Processing Attachments**

Emails act as containers for attachments—PDFs, images, spreadsheets—that often contain the critical information. A RAG system that ignores attachments is functionally incomplete.

### **9.1 Document Intelligence and Multimodal Embeddings**

* **Extraction:** Use Document Intelligence services (e.g., Azure Document Intelligence, Amazon Textract) to perform OCR and layout analysis on attachments. This converts a PDF attachment into structured text that can be chunked and indexed alongside the email body.50  
* **Multimodal Embeddings:** For image attachments, models like **CLIP** (Contrastive Language-Image Pre-training) allow the system to embed images into the same vector space as text. This enables a user to search for "the graph showing sales decline," and the system can retrieve an image attachment that matches that semantic description.52  
* **Table Parsing:** Tabular data in Excel or embedded images poses a specific challenge. Standard chunking destroys table structure. Algorithms that convert tables to Markdown or JSON representations preserve the row/column relationships, enabling the LLM to reason about the data (e.g., "What is the value in row 3, column 2?").54

---

## **10\. Privacy, Security, and Compliance**

Email data is a primary vector for sensitive information. Implementing RAG on email requires rigorous adherence to security protocols to prevent data leaks and compliance violations (GDPR, HIPAA).

### **10.1 PII Redaction at Ingestion**

Storing Personally Identifiable Information (PII) in a vector database creates a permanent security risk. The standard mitigation is **Redaction at Ingestion**.

* **Microsoft Presidio:** This open-source library utilizes Named Entity Recognition (NER) and regex patterns to identify PII entities (Credit Cards, SSNs, Phones, Emails).  
* **Tokenization:** Instead of simply blurring the data, Presidio can replace entities with consistent tokens (e.g., "Call me at 555-0199" \-\> "Call me at \<PHONE\_NUMBER\_1\>"). This preserves the *context* for the LLM (it knows a phone number was shared) without exposing the *raw data*. This is essential for maintaining the utility of the data for reasoning tasks.55

### **10.2 Role-Based Access Control (RBAC)**

A critical vulnerability in Enterprise RAG is Semantic Leaking, where a user retrieves confidential information because their query semantically matches a restricted document.  
Solution: RBAC must be enforced at the Vector Database Level.

1. **Metadata Tagging:** During ingestion, every chunk inherits the Access Control List (ACL) of its source email (e.g., acl\_groups: \['hr', 'execs'\]).  
2. **Query Injection:** When a user queries the system, the middleware looks up their group memberships (e.g., \['engineering'\]).  
3. Filtered Retrieval: The query sent to the Vector DB includes a mandatory filter: filter: { acl\_groups: { $in: \['engineering'\] } }.  
   This ensures that the retrieval engine physically skips any document the user is not authorized to see, preventing the LLM from ever accessing restricted context.57

---

## **11\. Evaluation and Benchmarking**

In the probabilistic world of LLMs, "vibe checking" is not a valid testing strategy. Rigorous, metric-based evaluation is required to deploy with confidence.

### **11.1 RAG Evaluation Metrics**

Frameworks like **RAGAS** (Retrieval Augmented Generation Assessment) and **TruLens** utilize "LLM-as-a-Judge" to calculate core metrics:

* **Faithfulness:** Does the generated answer derive strictly from the retrieved context? (Hallucination check).  
* **Answer Relevance:** Does the answer directly address the user's query?  
* **Context Precision:** Did the retrieval system find the relevant email amidst the noise?.59

### **11.2 The Golden Dataset**

For email RAG, generic benchmarks are insufficient. Teams must construct a **Golden Dataset**—a curated set of 100+ {Question, Answer, Source Email ID} triples specific to their domain.

* **Synthetic Generation:** To bootstrap this, use an LLM to read a sample of emails and generate synthetic questions and answers ("What is the invoice date in this email?"). This creates a test set at scale without manual annotation.  
* **Optimization:** Use this dataset to run parameter sweeps—testing different chunk sizes, overlap values, and top-k retrieval settings—to empirically determine the configuration that maximizes **Hit Rate** and **MRR (Mean Reciprocal Rank)**.60

---

## **12\. Conclusion and Strategic Outlook**

The development of an Email Data Module for RAG in 2025 is no longer a matter of simply connecting a data pipe to a vector database. It requires a holistic architecture that respects the structural, temporal, and social intricacies of human communication.

The convergence of technologies—**GraphRAG** for structural reasoning, **Agentic Workflows** for planning, and **Hybrid Search** for precision—provides the toolkit necessary to solve the "Email RAG" problem. By adopting the **Small-to-Big retrieval pattern**, enforcing **RBAC at the database layer**, and implementing rigorous **PII redaction**, organizations can transform their email archives from a dormant liability into a dynamic, intelligent asset.

**Summary of Key Architectural Recommendations:**

1. **Adhere to GraphRAG:** Supplement vectors with a knowledge graph to capture the "who knows who" and "who knows what" of the enterprise.  
2. **Privilege the Thread:** Never chunk emails in isolation; always preserve the thread context via parent-child indexing.  
3. **Filter Aggressively:** Invest heavily in pre-processing (signature/disclaimer removal) to prevent vector collapse.  
4. **Orchestrate via Agents:** Use router agents to dynamically select the best retrieval tool for the user's specific intent.  
5. **Secure by Design:** Bake access controls and redaction into the ingestion and retrieval layers, not the application layer.

This architecture represents the definitive path forward for building high-performance, secure, and context-aware Email RAG platforms.

#### **Works cited**

1. RAG in 2025: The enterprise guide to retrieval augmented generation, Graph RAG and agentic AI \- Data Nucleus, accessed on November 22, 2025, [https://datanucleus.dev/rag-and-agentic-ai/what-is-rag-enterprise-guide-2025](https://datanucleus.dev/rag-and-agentic-ai/what-is-rag-enterprise-guide-2025)  
2. Top 8 RAG Architectures to Know in 2025 \- Keywords AI, accessed on November 22, 2025, [https://www.keywordsai.co/blog/top-8-rag-architectures-to-know-in-2025](https://www.keywordsai.co/blog/top-8-rag-architectures-to-know-in-2025)  
3. Retrieval-Augmented Generation (RAG) \- Pinecone, accessed on November 22, 2025, [https://www.pinecone.io/learn/retrieval-augmented-generation/](https://www.pinecone.io/learn/retrieval-augmented-generation/)  
4. Best architecture for searching historical emails semantically? \- API, accessed on November 22, 2025, [https://community.openai.com/t/best-architecture-for-searching-historical-emails-semantically/592601](https://community.openai.com/t/best-architecture-for-searching-historical-emails-semantically/592601)  
5. The Future of AI: GraphRAG – A better way to query interlinked documents, accessed on November 22, 2025, [https://techcommunity.microsoft.com/blog/azure-ai-foundry-blog/the-future-of-ai-graphrag-%E2%80%93-a-better-way-to-query-interlinked-documents/4287182](https://techcommunity.microsoft.com/blog/azure-ai-foundry-blog/the-future-of-ai-graphrag-%E2%80%93-a-better-way-to-query-interlinked-documents/4287182)  
6. \[2501.09136\] Agentic Retrieval-Augmented Generation: A Survey on Agentic RAG \- arXiv, accessed on November 22, 2025, [https://arxiv.org/abs/2501.09136](https://arxiv.org/abs/2501.09136)  
7. email.parser: Parsing email messages — Python 3.14.0 documentation, accessed on November 22, 2025, [https://docs.python.org/3/library/email.parser.html](https://docs.python.org/3/library/email.parser.html)  
8. email — An email and MIME handling package — Python 3.14.0 documentation, accessed on November 22, 2025, [https://docs.python.org/3/library/email.html](https://docs.python.org/3/library/email.html)  
9. Parsing Mails in Python, How Difficult Can It Be? \- cybersim's blog, accessed on November 22, 2025, [https://cybersim.ch/posts/python-mail-parsing/](https://cybersim.ch/posts/python-mail-parsing/)  
10. Python script to remove email signature \- GitHub Gist, accessed on November 22, 2025, [https://gist.github.com/alfredfrancis/f46304960c83093af17c4d0678178847](https://gist.github.com/alfredfrancis/f46304960c83093af17c4d0678178847)  
11. Parse Raw Email (MIME) Guide \- SigParser, accessed on November 22, 2025, [https://www.sigparser.com/developers/email-parsing/parse-raw-email](https://www.sigparser.com/developers/email-parsing/parse-raw-email)  
12. Partitioning \- Unstructured, accessed on November 22, 2025, [https://docs.unstructured.io/open-source/core-functionality/partitioning](https://docs.unstructured.io/open-source/core-functionality/partitioning)  
13. RAG On Email Data. A General Guide Based On My Professional Experience. \- Medium, accessed on November 22, 2025, [https://medium.com/@jojokirby/rag-on-email-data-a-general-guide-based-on-my-professional-experience-bb7f55b11412](https://medium.com/@jojokirby/rag-on-email-data-a-general-guide-based-on-my-professional-experience-bb7f55b11412)  
14. Strip signatures and replies from emails \- Stack Overflow, accessed on November 22, 2025, [https://stackoverflow.com/questions/1372694/strip-signatures-and-replies-from-emails](https://stackoverflow.com/questions/1372694/strip-signatures-and-replies-from-emails)  
15. How to trim emails for just the body, when using email as input to an external system?, accessed on November 22, 2025, [https://softwareengineering.stackexchange.com/questions/116840/how-to-trim-emails-for-just-the-body-when-using-email-as-input-to-an-external-s](https://softwareengineering.stackexchange.com/questions/116840/how-to-trim-emails-for-just-the-body-when-using-email-as-input-to-an-external-s)  
16. Ultralytics/Signature · Datasets at Hugging Face, accessed on November 22, 2025, [https://huggingface.co/datasets/Ultralytics/Signature](https://huggingface.co/datasets/Ultralytics/Signature)  
17. tech4humans/signature-detection · Datasets at Hugging Face, accessed on November 22, 2025, [https://huggingface.co/datasets/tech4humans/signature-detection](https://huggingface.co/datasets/tech4humans/signature-detection)  
18. Algorithm to identify quoted content in email replies \- Gmail \- Latenode Official Community, accessed on November 22, 2025, [https://community.latenode.com/t/algorithm-to-identify-quoted-content-in-email-replies/24581](https://community.latenode.com/t/algorithm-to-identify-quoted-content-in-email-replies/24581)  
19. Extract email bodies, remove reply chains and signatures \- SigParser, accessed on November 22, 2025, [https://www.sigparser.com/developers/extract-reply-chains-from-emails](https://www.sigparser.com/developers/extract-reply-chains-from-emails)  
20. Managing Threads | Gmail \- Google for Developers, accessed on November 22, 2025, [https://developers.google.com/workspace/gmail/api/guides/threads](https://developers.google.com/workspace/gmail/api/guides/threads)  
21. Java library for grouping emails together by thread? \- Stack Overflow, accessed on November 22, 2025, [https://stackoverflow.com/questions/2330011/java-library-for-grouping-emails-together-by-thread](https://stackoverflow.com/questions/2330011/java-library-for-grouping-emails-together-by-thread)  
22. Email Threading Overview \- Reveal, accessed on November 22, 2025, [https://docs.revealdata.com/brainspace/docs/email-threading-overview](https://docs.revealdata.com/brainspace/docs/email-threading-overview)  
23. Vector Search retrieval quality guide \- Azure Databricks \- Microsoft Learn, accessed on November 22, 2025, [https://learn.microsoft.com/en-us/azure/databricks/vector-search/vector-search-retrieval-quality](https://learn.microsoft.com/en-us/azure/databricks/vector-search/vector-search-retrieval-quality)  
24. Create and get data lineage relationships for custom assets using the Microsoft Purview REST API, accessed on November 22, 2025, [https://learn.microsoft.com/en-us/purview/data-gov-api-create-lineage-relationships](https://learn.microsoft.com/en-us/purview/data-gov-api-create-lineage-relationships)  
25. Different Types of Chunking Strategies in RAG: How I Optimised Data for Better AI Responses, accessed on November 22, 2025, [https://medium.com/@mansoorsyed05/different-types-of-chunking-strategies-in-rag-how-i-optimised-data-for-better-ai-responses-ef4b079dd8f2](https://medium.com/@mansoorsyed05/different-types-of-chunking-strategies-in-rag-how-i-optimised-data-for-better-ai-responses-ef4b079dd8f2)  
26. The Ultimate Guide to Chunking Strategies for RAG Applications with Databricks \- Medium, accessed on November 22, 2025, [https://medium.com/@debusinha2009/the-ultimate-guide-to-chunking-strategies-for-rag-applications-with-databricks-e495be6c0788](https://medium.com/@debusinha2009/the-ultimate-guide-to-chunking-strategies-for-rag-applications-with-databricks-e495be6c0788)  
27. Semantic Chunking for RAG: Better Context, Better Results \- Multimodal, accessed on November 22, 2025, [https://www.multimodal.dev/post/semantic-chunking-for-rag](https://www.multimodal.dev/post/semantic-chunking-for-rag)  
28. Text splitters \- Docs by LangChain, accessed on November 22, 2025, [https://docs.langchain.com/oss/python/integrations/splitters](https://docs.langchain.com/oss/python/integrations/splitters)  
29. Chunking Strategies for LLM Applications \- Pinecone, accessed on November 22, 2025, [https://www.pinecone.io/learn/chunking-strategies/](https://www.pinecone.io/learn/chunking-strategies/)  
30. RAG 2.0 : Advanced Chunking Strategies with Examples. | by Vishal Mysore \- Medium, accessed on November 22, 2025, [https://medium.com/@visrow/rag-2-0-advanced-chunking-strategies-with-examples-d87d03adf6d1](https://medium.com/@visrow/rag-2-0-advanced-chunking-strategies-with-examples-d87d03adf6d1)  
31. Best Chunking Strategies for RAG in 2025 \- Firecrawl, accessed on November 22, 2025, [https://www.firecrawl.dev/blog/best-chunking-strategies-rag-2025](https://www.firecrawl.dev/blog/best-chunking-strategies-rag-2025)  
32. 5 Best Embedding Models for RAG: How to Choose the Right One \- GreenNode, accessed on November 22, 2025, [https://greennode.ai/blog/best-embedding-models-for-rag](https://greennode.ai/blog/best-embedding-models-for-rag)  
33. Embedding Models in 2025 — Technology, Pricing & Practical Advice | by Aleksandr Azimbaev | Medium, accessed on November 22, 2025, [https://medium.com/@alex-azimbaev/embedding-models-in-2025-technology-pricing-practical-advice-2ed273fead7f](https://medium.com/@alex-azimbaev/embedding-models-in-2025-technology-pricing-practical-advice-2ed273fead7f)  
34. Boosting RAG: Picking the Best Embedding & Reranker models \- LlamaIndex, accessed on November 22, 2025, [https://www.llamaindex.ai/blog/boosting-rag-picking-the-best-embedding-reranker-models-42d079022e83](https://www.llamaindex.ai/blog/boosting-rag-picking-the-best-embedding-reranker-models-42d079022e83)  
35. accessed on November 22, 2025, [https://artsmart.ai/blog/top-embedding-models-in-2025/](https://artsmart.ai/blog/top-embedding-models-in-2025/)  
36. Retrieval Augmented Generation (RAG) in Azure AI Search \- Microsoft Learn, accessed on November 22, 2025, [https://learn.microsoft.com/en-us/azure/search/retrieval-augmented-generation-overview](https://learn.microsoft.com/en-us/azure/search/retrieval-augmented-generation-overview)  
37. What is ColBERT and how does it differ from standard bi-encoder approaches? \- Milvus, accessed on November 22, 2025, [https://milvus.io/ai-quick-reference/what-is-colbert-and-how-does-it-differ-from-standard-biencoder-approaches](https://milvus.io/ai-quick-reference/what-is-colbert-and-how-does-it-differ-from-standard-biencoder-approaches)  
38. ColBERT \- Improve Retrieval Performance with Token Level Vector Embeddings, accessed on November 22, 2025, [https://www.analyticsvidhya.com/blog/2024/04/colbert-improve-retrieval-performance-with-token-level-vector-embeddings/](https://www.analyticsvidhya.com/blog/2024/04/colbert-improve-retrieval-performance-with-token-level-vector-embeddings/)  
39. Enhancing Search Relevancy with Cohere Rerank 3.5 and Amazon OpenSearch Service | AWS Big Data Blog, accessed on November 22, 2025, [https://aws.amazon.com/blogs/big-data/enhancing-search-relevancy-with-cohere-rerank-3-5-and-amazon-opensearch-service/](https://aws.amazon.com/blogs/big-data/enhancing-search-relevancy-with-cohere-rerank-3-5-and-amazon-opensearch-service/)  
40. Choosing Between RAG and GraphRAG Based on Data Type and Structure | by PrajnaAI, accessed on November 22, 2025, [https://prajnaaiwisdom.medium.com/choosing-between-rag-and-graphrag-based-on-data-type-and-structure-9911020501fc](https://prajnaaiwisdom.medium.com/choosing-between-rag-and-graphrag-based-on-data-type-and-structure-9911020501fc)  
41. GraphRAG: Using Knowledge in Unstructured Data to Build Apps with LLMs \- Graphlit, accessed on November 22, 2025, [https://www.graphlit.com/blog/graphrag-using-knowledge-in-unstructured-data-to-build-apps-with-llms](https://www.graphlit.com/blog/graphrag-using-knowledge-in-unstructured-data-to-build-apps-with-llms)  
42. How would I model an email marketing graph in neo4j \- Stack Overflow, accessed on November 22, 2025, [https://stackoverflow.com/questions/25629989/how-would-i-model-an-email-marketing-graph-in-neo4j](https://stackoverflow.com/questions/25629989/how-would-i-model-an-email-marketing-graph-in-neo4j)  
43. Microsoft GraphRAG: Redefining AI-Based Content Interpretation and Search | PART 1, accessed on November 22, 2025, [https://medium.com/@jinglemind.dev/microsoft-graphrag-redefining-ai-based-content-interpretation-and-search-part-1-6491dab0e2b3](https://medium.com/@jinglemind.dev/microsoft-graphrag-redefining-ai-based-content-interpretation-and-search-part-1-6491dab0e2b3)  
44. Building, Improving, and Deploying Knowledge Graph RAG Systems on Databricks, accessed on November 22, 2025, [https://www.databricks.com/blog/building-improving-and-deploying-knowledge-graph-rag-systems-databricks](https://www.databricks.com/blog/building-improving-and-deploying-knowledge-graph-rag-systems-databricks)  
45. GraphRAG: Utilising complex data relationships for more efficient LLM queries \- adesso SE, accessed on November 22, 2025, [https://www.adesso.de/en/news/blog/graphrag-utilising-complex-data-relationships-for-more-efficient-llm-queries.jsp](https://www.adesso.de/en/news/blog/graphrag-utilising-complex-data-relationships-for-more-efficient-llm-queries.jsp)  
46. Building an AI Email Assistant with Prompt Chaining and LangGraph :: Aamer Paul, accessed on November 22, 2025, [https://aamernabi.github.io/posts/prompt-chaining-using-langgraph/](https://aamernabi.github.io/posts/prompt-chaining-using-langgraph/)  
47. Email Workflows with LangGraph and GROQ \- Analytics Vidhya, accessed on November 22, 2025, [https://www.analyticsvidhya.com/blog/2024/11/streamline-email-workflows-with-langgraph-and-groq/](https://www.analyticsvidhya.com/blog/2024/11/streamline-email-workflows-with-langgraph-and-groq/)  
48. How are people syncing and indexing data from tools like Gmail or Slack for RAG? \- Reddit, accessed on November 22, 2025, [https://www.reddit.com/r/LocalLLaMA/comments/1oddd4r/how\_are\_people\_syncing\_and\_indexing\_data\_from/](https://www.reddit.com/r/LocalLLaMA/comments/1oddd4r/how_are_people_syncing_and_indexing_data_from/)  
49. Agentic Retrieval \- Azure AI Search \- Microsoft Learn, accessed on November 22, 2025, [https://learn.microsoft.com/en-us/azure/search/agentic-retrieval-overview](https://learn.microsoft.com/en-us/azure/search/agentic-retrieval-overview)  
50. Tutorial: Vectorize images and text \- Azure AI Search \- Microsoft Learn, accessed on November 22, 2025, [https://learn.microsoft.com/en-us/azure/search/tutorial-document-extraction-multimodal-embeddings](https://learn.microsoft.com/en-us/azure/search/tutorial-document-extraction-multimodal-embeddings)  
51. Tutorial: Vectorize from a structured document layout \- Azure AI Search | Microsoft Learn, accessed on November 22, 2025, [https://learn.microsoft.com/en-us/azure/search/tutorial-document-layout-multimodal-embeddings](https://learn.microsoft.com/en-us/azure/search/tutorial-document-layout-multimodal-embeddings)  
52. How Does Multimodal RAG Improve Context-Aware AI? | by Kanerika Inc \- Medium, accessed on November 22, 2025, [https://medium.com/@kanerika/how-does-multimodal-rag-improve-context-aware-ai-11561ec15ee2](https://medium.com/@kanerika/how-does-multimodal-rag-improve-context-aware-ai-11561ec15ee2)  
53. Multimodal Search Concepts and Guidance \- Azure AI Search | Microsoft Learn, accessed on November 22, 2025, [https://learn.microsoft.com/en-us/azure/search/multimodal-search-overview](https://learn.microsoft.com/en-us/azure/search/multimodal-search-overview)  
54. Building a Multimodal RAG That Responds with Text, Images, and Tables from Sources, accessed on November 22, 2025, [https://towardsdatascience.com/building-a-multimodal-rag-with-text-images-tables-from-sources-in-response/](https://towardsdatascience.com/building-a-multimodal-rag-with-text-images-tables-from-sources-in-response/)  
55. Home \- Microsoft Presidio \- Microsoft Open Source, accessed on November 22, 2025, [https://microsoft.github.io/presidio/](https://microsoft.github.io/presidio/)  
56. microsoft/presidio: An open-source framework for detecting, redacting, masking, and anonymizing sensitive data (PII) across text, images, and structured data. Supports NLP, pattern matching, and customizable pipelines. \- GitHub, accessed on November 22, 2025, [https://github.com/microsoft/presidio](https://github.com/microsoft/presidio)  
57. Best Practices for Privacy in RAG Chatbots \- Artech Digital, accessed on November 22, 2025, [https://www.artech-digital.com/blog/best-practices-for-privacy-in-rag-chatbots](https://www.artech-digital.com/blog/best-practices-for-privacy-in-rag-chatbots)  
58. Optimizing RAG Systems for Sensitive Data and Privacy Compliance \- Chitika, accessed on November 22, 2025, [https://www.chitika.com/optimizing-rag-sensitive-data-privacy/](https://www.chitika.com/optimizing-rag-sensitive-data-privacy/)  
59. RAGBench: Explainable Benchmark for Retrieval-Augmented Generation Systems \- arXiv, accessed on November 22, 2025, [https://arxiv.org/html/2407.11005v2](https://arxiv.org/html/2407.11005v2)  
60. How to create custom evaluation/benchmark for your own dataset? : r/Rag \- Reddit, accessed on November 22, 2025, [https://www.reddit.com/r/Rag/comments/1jyn3i1/how\_to\_create\_custom\_evaluationbenchmark\_for\_your/](https://www.reddit.com/r/Rag/comments/1jyn3i1/how_to_create_custom_evaluationbenchmark_for_your/)  
61. Retrieval-Augmented Generation (RAG) Evaluators for Generative AI \- Microsoft Foundry, accessed on November 22, 2025, [https://learn.microsoft.com/en-us/azure/ai-foundry/concepts/evaluation-evaluators/rag-evaluators?view=foundry-classic](https://learn.microsoft.com/en-us/azure/ai-foundry/concepts/evaluation-evaluators/rag-evaluators?view=foundry-classic)