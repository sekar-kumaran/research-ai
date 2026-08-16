# School of Computer Science and Engineering
# Lovely Professional University, Phagwara, Punjab (India)

**Month:** May  
**Year:** 2026

<br>

# Machine Learning Project
# Research AI Intelligence Platform
# Final Report

<br>

**Submitted By:** [Student Name]  
**Registration No.:** [Registration Number]

<br><br>

**Project Title:** Research AI Intelligence Platform  
**Domain:** Machine Learning, Natural Language Processing, Information Retrieval, Research Analytics  
**Dataset:** arXiv Metadata Dataset  
**Backend Framework:** FastAPI  
**Core ML Stack:** scikit-learn, FAISS, SentenceTransformers, pandas, PyArrow  

<div style="page-break-after: always;"></div>

# SUPERVISOR'S CERTIFICATE

This is to certify that the work reported in the B.Tech Dissertation / project report entitled **"Research AI Intelligence Platform"**, submitted by **[Student Name]** at Lovely Professional University, Phagwara, India, is a bonafide record of original work carried out under supervision.

The project presents a machine learning and information retrieval system for analyzing research papers using arXiv metadata. The work includes large-scale data preprocessing, exploratory data analysis, supervised category classification, semantic similarity search, hybrid retrieval, retrieval-augmented answer generation, citation proxy analysis, trend analysis, paper-chat functionality, and backend integration through API services.

This work has not been submitted elsewhere for any other degree or diploma.

<br><br><br>

**Signature of Supervisor**

<br><br>

**Name of Supervisor:** [Supervisor Name]  
**Date:** [Date]

<div style="page-break-after: always;"></div>

# DECLARATION

I hereby declare that the project work reported in the dissertation / project report entitled **"Research AI Intelligence Platform"**, submitted in partial fulfilment of the requirement for the award of the degree of Bachelor of Technology in Computer Science and Engineering at Lovely Professional University, Phagwara, Punjab, is an authentic work carried out by me.

The project has been developed as a machine learning based research intelligence platform that can process scientific paper metadata, classify research papers into subject categories, retrieve relevant papers using semantic search, summarize research content, support paper-based question answering, and expose these capabilities through a backend API and frontend interface.

I understand that the work presented herewith is in direct compliance with Lovely Professional University's policy on plagiarism, intellectual property rights, and the highest standards of moral and ethical conduct. Therefore, to the best of my knowledge, the content of this report represents an authentic and honest project effort conducted by me. I am fully responsible for the contents of this project work.

<br><br><br>

**Signature of Candidate**

**Name of Candidate:** [Student Name]  
**Registration No.:** [Registration Number]

<div style="page-break-after: always;"></div>

# ACKNOWLEDGEMENT

I would like to express my sincere gratitude to Lovely Professional University for providing the academic environment, learning resources, and technical foundation required to complete this project. I am thankful to the School of Computer Science and Engineering for encouraging practical, implementation-oriented learning in the areas of machine learning, artificial intelligence, and software engineering.

I am deeply grateful to my project supervisor for guidance, feedback, and support throughout the development of this work. The guidance helped in shaping the project from a basic machine learning idea into a complete research intelligence platform with multiple connected components, including classification, retrieval, summarization, trend analysis, citation proxy analysis, and backend API integration.

I also acknowledge the creators and maintainers of the arXiv dataset. arXiv has made a large scientific metadata corpus available to the research and machine learning community, enabling students and researchers to build systems for paper recommendation, category prediction, semantic search, trend discovery, and knowledge exploration.

Finally, I would like to thank my peers, teachers, and family members for their support and encouragement during the project. Their motivation helped me complete the implementation, experiments, evaluation, and final documentation.

<div style="page-break-after: always;"></div>

# TABLE OF CONTENTS

1. Problem Statement  
2. Introduction  
3. Objectives  
4. Scope of the Project  
5. Literature Review  
6. Dataset Selection and Description  
7. Data Cleaning and Preprocessing  
8. Exploratory Data Analysis  
9. Feature Engineering  
10. Machine Learning Algorithms  
11. Similarity Search and Retrieval System  
12. Backend Architecture  
13. Model Training and Evaluation  
14. Comparative Analysis  
15. System Implementation  
16. API and User Workflow  
17. Testing and Validation  
18. Deployment  
19. Results and Discussion  
20. Limitations  
21. Future Scope  
22. Conclusion  
23. References  

<div style="page-break-after: always;"></div>

# 1. PROBLEM STATEMENT

The volume of scientific research has increased rapidly over the last three decades. Platforms such as arXiv publish thousands of papers across computer science, mathematics, physics, statistics, electrical engineering, quantitative biology, economics, and related fields. While open-access repositories have made research easier to access, they have also introduced a new problem: researchers, students, and academic professionals often struggle to identify relevant papers, understand research trends, compare methods, and extract meaningful insights from a large corpus of scientific metadata.

Traditional keyword search is useful but limited. A user may search for a topic such as "transformer architecture", "graph neural networks", or "quantum computing", but simple keyword matching may miss semantically related papers whose abstracts use different terminology. Similarly, a student may want to classify a paper into a broad subject area, identify related works, summarize a paper, or ask questions about an uploaded paper. Doing all of this manually is time-consuming and becomes increasingly difficult as the corpus size grows into millions of records.

The project addresses this problem by developing a **Research AI Intelligence Platform** that combines machine learning, natural language processing, semantic search, and agentic orchestration. The platform uses the arXiv metadata dataset as its knowledge base and provides intelligent services such as paper classification, hybrid search, summarization, similarity comparison, paper chat, citation proxy analysis, trend analysis, metadata analysis, and retrieval-augmented answer generation.

The major challenge is not only to train a classifier or build a search index, but to connect multiple models and services into a working system. The platform must handle large-scale data, preprocess research text, create reliable artifacts, expose backend APIs, and provide grounded responses without fabricating research information. Therefore, the problem statement can be summarized as:

> To design and implement an AI-powered research intelligence platform that can process large-scale scientific metadata, classify research papers, retrieve semantically relevant papers, analyze research trends, and support grounded question answering through an integrated backend system.

This problem is important because research discovery is a real academic and industrial need. Students need relevant papers for literature reviews. Researchers need quick access to related work and emerging trends. Institutions need tools that can organize and analyze large academic datasets. A machine learning based research assistant can reduce manual effort and improve the quality of literature exploration.

<div style="page-break-after: always;"></div>

# 2. INTRODUCTION

Research papers contain valuable scientific knowledge, but the size and complexity of modern paper repositories make it difficult to extract useful information manually. arXiv is one of the most important open-access research repositories and contains millions of scientific papers. Each paper record includes fields such as title, abstract, authors, categories, update date, DOI, journal reference, and version history. These fields provide enough information to support multiple machine learning and information retrieval tasks.

The **Research AI Intelligence Platform** is built around the idea that research discovery should be more interactive and intelligent than static search. Instead of only listing papers for a keyword, the system can classify user queries, retrieve related papers, rank results, summarize content, identify trends, analyze metadata, and answer questions using retrieved evidence.

The project follows a modular architecture. The backend is implemented using FastAPI and is organized into services for classification, retrieval, summarization, ranking, trend analysis, citation analysis, paper ingestion, session memory, and orchestration. The platform is not a single model system; it is a collection of connected components. Some components rely on trained artifacts, such as the classification model and FAISS similarity index. Other components use rule-based logic, external/local language models, or runtime services.

The cleaned artifact structure after project refinement contains only the most valuable and backend-used model artifacts:

- `artifacts/classification/classifier.joblib`
- `artifacts/classification/tfidf_vectorizer.joblib`
- `artifacts/classification/model_report.json`
- `artifacts/similarity/paper_index.faiss`
- `artifacts/similarity/paper_metadata.parquet`
- `artifacts/similarity/embedding_model_name.joblib`
- `artifacts/feature_engineering/feature_manifest.json`
- EDA and report files in `artifacts/showcase/`

The active backend classifier is the promoted best model from the full-scale classification experiment. According to the production model report, the selected classifier is **Multinomial Naive Bayes**, trained on 2,683,334 records and tested on 298,669 records. It achieved an accuracy of approximately **81.40%** on the held-out test set.

The retrieval system is based on a FAISS vector index and a sentence embedding model. The embedding model recorded in the artifacts is `all-MiniLM-L6-v2`, a lightweight sentence-transformer model commonly used for semantic similarity tasks. The vector store enables the platform to retrieve papers based on semantic meaning rather than only exact keywords.

The final system therefore represents a practical machine learning application: it combines supervised learning, vector search, natural language processing, backend engineering, and user-facing research assistance.

<div style="page-break-after: always;"></div>

# 3. OBJECTIVES

The main objective of this project is to build a machine learning based platform that assists users in discovering, classifying, analyzing, and understanding scientific research papers.

The specific objectives are:

1. To collect and process a large-scale arXiv metadata dataset containing paper titles, abstracts, categories, authors, update dates, and related metadata.

2. To clean the dataset by removing invalid records, handling missing text fields, standardizing paper content, and preparing the data for machine learning tasks.

3. To perform exploratory data analysis on the dataset and understand category distribution, text length distribution, year-based trends, and research-domain imbalance.

4. To create feature engineering artifacts for text and numerical characteristics such as token count, title length, abstract length, title-to-abstract ratio, and log-transformed token count.

5. To train and evaluate supervised classification models for predicting the broad arXiv category of a paper from its title and abstract.

6. To promote the best classification model into the backend so that the API can classify new research queries or papers.

7. To build a semantic retrieval system using sentence embeddings and FAISS so that users can search for papers based on meaning, not only exact keywords.

8. To combine semantic retrieval with keyword and metadata-based reranking for more reliable search results.

9. To provide summarization, methodology extraction, citation proxy analysis, metadata analysis, and trend analysis services through a backend API.

10. To create a conversational research interface where user queries are planned, executed through appropriate tools, evaluated, and synthesized into grounded responses.

11. To maintain a clean artifact structure by keeping only backend-used models and high-value report artifacts.

12. To prepare a final project report that documents the problem, dataset, methodology, implementation, model results, system architecture, limitations, and future scope.

The project is therefore both a machine learning project and a software engineering project. It does not stop at model training; it integrates the trained model and retrieval artifacts into a working research assistant platform.

<div style="page-break-after: always;"></div>

# 4. SCOPE OF THE PROJECT

The scope of this project covers the design and implementation of a research intelligence platform using arXiv metadata. The project includes data preprocessing, EDA, supervised classification, semantic search, backend APIs, and agent-based orchestration.

The project uses structured metadata fields such as:

- Paper ID
- Title
- Abstract
- Authors
- arXiv categories
- Update date
- DOI
- Journal reference
- Version information

The major machine learning scope includes:

- Text vectorization of title and abstract content
- Broad category classification
- Evaluation using accuracy, macro F1, weighted F1, precision, recall, and support
- Semantic embedding generation
- FAISS-based nearest-neighbor retrieval
- Similarity comparison between research texts

The backend scope includes:

- FastAPI endpoints for classification, search, summarization, similarity, metadata analysis, citation proxy analysis, trend analysis, paper chat, and pipeline execution
- Tool registry for orchestrated agent execution
- Retrieval-augmented answer generation
- Conversation memory for multi-turn interaction
- Session memory for uploaded or arXiv-loaded papers
- Sandbox validation for controlled Python execution

The project also includes an EDA notebook:

- `notebooks/arxiv_eda_showcase.ipynb`

This notebook is intended to showcase dataset exploration and generate report-ready tables and figures. It supports memory-safe sampling of large parquet files, category distribution analysis, time trend analysis, missing value analysis, and text length visualization.

The scope does not include full production deployment with authentication, user accounts, rate limiting, or a cloud-hosted vector database. The current system is designed as a local-first research intelligence platform. It can be extended for production, but the current project focuses on the academic demonstration of machine learning and backend integration.

The project also does not use full-text PDFs for all arXiv papers. The main knowledge base is metadata-driven. Paper upload and arXiv paper chat features can process individual papers, but the large-scale corpus is based on metadata rather than full PDF content.

<div style="page-break-after: always;"></div>

# 5. LITERATURE REVIEW

Research paper analysis has become an important area in natural language processing and information retrieval. Earlier academic search systems relied primarily on keyword-based retrieval, where a query was matched against paper titles, abstracts, or indexed terms. While keyword search is computationally efficient, it often fails when a paper uses different terminology from the user's query. For example, a user searching for "deep learning for images" may miss papers that use terms such as "convolutional neural networks", "visual representation learning", or "vision transformers".

To overcome this limitation, modern systems use semantic representation learning. Sentence embedding models transform text into dense numerical vectors, allowing semantically similar documents to be close in vector space. Models such as Sentence-BERT and MiniLM have made semantic search practical because they generate meaningful embeddings while remaining efficient enough for local deployment.

FAISS, developed by Meta AI, is widely used for efficient similarity search over high-dimensional vectors. It supports exact and approximate nearest-neighbor search. In this project, FAISS is used to store paper embeddings and retrieve similar papers for a user query. The selected retrieval method uses inner-product similarity over normalized vectors, which is equivalent to cosine similarity.

Supervised paper classification is another well-known research task. arXiv papers are assigned categories such as computer science, mathematics, physics, statistics, and quantitative biology. Text classification models can learn category patterns from titles and abstracts. Traditional models such as Naive Bayes, logistic regression, and linear support vector machines remain strong baselines for text classification, especially when paired with TF-IDF or hashing vectorization. Although transformer-based classifiers can improve performance, classical models are easier to train at scale and require fewer computational resources.

Multinomial Naive Bayes is commonly used for text classification because it works well with word frequency and TF-IDF features. It assumes conditional independence between features, which is not fully true for natural language, but the model is fast, scalable, and often effective. Logistic regression and linear SVM models are also popular because they learn discriminative boundaries between categories. Ensemble models such as Random Forest can be useful for structured features but are not always ideal for sparse high-dimensional text unless dimensionality reduction is applied.

Retrieval-augmented generation is another relevant area. Instead of asking a language model to answer from memory, a RAG system first retrieves relevant documents and then generates an answer grounded in those documents. This reduces hallucination and improves trustworthiness. The platform follows this principle by retrieving paper metadata before synthesizing answers.

The project also draws from agentic system design. In an agentic architecture, the system does not execute a fixed pipeline for every user query. Instead, a planner decides which tools are needed, an execution agent runs those tools, an evaluator checks output quality, and a synthesizer produces the final answer. This allows a single platform to support classification, search, summarization, trend analysis, and conversation.

Overall, this project combines ideas from:

- Text classification
- Semantic search
- Vector databases
- Hybrid retrieval
- Metadata analysis
- Retrieval-augmented generation
- Agent-based orchestration
- Scientific document intelligence

<div style="page-break-after: always;"></div>

# 6. DATASET SELECTION AND DESCRIPTION

The dataset used in this project is the arXiv metadata dataset. arXiv is a large open-access repository of scientific papers. It contains papers from multiple domains including physics, mathematics, computer science, statistics, electrical engineering, quantitative biology, quantitative finance, and economics.

The project uses parquet versions of arXiv metadata stored under the `dataset/` directory. The major source files include:

- `dataset/arxiv_chunks/arxiv_part_A.parquet`
- `dataset/arxiv_chunks/arxiv_part_B.parquet`
- `dataset/data/arxiv_cleaned_A.parquet`
- `dataset/data/arxiv_cleaned_B.parquet`
- chunked parquet files under `dataset/data/chunks_A/`
- chunked parquet files under `dataset/data/chunks_B/`

The dataset fields include:

| Column | Description |
|---|---|
| `id` | Unique arXiv paper identifier |
| `submitter` | Person who submitted the paper |
| `authors` | Author names as text |
| `title` | Paper title |
| `comments` | Additional paper comments |
| `journal-ref` | Journal reference, if available |
| `doi` | Digital Object Identifier |
| `report-no` | Report number, if available |
| `categories` | arXiv categories assigned to the paper |
| `license` | License information |
| `abstract` | Paper abstract |
| `versions` | Version history |
| `update_date` | Last update date |
| `authors_parsed` | Parsed author list |

According to the cleaning summary artifact:

| Metric | Value |
|---|---:|
| Source parquet files | 2 |
| Raw rows | 2,982,054 |
| Clean rows | 2,982,003 |
| Unique cleaned IDs | 2,982,003 |
| Broad categories | 38 |
| Cleaned parts | 30 |
| NLP preprocessing | NLTK enabled |
| Deduplication mode | SQLite |
| Resume mode | Enabled |

The dataset is suitable for this project because it contains both text fields and metadata fields. Title and abstract fields are useful for classification, similarity search, summarization, and methodology extraction. Category labels are useful for supervised learning. Update dates support trend analysis. Author and journal fields support metadata analysis.

The size of the dataset also makes it realistic. A small academic demo dataset may not show the challenges of memory-safe processing, class imbalance, and scalable indexing. In contrast, the arXiv metadata corpus requires chunked loading, memory-aware preprocessing, and artifact-based storage.

<div style="page-break-after: always;"></div>

# 7. DATA CLEANING AND PREPROCESSING

The raw arXiv metadata contains millions of records. Before training models or building retrieval indexes, the data must be cleaned and transformed into a consistent form.

The cleaning process focused on the following tasks:

1. Loading parquet files in a memory-safe manner.
2. Selecting useful metadata fields.
3. Removing records with invalid or unusable text fields.
4. Standardizing title and abstract text.
5. Extracting broad category labels from arXiv category strings.
6. Removing duplicate IDs.
7. Splitting large cleaned data into manageable parts.
8. Saving cleaning statistics for reproducibility.

The cleaning summary shows that the raw dataset contained **2,982,054** rows and the cleaned corpus contained **2,982,003** rows. This means only a small number of records were removed, which indicates that the dataset was already fairly complete and consistent.

The project uses the first category in the arXiv category string as the primary label. For example:

- `cs.LG` becomes `cs`
- `math.CO` becomes `math`
- `stat.ML` becomes `stat`
- `cond-mat.mtrl-sci` becomes `cond-mat`

Some categories are already broad labels, while others are subcategories. The broad-category mapping reduces complexity and makes classification more meaningful for a general research assistant. Instead of predicting hundreds of fine-grained categories, the system focuses on broader academic domains.

Text preprocessing includes:

- Lowercasing
- Removing URLs and email-like strings
- Removing mathematical symbols where needed
- Removing non-alphabetic noise for classifier input
- Stopword removal
- Optional lemmatization through NLTK

The cleaned text is then used for vectorization and classification. For retrieval, the platform uses sentence embeddings, where aggressive cleaning is less necessary because transformer embedding models can handle natural text directly.

The cleaning process is important because machine learning models are sensitive to noisy input. Poor preprocessing can cause duplicate records, missing labels, irrelevant tokens, and inconsistent feature representations. By saving a cleaning summary artifact, the project keeps the preprocessing stage transparent and reproducible.

<div style="page-break-after: always;"></div>

# 8. EXPLORATORY DATA ANALYSIS

Exploratory data analysis was performed using the EDA showcase notebook:

`notebooks/arxiv_eda_showcase.ipynb`

The notebook performs memory-safe sampling from parquet chunks and generates tables and figures under `artifacts/showcase/`. The EDA process helps understand category distribution, text length, year trends, missing values, and dataset imbalance.

## 8.1 Category Distribution

The cleaned broad category counts show that the dataset is highly imbalanced. The largest categories are:

| Category | Count |
|---|---:|
| `cs` | 733,461 |
| `math` | 570,140 |
| `cond-mat` | 344,417 |
| `astro-ph` | 331,891 |
| `physics` | 207,108 |
| `hep-ph` | 141,477 |
| `quant-ph` | 128,182 |
| `hep-th` | 112,077 |
| `gr-qc` | 70,608 |
| `eess` | 70,243 |
| `stat` | 60,482 |

The top categories dominate the corpus. Computer science and mathematics together form a large share of the dataset. Smaller historical or specialized categories such as `bayes-an`, `ao-sci`, `plasm-ph`, `acc-phys`, and `atom-ph` have very few records. This imbalance has a direct effect on classification performance. Models learn large categories more effectively than rare categories.

The EDA notebook also generated a category distribution figure:

![Top category distribution](artifacts/showcase/figures/eda_top_category_distribution.png)

## 8.2 Text Length Distribution

The platform uses titles and abstracts as the main text source. Abstracts vary in length depending on field, paper type, and author writing style. The EDA notebook generates a token histogram:

![Token count histogram](artifacts/showcase/figures/eda_token_hist_full.png)

Text length analysis is useful because very short abstracts may not contain enough information for classification or retrieval, while extremely long abstracts may need truncation in language model prompts. The platform handles this by using compact text representations for classification and capped context windows for retrieval-augmented generation.

## 8.3 Inferential Analysis

The project includes inferential statistics stored in `artifacts/showcase/tables/inferential_results.json`.

| Test | Result |
|---|---:|
| Rows analyzed | 2,982,003 |
| Categories | 38 |
| Kruskal-Wallis statistic | 62,621.65 |
| Kruskal-Wallis p-value | 0.0 |
| Chi-square statistic | 356,442.11 |
| Chi-square p-value | 0.0 |
| Chi-square dof | 10 |
| Cramer's V | 0.2767 |
| Bootstrap comparison | `cs` vs `math` |
| Observed mean difference | 37.8033 |
| 95% bootstrap CI | [37.3808, 38.2068] |
| Permutation p-value | 0.0 |

These results indicate statistically significant differences in distributions across categories. The Cramer's V value suggests a moderate association in the contingency analysis. This supports the observation that research categories differ meaningfully in text and metadata patterns.

<div style="page-break-after: always;"></div>

# 9. FEATURE ENGINEERING

Feature engineering is the process of converting raw dataset fields into machine learning-ready inputs. In this project, feature engineering has value mainly as a training and reporting component rather than a separate backend runtime model.

The remaining high-value feature engineering artifact is:

`artifacts/feature_engineering/feature_manifest.json`

The feature manifest records:

| Field | Value |
|---|---|
| Memory safe mode | True |
| Training rows | 2,683,334 |
| Testing rows | 298,669 |
| Vectorizer | HashingVectorizer |
| Numeric columns | `token_count`, `title_chars`, `abstract_chars`, `title_abstract_ratio`, `log_token_count` |

The important engineered features are:

1. **Token Count**  
   Measures the number of tokens in the title and abstract text. This helps understand document length and can support models that combine text and numerical features.

2. **Title Characters**  
   Measures title length. Some categories may use shorter or longer titles depending on domain conventions.

3. **Abstract Characters**  
   Measures abstract length. Longer abstracts may provide more classification signal, while short abstracts may be harder to classify.

4. **Title-Abstract Ratio**  
   Captures the relationship between title length and abstract length.

5. **Log Token Count**  
   Reduces the effect of very large token counts and stabilizes the distribution.

For the active backend classifier, the runtime artifacts are:

- `artifacts/classification/classifier.joblib`
- `artifacts/classification/tfidf_vectorizer.joblib`

This means the backend does not separately load the feature-engineering vectorizer files. The project cleanup intentionally removed unused feature-engineering binaries and kept only the manifest because it is useful for documentation and reproducibility.

Feature engineering remains important in the project report because it explains how raw metadata was transformed and how model training was structured. It also shows that the project was designed with memory-safe processing in mind.

<div style="page-break-after: always;"></div>

# 10. MACHINE LEARNING ALGORITHMS

The project compares multiple machine learning approaches for arXiv category classification. The active backend model is the promoted best classifier stored as `classifier.joblib`. The classifier predicts the research category from title and abstract text.

## 10.1 Multinomial Naive Bayes

Multinomial Naive Bayes is a probabilistic classifier commonly used for text classification. It works well with word count, term frequency, and TF-IDF features. The model assumes that features are conditionally independent given the class label. Although this assumption is simplified, Naive Bayes often performs strongly in large-scale text problems.

Advantages:

- Fast training
- Efficient prediction
- Works well with sparse text vectors
- Requires fewer resources than deep learning models
- Easy to deploy in a backend

In the production classification report, Multinomial Naive Bayes was selected as the best model and promoted into the backend classifier path.

## 10.2 Logistic Regression with SGD

Logistic regression is a discriminative classification model. The SGD version trains the model using stochastic gradient descent, making it suitable for large datasets and high-dimensional sparse text vectors. Logistic regression can provide strong classification boundaries, but performance depends heavily on class imbalance handling and regularization.

In the project, logistic SGD was used as a comparison model. The small-sample validation report showed good macro F1 among the newly generated experimental reports, but the production backend remained connected to the full-scale promoted classifier.

## 10.3 Linear SVM with SGD

Linear support vector machines are widely used in text classification. The hinge loss objective aims to create a maximum-margin decision boundary between classes. Like logistic SGD, linear SVM can scale to large sparse feature spaces.

Linear SVM is useful as a baseline because it often performs well when categories are linearly separable in TF-IDF space. However, in heavily imbalanced multi-class settings, it may underperform on rare labels.

## 10.4 Random Forest with SVD

Random Forest is an ensemble of decision trees. It is powerful for tabular data, but raw TF-IDF vectors are very high-dimensional and sparse. Therefore, dimensionality reduction using TruncatedSVD is applied before training the Random Forest model.

This approach creates dense latent semantic features from text. The Random Forest then learns non-linear relationships in the reduced feature space.

Advantages:

- Captures non-linear patterns
- Reduces overfitting by averaging many trees
- Can use dense reduced features

Limitations:

- Heavier model
- Slower training
- Less natural fit for sparse high-dimensional text than linear models

The Random Forest SVD model was useful for experimentation, but it was not retained as a backend runtime artifact after cleanup because only the promoted best classifier is needed by the backend.

<div style="page-break-after: always;"></div>

# 11. SIMILARITY SEARCH AND RETRIEVAL SYSTEM

Classification alone is not sufficient for a research assistant. A user also needs to retrieve relevant papers for a topic or question. For this reason, the platform includes a semantic retrieval system based on sentence embeddings and FAISS.

The connected similarity artifacts are:

| Artifact | Purpose |
|---|---|
| `paper_index.faiss` | Stores paper embedding vectors for nearest-neighbor search |
| `paper_metadata.parquet` | Stores paper metadata aligned with FAISS vector positions |
| `embedding_model_name.joblib` | Stores the embedding model name |
| `metadata_parts/part_00000.parquet` | Metadata partition used for retrieval artifacts |

The embedding model recorded in the artifact is:

`all-MiniLM-L6-v2`

This model generates compact semantic embeddings. The platform normalizes embeddings so that inner product search in FAISS becomes equivalent to cosine similarity. This allows efficient semantic retrieval.

The retrieval process works as follows:

1. The user enters a research query.
2. The query is encoded into a sentence embedding.
3. FAISS searches the paper index for nearest vectors.
4. Retrieved vector IDs are mapped back to rows in `paper_metadata.parquet`.
5. The system returns paper titles, abstracts, authors, categories, years, and similarity scores.
6. The retrieved papers may be reranked using keyword overlap and metadata-aware ranking.
7. The results are used for direct search, RAG answers, trend analysis, or citation proxy analysis.

The FAISS vector store is lazy-loaded. It does not load the full index into memory until a search request is made. This improves startup performance and allows the backend to respond to health checks even before the retrieval index is loaded.

The vector store also validates embedding dimensions. If the embedding model changes after the FAISS index is built, the query vector dimension may not match the index dimension. The platform detects this and raises a clear error. This prevents silent retrieval failures.

Semantic retrieval is one of the most important parts of the project because it allows the system to move beyond keyword matching. A query can retrieve papers with related meaning even if the exact query words are not present in the title.

<div style="page-break-after: always;"></div>

# 12. BACKEND ARCHITECTURE

The backend is implemented using FastAPI. The main composition root is:

`src/research_ai/platform.py`

This file wires together all services and exposes them through the API layer. The architecture is modular, which means each major function is handled by a separate service.

The major backend services are:

| Service | Purpose |
|---|---|
| `ClassifierService` | Predicts arXiv category from title and abstract |
| `EmbeddingService` | Generates sentence embeddings |
| `FaissVectorStore` | Loads and searches FAISS index |
| `HybridSearchService` | Combines semantic search and reranking |
| `ScientificSummarizer` | Summarizes scientific text |
| `SimilarityService` | Compares similarity between two texts |
| `MethodologyExtractor` | Extracts methodology-related signals |
| `RankingService` | Reranks retrieved papers |
| `CitationGraphService` | Produces citation-like related signals |
| `PaperChatService` | Supports question answering over uploaded or loaded papers |
| `TrendAnalysisService` | Analyzes research trends |
| `CitationEngine` | Builds proxy citation relationships |
| `MetadataService` | Analyzes authors, categories, dates, and metadata quality |
| `PythonRunner` | Runs controlled code execution when enabled |
| `KnowledgeGraph` | Tracks concepts across sessions |
| `ConversationStore` | Maintains chat history |

The platform also includes an agent layer:

1. **PlannerAgent**  
   Determines which tools should be used for a user query.

2. **MLExecutionAgent**  
   Executes tools and handles tool errors safely.

3. **EvaluatorAgent**  
   Scores response quality and decides whether retry is needed.

4. **SynthesisAgent**  
   Produces the final answer from grounded outputs.

5. **ResearchOrchestrator**  
   Coordinates planning, execution, evaluation, optional retry, and synthesis.

The backend tool registry exposes 14 tools:

- `hybrid_search`
- `smart_retrieve`
- `classify_query`
- `summarize`
- `methodology_extract`
- `citation_signals`
- `trend_analysis`
- `citation_proxy`
- `metadata_analyse`
- `paper_chat`
- `metadata_rag`
- `python_execute`
- `run_pipeline`
- `conversation`

The system is therefore not just a classifier API. It is a connected research intelligence backend.

<div style="page-break-after: always;"></div>

# 13. MODEL TRAINING AND EVALUATION

The classification task is a supervised multi-class text classification problem. The input is the paper title and abstract, and the output is the broad arXiv category.

According to `artifacts/classification/model_report.json`, the full-scale training and testing setup was:

| Metric | Value |
|---|---:|
| Training rows | 2,683,334 |
| Test rows | 298,669 |
| Best model | Multinomial Naive Bayes |
| Test accuracy | 0.8139847 |
| Macro F1 | 0.34394 |
| Weighted F1 | 0.8052 |

The full-scale model comparison in the production report includes:

| Model | Accuracy | Macro F1 |
|---|---:|---:|
| Multinomial Naive Bayes | 0.8140 | 0.3439 |
| Logistic SGD | 0.5329 | 0.1763 |
| Linear SVM SGD | 0.4732 | 0.1501 |

The promoted backend classifier is Multinomial Naive Bayes. It is connected through:

- `artifacts/classification/classifier.joblib`
- `artifacts/classification/tfidf_vectorizer.joblib`

The model performs very well on high-volume categories. Selected per-category results from the production report are:

| Category | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| `astro-ph` | 0.9623 | 0.9045 | 0.9325 | 33,307 |
| `cond-mat` | 0.8451 | 0.8679 | 0.8563 | 34,446 |
| `cs` | 0.8370 | 0.8972 | 0.8661 | 73,433 |
| `math` | 0.8473 | 0.9111 | 0.8780 | 56,613 |
| `hep-ph` | 0.7993 | 0.8690 | 0.8327 | 14,250 |
| `hep-th` | 0.7297 | 0.8077 | 0.7667 | 11,260 |
| `quant-ph` | 0.8131 | 0.7790 | 0.7957 | 12,738 |
| `physics` | 0.6610 | 0.6506 | 0.6558 | 20,791 |
| `stat` | 0.5792 | 0.6416 | 0.6088 | 6,072 |

The model performs poorly on very rare classes. This is expected because categories with extremely low support provide very little training signal. Examples include `bayes-an`, `plasm-ph`, `atom-ph`, and `supr-con`. These categories have very small test support and often show zero precision/recall.

The difference between accuracy and macro F1 is important. Accuracy is high because the model performs well on large categories. Macro F1 is lower because it gives equal weight to every category, including rare categories. This highlights the class imbalance problem.

<div style="page-break-after: always;"></div>

# 13.1 EXTENDED MACHINE LEARNING METHODOLOGY

This section explains the machine learning part of the project in more depth. Since the project is centered on research-paper intelligence, the most important ML tasks are classification and semantic retrieval. Classification gives the system the ability to identify the broad domain of a paper or query. Retrieval gives the system the ability to find relevant scientific papers from a large corpus.

The machine learning workflow can be understood as a sequence of stages:

1. Dataset ingestion.
2. Data cleaning.
3. Label extraction.
4. Feature engineering.
5. Text vectorization.
6. Train-test split.
7. Model training.
8. Model evaluation.
9. Artifact generation.
10. Backend integration.
11. Runtime inference.

Each stage is important. If the dataset is not cleaned properly, the classifier receives noisy input. If labels are extracted incorrectly, the model learns wrong targets. If vectorization is inconsistent between training and inference, predictions become unreliable. If the backend loads the wrong artifact, the model may fail at runtime. Therefore, the machine learning pipeline is treated as an end-to-end system rather than a single training script.

## 13.1.1 Dataset Ingestion

The dataset is stored in parquet format because parquet is efficient for large tabular data. It supports column-based reading, compression, and faster loading compared to CSV. This is useful because the arXiv metadata corpus has millions of rows.

The project uses chunked parquet files to avoid loading the full dataset into memory at once.

The ingestion stage focuses on reading only the columns required for machine learning:

- `id`
- `title`
- `abstract`
- `categories`
- `update_date`
- `authors`

For classification, the most important columns are `title`, `abstract`, and `categories`.

For retrieval, the most important columns are `id`, `title`, `abstract`, `authors`, `categories`, and `update_date`.

For trend analysis, the `update_date` column is also important because it allows grouping by year.

Reading only required columns reduces memory consumption.

The dataset is large enough that careless loading can crash normal systems.

Therefore, memory-safe processing is a practical requirement, not just an optimization.

## 13.1.2 Label Extraction

arXiv papers can contain multiple categories.

For example, a paper may have categories:

```text
cs.LG stat.ML
```

This means the paper is related to both machine learning in computer science and statistics.

For supervised classification, the project uses the first category as the primary label.

The first category is commonly treated as the main subject category of the paper.

Fine-grained labels are then mapped into broader labels.

Examples:

```text
cs.LG       -> cs
cs.CL       -> cs
math.CO     -> math
stat.ML     -> stat
q-bio.NC    -> q-bio
q-fin.ST    -> q-fin
cond-mat.mtrl-sci -> cond-mat
```

This label mapping helps the classifier solve a more stable problem.

Predicting hundreds of fine-grained labels would require more balanced data and more complex models.

Broad category prediction is more suitable for a research assistant because users often need a domain-level understanding first.

The production classification report contains 38 broad categories.

The class distribution is highly imbalanced.

This imbalance is one of the main reasons macro F1 is lower than weighted F1.

## 13.1.3 Text Input Construction

The classifier uses both title and abstract.

The title is short but highly informative.

The abstract is longer and contains methodology, motivation, and result signals.

The combined input is:

```text
title + " " + abstract
```

This simple concatenation works well because traditional text vectorizers treat the full string as a bag of tokens or n-grams.

The title often contains strong keywords such as:

- transformer
- graph
- quantum
- cosmology
- lattice
- neural
- algebra
- optimization

The abstract provides additional context that helps disambiguate terms.

For example, the word "field" can appear in physics, mathematics, or machine learning.

The abstract helps the model understand which domain is more likely.

## 13.1.4 Text Cleaning for Classification

Scientific text contains symbols, equations, abbreviations, and formatting artifacts.

Examples include:

- LaTeX math fragments
- URLs
- Email addresses
- Greek letters
- Citation markers
- Special characters
- Newline formatting

The cleaning function standardizes text before vectorization.

The main cleaning steps are:

- Convert text to lowercase.
- Remove URLs.
- Remove email-like tokens.
- Remove non-alphabetic noise where appropriate.
- Remove stopwords.
- Remove very short tokens.
- Optionally lemmatize words using NLTK.

Cleaning reduces vocabulary noise.

It also helps the model focus on domain-specific terms.

However, scientific text cleaning must be done carefully.

Some short tokens may be meaningful in science, such as `t5`, `gpt`, `qcd`, or `cnn`.

This is why retrieval tokenization and classification preprocessing may not be identical.

The project treats classification and retrieval differently because they have different requirements.

## 13.1.5 Text Vectorization

Machine learning models cannot directly process raw text.

Text must be converted into numerical vectors.

The production backend classifier uses:

```text
HashingVectorizer(alternate_sign=False, lowercase=False, ngram_range=(1, 2))
```

This means the model uses both unigrams and bigrams.

Unigrams are single words.

Bigrams are two-word phrases.

Examples of unigrams:

- neural
- quantum
- graph
- algebra
- cosmology

Examples of bigrams:

- neural network
- graph neural
- quantum field
- dark matter
- support vector

Bigrams are useful because many research concepts are phrase-based.

For example, "field theory" has a different meaning from the word "field" alone.

The HashingVectorizer is useful for large-scale datasets because it does not store a vocabulary dictionary.

Instead, it maps tokens into fixed feature positions using a hashing function.

Advantages of HashingVectorizer:

- Memory efficient.
- Fast transformation.
- Suitable for very large corpora.
- No need to store vocabulary.
- Handles unseen tokens at inference time.

Limitations of HashingVectorizer:

- Feature names are not easily interpretable.
- Hash collisions can occur.
- It is harder to inspect top words per class.

The project uses `alternate_sign=False`, which keeps feature values non-negative.

This is important because Multinomial Naive Bayes expects non-negative feature values.

## 13.1.6 N-Gram Representation

The vectorizer uses `ngram_range=(1, 2)`.

This allows the classifier to learn both word-level and phrase-level patterns.

For scientific classification, phrase-level patterns are very important.

Examples:

```text
machine learning       -> often cs or stat
quantum field          -> often hep-th or quant-ph
dark matter            -> often astro-ph or hep-ph
neural network         -> often cs, stat, or eess
black hole             -> often gr-qc or astro-ph
partial differential   -> often math
```

Without bigrams, the model may lose important domain context.

With bigrams, the feature space becomes larger.

The hashing approach makes this manageable because it maps features into a fixed-dimensional representation.

## 13.1.7 Classifier Model Type

The saved backend classifier was inspected directly.

The model type is:

```text
sklearn.naive_bayes.MultinomialNB
```

The saved model configuration is:

```text
MultinomialNB(alpha=0.2)
```

The `alpha` parameter controls smoothing.

Smoothing prevents zero probabilities for words that do not appear in a class during training.

This is important in text classification because test examples often contain words or phrases that were rare during training.

With smoothing, the model becomes more stable.

If alpha is too small, the model may overfit.

If alpha is too large, probabilities become too uniform and the model may underfit.

The selected alpha value of 0.2 provides a balance between strict word evidence and smoothing.

## 13.1.8 Why Multinomial Naive Bayes Works Here

Multinomial Naive Bayes is simple but effective for text classification.

It works especially well when:

- The dataset is large.
- The input is text.
- Features are word counts or non-negative text frequencies.
- The classes have domain-specific vocabulary.
- Training speed matters.

The arXiv dataset matches these conditions.

Each research category has distinct vocabulary.

For example:

`cs` papers often contain terms such as:

- algorithm
- neural
- learning
- model
- network
- optimization

`math` papers often contain terms such as:

- theorem
- algebra
- manifold
- proof
- equation
- operator

`astro-ph` papers often contain terms such as:

- galaxy
- cosmological
- stellar
- redshift
- black hole
- dark matter

`hep-ph` papers often contain terms such as:

- particle
- cross section
- collider
- quark
- higgs
- perturbative

Naive Bayes can learn these vocabulary distributions efficiently.

The independence assumption is not fully correct, but with enough data it still produces strong results.

## 13.1.9 Train-Test Split

The production artifact records:

```text
Training rows: 2,683,334
Test rows: 298,669
```

This is roughly a 90/10 split.

A held-out test set is essential because it measures performance on unseen data.

If evaluation is done on training data, the score can be misleading.

The test split helps estimate how the model will perform on new paper metadata.

Because the dataset is large, the test set itself is also large.

This makes the accuracy estimate more reliable.

The challenge is that not all classes have equal support.

Some classes still have very small test counts even in a large dataset.

Therefore, both weighted metrics and macro metrics are needed.

## 13.1.10 Evaluation Metrics

The project uses multiple classification metrics.

Accuracy measures the percentage of correct predictions.

Precision measures how many predicted examples of a class were actually correct.

Recall measures how many actual examples of a class were found by the model.

F1-score is the harmonic mean of precision and recall.

Macro F1 gives equal weight to every class.

Weighted F1 gives higher weight to classes with more examples.

The production results are:

```text
Accuracy: 0.8139847
Macro F1: 0.34394
Weighted F1: 0.8052
```

The high accuracy and weighted F1 show good overall performance.

The lower macro F1 shows weakness on rare classes.

This is expected in imbalanced multi-class classification.

## 13.1.11 Accuracy Interpretation

The accuracy of 81.40% means that about 81 out of every 100 papers in the test set were assigned the correct broad category.

For a 38-class scientific classification problem, this is a strong result.

The model is especially strong for large categories.

However, accuracy alone can hide poor minority-class performance.

For example, if a rare class appears only a few times, the model may ignore it and still maintain high accuracy.

Therefore, accuracy should be reported with macro F1.

## 13.1.12 Macro F1 Interpretation

Macro F1 is lower because it treats every class equally.

This means a rare class with 20 records has the same importance as a large class with 70,000 records.

The macro F1 score of 0.3439 indicates that the model struggles with minority categories.

This does not mean the model is poor overall.

It means the dataset is imbalanced and some categories lack enough training data.

For academic honesty, this distinction is important.

The report should not only highlight the high accuracy.

It should also discuss the lower macro F1 and its cause.

## 13.1.13 Weighted F1 Interpretation

Weighted F1 is 0.8052.

This value is close to the accuracy.

It shows that the model performs well on categories that represent most of the data.

For a deployed research assistant, this means the system will work well for common domains such as computer science, mathematics, physics, astrophysics, and condensed matter.

However, a user working in a rare historical arXiv category may receive less reliable predictions.

## 13.1.14 Strong Categories

The classifier performs strongly on several major categories.

`astro-ph` has F1-score 0.9325.

`math` has F1-score 0.8780.

`cs` has F1-score 0.8661.

`cond-mat` has F1-score 0.8563.

`hep-ph` has F1-score 0.8327.

These categories have distinctive vocabulary and large support.

Large support allows the model to estimate class-specific word probabilities more accurately.

Distinctive vocabulary helps separate one category from another.

## 13.1.15 Weak Categories

The classifier performs weakly on very rare categories.

Examples include:

- `bayes-an`
- `plasm-ph`
- `supr-con`
- `atom-ph`
- `acc-phys`
- `chem-ph`

Some of these categories have extremely low support.

When only a few test examples exist, the metric can become unstable.

When very few training examples exist, the model cannot learn a reliable vocabulary distribution.

Some rare categories may also overlap with larger categories.

For example, physics-related rare categories may share vocabulary with broader physics classes.

The model may therefore predict the larger class instead.

## 13.1.16 Error Sources

Classification errors can occur for several reasons.

First, some papers are genuinely interdisciplinary.

A paper may belong to both computer science and statistics.

If the first arXiv category is `stat.ML` but the title looks like computer science, the model may predict `cs`.

Second, abstracts may use general scientific language.

Words such as "model", "method", "analysis", and "data" appear across many fields.

Third, rare categories may be underrepresented.

Fourth, broad category mapping can merge subfields unevenly.

Fifth, some categories have historical naming differences.

For example, older arXiv categories may not align perfectly with newer category conventions.

## 13.1.17 Confusion Matrix Discussion

The artifact folder contains a confusion matrix image:

`artifacts/showcase/figures/supervised_confusion_matrix.png`

The confusion matrix is useful because it shows where the classifier makes mistakes.

Rows represent actual classes.

Columns represent predicted classes.

Strong diagonal values indicate correct predictions.

Off-diagonal values indicate confusion between categories.

Likely confusion areas include:

- `cs` and `stat`
- `math` and `math-ph`
- `physics` and `cond-mat`
- `hep-th` and `hep-ph`
- `quant-ph` and `physics`
- `astro-ph` and `gr-qc`

These confusions are reasonable because the fields overlap scientifically.

For example, quantum field theory can appear in high energy physics, mathematical physics, and quantum physics.

Machine learning papers can appear in computer science, statistics, and electrical engineering.

## 13.1.18 Model Persistence

After training, the selected classifier is saved using `joblib`.

The backend loads:

```text
artifacts/classification/classifier.joblib
artifacts/classification/tfidf_vectorizer.joblib
```

This is important because the same vectorizer used during training must be used during inference.

If a different vectorizer is used, feature positions will not match the classifier weights.

The model would then produce invalid predictions.

Saving both classifier and vectorizer together ensures consistency.

The classifier service checks that both files exist before reporting itself as ready.

## 13.1.19 Runtime Inference Flow

At runtime, classification works as follows:

1. The user sends a title and abstract.
2. The backend builds a full text string.
3. The text is cleaned.
4. The vectorizer transforms the text into numerical features.
5. The classifier predicts the most likely category.
6. If probability output is available, top class confidence scores are returned.
7. The result is returned as JSON.

The endpoint for this workflow is:

```text
POST /classify
```

The same classifier is also used internally during search.

When search results are retrieved, the query can be classified to infer a preferred category.

The ranking service can use this preferred category to improve result ordering.

Thus, the classifier supports both direct prediction and retrieval ranking.

## 13.1.20 ML Artifact Cleanup

Initially, the artifact folder contained multiple experimental classifier files.

These included older models and newly generated model variants.

However, the backend only uses the promoted best classifier.

Keeping unused model binaries creates confusion.

It also increases project size.

Therefore, the artifacts were cleaned.

The retained classification artifacts are:

- `classifier.joblib`
- `tfidf_vectorizer.joblib`
- `labels.joblib`
- `model_report.json`
- report JSON files under `artifacts/classification/reports/`

The removed files were not connected to the backend.

This cleanup makes the project easier to explain in evaluation.

The report can now clearly state which model is deployed.

## 13.1.21 Why Not Deploy Every Model

Deploying every trained model is not always useful.

It may sound impressive to expose many models, but it can weaken the design.

A production system should usually deploy the selected best model.

Experimental models should be kept only as reports or reproducibility artifacts.

Multiple runtime models increase:

- Loading complexity.
- Memory usage.
- API complexity.
- Testing burden.
- User confusion.

Since this project already has a selected best classifier, the backend should use that model.

The comparison models are useful for the report, not necessarily for deployment.

## 13.1.22 Relationship Between Classification and Retrieval

The classifier and retriever solve different problems.

Classification answers:

```text
What category does this paper or query belong to?
```

Retrieval answers:

```text
Which papers are most relevant to this query?
```

A complete research assistant needs both.

Classification provides domain awareness.

Retrieval provides evidence.

The platform combines them by using classification for category-aware ranking.

For example, if a query is classified as `cs`, retrieved computer science papers may be ranked slightly higher when relevant.

This helps reduce noise in search results.

## 13.1.23 Semantic Embeddings as ML Features

The similarity system uses sentence embeddings.

An embedding is a dense numerical vector that represents meaning.

Unlike bag-of-words features, embeddings can capture semantic similarity.

For example:

```text
"neural machine translation"
```

and

```text
"sequence-to-sequence language translation"
```

may be close in embedding space even if they do not share all exact words.

The platform uses:

```text
all-MiniLM-L6-v2
```

This model is lightweight and suitable for local semantic search.

The embedding vectors are stored in a FAISS index.

At query time, the query embedding is compared to stored paper embeddings.

## 13.1.24 FAISS Index as a Retrieval Model

The FAISS index is not a classifier, but it is a learned-representation retrieval artifact.

It stores vectors generated by an embedding model.

The index enables nearest-neighbor search.

The backend uses:

```text
artifacts/similarity/paper_index.faiss
artifacts/similarity/paper_metadata.parquet
```

The FAISS index stores vector positions.

The metadata parquet file maps those positions back to paper information.

The positional alignment is critical.

If the metadata order changes without rebuilding the index, search results will point to the wrong papers.

The code explicitly documents this invariant.

## 13.1.25 Cosine Similarity

The project normalizes embeddings before search.

For normalized vectors, inner product is equivalent to cosine similarity.

Cosine similarity measures the angle between two vectors.

It is widely used in text retrieval because it focuses on direction rather than magnitude.

If two texts have similar meaning, their embeddings should point in similar directions.

FAISS can compute inner products efficiently.

Therefore, the system gets cosine-like semantic search using a fast vector index.

## 13.1.26 Hybrid Retrieval

Pure semantic retrieval can sometimes retrieve conceptually related but not exact papers.

Pure keyword retrieval can miss semantically related papers.

Hybrid retrieval combines both ideas.

The platform retrieves candidates using semantic similarity.

It then reranks using keyword overlap and metadata signals.

This helps balance meaning-based retrieval with exact term relevance.

For research search, this is important.

Some terms are highly specific.

Examples:

- `BERT`
- `GPT-3`
- `T5`
- `QCD`
- `LHC`
- `Navier-Stokes`

Keyword overlap helps preserve these exact scientific signals.

## 13.1.27 Retrieval Evaluation Considerations

The current project focuses more on classification metrics than retrieval metrics.

However, retrieval quality can be evaluated using:

- Precision@K
- Recall@K
- Mean Reciprocal Rank
- Normalized Discounted Cumulative Gain
- Human relevance judgments

For this project, retrieval validation is supported through tests and manual search behavior.

Future work can add labeled query-paper relevance datasets.

This would allow more formal retrieval evaluation.

## 13.1.28 RAG and Grounding

The platform uses retrieval-augmented generation principles.

The system first retrieves papers.

Then it builds a compact context from paper titles and abstracts.

The language model is instructed to answer only using provided context.

This is important because language models can hallucinate.

In a scientific assistant, hallucination is dangerous.

A fabricated paper title or incorrect claim can mislead users.

The backend includes no-fabrication instructions in the synthesis prompts.

It also has fallback behavior when no retrieval results are found.

## 13.1.29 Evaluation of Grounded Answers

The project includes hallucination-resistance tests.

These tests check that:

- Empty retrieval does not produce fake paper titles.
- Errored tools are excluded from synthesis context.
- The system prompt contains grounding instructions.
- Very short or weak language model outputs fall back to structured answers.

This is an important part of ML system quality.

For modern AI applications, model accuracy alone is not enough.

The system must also avoid unsafe or misleading output.

## 13.1.30 Methodology Extraction

The methodology extraction component identifies method-related signals in paper text.

It can operate on retrieved papers or direct text.

This supports research analysis because users often care not only about what a paper is about, but how the research was conducted.

Methodology signals can include:

- datasets
- experiments
- simulations
- models
- algorithms
- evaluation metrics
- baselines

The current implementation is lightweight and extensible.

Future work can replace or augment it with a trained sequence labeling model.

## 13.1.31 Trend Analysis

Trend analysis uses metadata such as publication year and category.

It helps identify how research activity changes over time.

For example, a user may ask:

```text
How has transformer research changed over recent years?
```

The retrieval system first finds relevant papers.

Trend analysis then summarizes year and category patterns in those papers.

This is not a supervised learning model, but it is an analytical ML-support service.

It turns retrieved data into interpretable research intelligence.

## 13.1.32 Citation Proxy Analysis

The project does not contain complete citation edges.

Therefore, citation analysis is implemented as a proxy service.

It uses metadata, category, and year relationships to infer related signals.

This is useful for demonstration but should not be treated as a true citation graph.

In future work, reference metadata can be added.

Then the system could build real citation networks.

Graph algorithms such as PageRank, community detection, and co-citation analysis could be used.

## 13.1.33 Model Deployment Considerations

The selected classifier is lightweight.

This is a major deployment advantage.

Multinomial Naive Bayes can make predictions quickly.

HashingVectorizer transformation is also fast.

This makes the `/classify` endpoint suitable for interactive use.

The retrieval stack is heavier because it uses sentence embeddings and FAISS.

However, FAISS is optimized for vector search.

The embedding model is loaded lazily and cached.

The backend also caches single-query embeddings through an LRU cache.

This improves repeated-query performance.

## 13.1.34 Model Version Warning

When inspecting the classifier artifact, scikit-learn produced a version warning.

The model was saved using a different scikit-learn version than the current environment.

This is a known limitation of pickle/joblib model persistence.

It does not necessarily mean the model is broken.

However, for production deployment, the training and inference environments should use matching package versions.

A good practice is to record:

- Python version
- scikit-learn version
- numpy version
- pandas version
- vectorizer configuration
- model hyperparameters

This improves reproducibility.

## 13.1.35 Why Classical ML Is Acceptable

A common question is why the project uses classical ML instead of only deep learning.

Classical ML is appropriate here because:

- The dataset is very large.
- Text categories have strong vocabulary signals.
- Training deep models is expensive.
- Classical models are easier to deploy locally.
- The project also uses embeddings for semantic retrieval.

The platform is not limited to classical ML.

It combines classical classification with neural embeddings and optional language models.

This hybrid design is practical.

It gives the speed of classical ML and the semantic power of embeddings.

## 13.1.36 Possible ML Improvements

The classifier can be improved in future iterations.

Possible improvements include:

- Hierarchical classification.
- Balanced class sampling.
- Class-weighted training.
- Rare-class grouping.
- Transformer fine-tuning.
- Better text normalization for scientific abbreviations.
- Feature combination of title, abstract, year, and author metadata.
- Calibration of probabilities.
- Top-k category prediction.

Top-k prediction would be especially useful.

For interdisciplinary papers, the correct label may be among the top three predictions even if not ranked first.

This would better match how research categories actually work.

## 13.1.37 ML Contribution of the Project

The machine learning contribution of the project is not only model training.

It includes:

- Large-scale corpus preparation.
- Broad category mapping.
- Memory-safe processing.
- Text vectorization.
- Supervised classification.
- Semantic embedding index.
- Hybrid retrieval.
- RAG grounding.
- Evaluation reporting.
- Artifact cleanup.
- Backend deployment of selected models.

This makes the project a complete applied ML system.

The final report should emphasize this complete pipeline.

Many projects stop after model accuracy.

This project goes further by connecting the model to a working backend and research assistant workflow.

<div style="page-break-after: always;"></div>

# 14. COMPARATIVE ANALYSIS

The project compared multiple classification approaches. The final backend uses only the best promoted model, but the comparison is useful for understanding model behavior.

## 14.1 Production-Scale Comparison

The production-scale report shows that Multinomial Naive Bayes performed best among the models recorded in `model_report.json`.

Multinomial Naive Bayes achieved the strongest accuracy. This result is reasonable because Naive Bayes is highly effective for large-scale sparse text classification. It benefits from millions of training examples and can learn strong word-category associations.

Logistic SGD and Linear SVM SGD performed lower in the recorded production report. These models can be strong in many text classification tasks, but their performance depends on training settings, class weighting, convergence, vectorizer configuration, and class imbalance. In this project's recorded full-scale result, they did not outperform Multinomial Naive Bayes.

## 14.2 Additional Small-Sample Validation Reports

After artifact cleanup and reproducibility improvements, small-sample validation reports were generated under:

`artifacts/classification/reports/`

These reports were based on smaller samples and are not the main production benchmark. They are useful for showing that the training scripts can run and evaluate multiple algorithms. The small-sample reports include:

| Model | Sample Accuracy | Sample Macro F1 |
|---|---:|---:|
| Multinomial Naive Bayes | 0.7176 | 0.4380 |
| Logistic SGD | 0.7583 | 0.5756 |
| Linear SVM SGD | 0.7048 | 0.4903 |
| Random Forest SVD | 0.7011 | 0.5301 |

These small-sample results should not be confused with the production-scale model report. Because sample size and category coverage are different, the numbers are not directly comparable to the full dataset evaluation.

## 14.3 Final Model Selection

The final backend model selection is based on the existing production artifact:

`artifacts/classification/model_report.json`

This artifact identifies **Multinomial Naive Bayes** as the best model and the backend uses the corresponding promoted classifier artifact. The cleanup removed unconnected experimental model binaries from `artifacts/classification/models/`, leaving only the runtime classifier pair and high-value reports.

This is a good design decision for the project because the backend does not need to load every experimental model. It only needs the selected production classifier. Keeping unused model binaries increases storage size and creates confusion. A clean artifact structure makes the project easier to explain, maintain, and deploy.

<div style="page-break-after: always;"></div>

# 15. SYSTEM IMPLEMENTATION

The system is implemented as a Python project with a modular source layout. The major directories are:

| Directory | Purpose |
|---|---|
| `src/research_ai/api/` | FastAPI routes and schemas |
| `src/research_ai/agents/` | Planner, executor, evaluator, synthesis, retrieval agents |
| `src/research_ai/ml_models/` | Classifier, summarizer, similarity, ranking, methodology extraction |
| `src/research_ai/retrieval/` | Embeddings, FAISS vector store, hybrid search, rerankers |
| `src/research_ai/research/` | Citation engine, metadata analysis, paper ingestion, trend analysis |
| `src/research_ai/memory/` | Conversation store, session memory, knowledge graph |
| `src/research_ai/execution/` | Sandbox and pipeline execution |
| `frontend/` | Static frontend interface |
| `artifacts/` | Model, retrieval, EDA, and report artifacts |
| `notebooks/` | EDA notebook |

The backend startup creates a `ResearchAIPlatform` object. This object wires all services together. Runtime model loading is mostly lazy. The classifier artifacts are loaded when classification is first requested. The FAISS index is loaded when search is first requested. The embedding model is also loaded lazily.

Lazy loading improves developer experience because the server can start even if heavyweight models are not immediately needed. It also allows health-check endpoints to run faster.

The platform includes production hardening notes in the API code. These notes mention:

- CORS configuration
- Upload size limits
- Need for rate limiting in production
- Need for authentication in production
- Safe error handling
- Secret redaction

The implementation also includes safety features:

- Upload size limit for paper chat
- Sandboxed code execution disabled by default
- AST-level sandbox validation
- Retrieval grounding in synthesis prompts
- No-fabrication instructions for RAG answers
- Error isolation in tool execution
- Retry logic when retrieval quality is low

The implementation is therefore designed not only to work, but to handle failure cases gracefully.

<div style="page-break-after: always;"></div>

# 16. API AND USER WORKFLOW

The backend exposes multiple API endpoints. The most important endpoints include:

| Endpoint | Purpose |
|---|---|
| `GET /health` | Check system health |
| `GET /stats` | Return index/model/runtime statistics |
| `POST /classify` | Classify paper title/abstract |
| `POST /search` | Search papers using hybrid retrieval |
| `POST /summarize` | Summarize scientific text |
| `POST /similarity` | Compare two text inputs |
| `POST /metadata/analyse` | Analyze metadata for paper lists |
| `POST /citation/proxy` | Build proxy citation signals |
| `POST /pipeline/run` | Run predefined multi-step pipelines |
| `POST /ask` | Agentic research question answering |
| `POST /chat/message` | Unified conversational research assistant |
| `POST /chat/upload` | Upload PDF/text for paper chat |
| `POST /chat/load-arxiv` | Load paper by arXiv ID |
| `POST /chat/ask` | Ask questions about a loaded paper |

The typical user workflow is:

1. User opens the frontend or sends a request to the backend.
2. User asks a research question such as "Find papers about graph neural networks for molecular property prediction."
3. The planner decides that retrieval and synthesis are required.
4. The system generates an embedding for the query.
5. FAISS retrieves relevant paper metadata.
6. The classifier may infer a preferred category for reranking.
7. The ranking service reorders results.
8. The synthesis service generates a grounded response with paper evidence.
9. The response is returned with sources, confidence, conversation ID, and metadata.

For classification workflow:

1. User provides title and abstract.
2. The classifier vectorizer transforms text.
3. The promoted classifier predicts the category.
4. Top confidence scores are returned if supported by the model.

For paper chat workflow:

1. User uploads a PDF/text file or loads an arXiv paper by ID.
2. The paper is chunked and indexed into session memory.
3. User asks questions about the paper.
4. Relevant chunks are retrieved.
5. The system answers using the paper context.

The workflow demonstrates that the system supports both corpus-level research discovery and document-level paper understanding.

<div style="page-break-after: always;"></div>

# 17. TESTING AND VALIDATION

The project includes a test suite under the `tests/` directory. The tests cover important system behavior and reduce the risk of silent failures.

Major tested areas include:

1. **Pipeline Integrity**  
   Tests verify that the system follows the intended ML-first pipeline. The agent should not fabricate answers when retrieval fails. Tool outputs should flow correctly from one step to another.

2. **Retrieval Components**  
   Tests validate BM25 tokenization, alphanumeric token handling, scoring behavior, reranker weights, FAISS dimension validation, embedding cache behavior, and arXiv ID normalization.

3. **Sandbox Security**  
   Tests verify that dangerous Python operations are blocked. Imports, dynamic execution, dunder access, global statements, and suspicious names are checked by the sandbox validator.

4. **Hallucination Resistance**  
   Tests ensure that synthesis falls back safely when retrieval context is missing or when the language model output is too short or unreliable.

5. **Evaluator Agent**  
   Tests validate scoring and retry decisions.

6. **Chunking and Retrieval**  
   Tests check that paper chunks and retrieval behavior remain stable.

The project is careful about not requiring full artifact loading for every test. Some tests use mocks so that the test suite can run even when heavyweight FAISS or model files are absent. Performance tests that require real artifacts are marked separately.

Validation was also done through artifact checks. The cleaned artifact structure confirms that only backend-used and high-value files remain. The active runtime model files are:

- Classification: `classifier.joblib`, `tfidf_vectorizer.joblib`
- Similarity: `paper_index.faiss`, `paper_metadata.parquet`, `embedding_model_name.joblib`

The EDA notebook and report tables validate dataset-level assumptions. The model report validates classification performance. The backend code validates artifact loading and service wiring.

<div style="page-break-after: always;"></div>

# 18. DEPLOYMENT

The project is designed as a local-first FastAPI application. The basic deployment workflow is:

```bash
pip install -r requirements.txt
```

```bash
python -m uvicorn research_ai.api.main:app --host 127.0.0.1 --port 8000
```

The repository also includes startup scripts such as:

- `start.sh`
- `start_ollama.bat`
- `start_all_ollama_app.bat`

The backend reads settings from environment variables. Important variables include:

| Variable | Purpose |
|---|---|
| `LLM_BACKEND` | Select cloud or local LLM backend |
| `CLOUD_LLM_PROVIDER` | Select Groq, OpenRouter, Google, or Ollama-style provider |
| `GROQ_API_KEY` | API key for Groq |
| `OPENROUTER_API_KEY` | API key for OpenRouter |
| `GOOGLE_API_KEY` | API key for Google |
| `EMBEDDING_MODEL` | Sentence embedding model |
| `ARTIFACTS_ROOT` | Artifact directory |
| `ENABLE_PYTHON_EXECUTION` | Enable or disable sandboxed execution |
| `PYTHON_EXEC_TIMEOUT` | Execution timeout |

The backend is designed to start even if the cloud LLM key is not immediately available. The cloud client is created lazily when it is actually needed. This prevents startup failure in local-only environments.

For production deployment, the following improvements would be required:

- Restrict CORS origins
- Add authentication
- Add rate limiting
- Use HTTPS
- Store large artifacts in managed storage
- Add monitoring and structured logging
- Use a production ASGI server setup
- Consider a persistent vector database if the corpus grows significantly

The current deployment is appropriate for an academic project, local demonstration, and controlled evaluation.

<div style="page-break-after: always;"></div>

# 19. RESULTS AND DISCUSSION

The project achieved its main goal of creating an integrated research intelligence platform. It successfully combines a trained classifier, semantic retrieval, backend services, and agentic orchestration.

The dataset cleaning stage produced a cleaned corpus of **2,982,003** unique records across **38** broad categories. The EDA showed strong class imbalance, with computer science, mathematics, condensed matter, astrophysics, and physics dominating the corpus. This imbalance explains why the classifier performs better on large categories than on rare categories.

The active production classifier achieved **81.40% accuracy** on a large test set of **298,669** rows. This is a strong result considering the large number of categories and the noisy nature of broad arXiv labels. The weighted F1 score of approximately **0.8052** confirms that the model performs well for the majority of papers.

The macro F1 score is lower at **0.3439**, which indicates weaker performance on rare classes. This is an important finding. In real-world scientific corpora, rare categories may not have enough examples for reliable prediction. The project documents this limitation clearly rather than hiding it behind accuracy alone.

The semantic retrieval system adds major practical value. A classifier can tell what category a paper belongs to, but retrieval helps users find relevant papers. FAISS and sentence embeddings allow the system to search by meaning. This makes the platform closer to a real research assistant.

The backend architecture is one of the strongest parts of the project. Instead of building isolated notebooks, the work integrates models into APIs and workflows. The project supports:

- Direct classification
- Semantic search
- Similarity comparison
- Scientific summarization
- Metadata analysis
- Citation proxy analysis
- Trend analysis
- Paper chat
- Conversational research assistance
- Multi-step pipelines

The artifact cleanup also improved the project quality. Before cleanup, there were many unconnected model binaries in `artifacts/`, which could confuse evaluation. After cleanup, the project keeps only backend-used models and high-value report artifacts. This makes the final project easier to explain and more production-like.

Overall, the results show that the project is not just a machine learning experiment. It is a working applied AI system for scientific research discovery.

<div style="page-break-after: always;"></div>

# 20. LIMITATIONS

Despite its strengths, the project has several limitations.

1. **Class Imbalance**  
   The dataset is highly imbalanced. Large categories such as `cs`, `math`, and `cond-mat` dominate the corpus, while rare categories have very few records. This reduces macro F1 and makes rare-category classification difficult.

2. **Metadata-Based Corpus**  
   The main large-scale corpus uses metadata, especially title and abstract. Full-paper PDFs are not indexed for every arXiv paper. Some research questions require full methodology, experiments, tables, or references, which may not be available in the abstract.

3. **Limited Citation Ground Truth**  
   The project includes citation proxy analysis, but the metadata artifacts do not contain complete citation edges. Therefore, citation analysis is approximate rather than a true citation graph.

4. **Local Resource Constraints**  
   Large-scale model training and FAISS indexing require memory and storage. The project uses memory-safe processing, but full retraining can still be resource-intensive.

5. **Language Model Dependency**  
   Some synthesis and summarization features depend on a cloud or local LLM. If the LLM backend is unavailable, the system falls back to structured answers, but output quality may decrease.

6. **No Production Authentication**  
   The current backend does not implement full authentication or user management. This is acceptable for a project prototype but not for public deployment.

7. **No Real-Time Dataset Updates**  
   arXiv updates frequently. The current artifact index reflects the dataset version used during preprocessing. A production system would require scheduled updates and re-indexing.

8. **Limited Frontend Evaluation**  
   The project includes a frontend, but the main evaluation focuses on backend models and services. A complete usability study is outside the scope.

9. **Rare Category Performance**  
   Several rare classes show zero or near-zero precision and recall. This should be addressed through label grouping, resampling, or hierarchical classification in future work.

These limitations do not reduce the value of the project; they identify realistic future improvements.

<div style="page-break-after: always;"></div>

# 21. FUTURE SCOPE

The project can be extended in several directions.

1. **Hierarchical Classification**  
   Instead of predicting all categories directly, the system can first predict broad domains and then predict subcategories. This would better match the structure of arXiv labels.

2. **Transformer-Based Classifier**  
   A fine-tuned transformer model such as SciBERT, DistilBERT, or MiniLM could improve classification accuracy, especially if trained with balanced sampling.

3. **Better Rare-Class Handling**  
   Techniques such as class weighting, oversampling, focal loss, or rare-class grouping can improve macro F1.

4. **Full-Text Indexing**  
   The system can be extended to index full PDF content, not only metadata. This would improve methodology extraction, experiment comparison, and detailed paper chat.

5. **Real Citation Graph**  
   If reference metadata is added, the citation proxy system can be replaced with a true citation graph.

6. **Incremental Index Updates**  
   arXiv updates weekly. The system could support incremental ingestion and index updates without rebuilding everything from scratch.

7. **Advanced Vector Search**  
   For larger corpora, FAISS approximate indexes such as HNSW or IVF can improve retrieval speed.

8. **User Personalization**  
   The platform could remember user interests and recommend papers based on past queries.

9. **Research Dashboard**  
   Trend charts, category heatmaps, author networks, and topic timelines could be added to the frontend.

10. **Production Deployment**  
   Authentication, rate limiting, logging, monitoring, HTTPS, and containerized deployment can make the system production-ready.

11. **Evaluation of RAG Quality**  
   The generated answers can be evaluated using factuality metrics, source coverage, and human review.

12. **Notebook-to-Report Automation**  
   The EDA notebook can automatically export tables and figures into the final project report.

Future work can therefore improve both model quality and system usability.

<div style="page-break-after: always;"></div>

# 22. CONCLUSION

The **Research AI Intelligence Platform** successfully demonstrates how machine learning, natural language processing, semantic search, and backend engineering can be combined to build a useful research discovery system.

The project uses the large-scale arXiv metadata dataset and processes nearly three million records. It performs exploratory data analysis, feature engineering, supervised classification, semantic indexing, and backend integration. The active backend classifier is a promoted Multinomial Naive Bayes model that achieved approximately **81.40% accuracy** on a large held-out test set. The semantic retrieval system uses FAISS and sentence embeddings to retrieve relevant papers based on meaning.

The platform is designed as a practical system rather than only a notebook experiment. It includes APIs for classification, search, summarization, similarity comparison, metadata analysis, citation proxy analysis, trend analysis, paper chat, and agentic research question answering. The architecture supports planning, execution, evaluation, and synthesis, making the system flexible for different research workflows.

The project also follows good artifact management. Unused model binaries were removed, while backend-used and report-relevant artifacts were retained. This makes the system cleaner and easier to explain. The EDA notebook and saved artifacts provide evidence for dataset understanding and model evaluation.

The final result is a local-first AI research assistant that can help users explore scientific literature more efficiently. While there are limitations related to class imbalance, metadata-only indexing, and production hardening, the project provides a strong foundation for future development.

In conclusion, this project shows that a well-designed machine learning platform can reduce the manual effort involved in literature exploration and provide intelligent support for students, researchers, and academic users.

<div style="page-break-after: always;"></div>

# 23. REFERENCES

1. arXiv Dataset, Kaggle Metadata Dataset.  
   https://www.kaggle.com/datasets/Cornell-University/arxiv

2. arXiv.org, Open Access Scientific Repository.  
   https://arxiv.org/

3. scikit-learn Documentation: Naive Bayes, SGDClassifier, Metrics, Text Feature Extraction.  
   https://scikit-learn.org/

4. FAISS Documentation: Efficient Similarity Search and Clustering of Dense Vectors.  
   https://faiss.ai/

5. SentenceTransformers Documentation.  
   https://www.sbert.net/

6. FastAPI Documentation.  
   https://fastapi.tiangolo.com/

7. pandas Documentation.  
   https://pandas.pydata.org/

8. PyArrow Documentation.  
   https://arrow.apache.org/docs/python/

9. Hugging Face Transformers Documentation.  
   https://huggingface.co/docs/transformers/

10. Manning, C. D., Raghavan, P., and Schutze, H. *Introduction to Information Retrieval*. Cambridge University Press.

11. Jurafsky, D., and Martin, J. H. *Speech and Language Processing*. Stanford University draft.

12. Lewis, P. et al. "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks." NeurIPS, 2020.

13. Johnson, J., Douze, M., and Jegou, H. "Billion-scale similarity search with GPUs." IEEE Transactions on Big Data, 2019.

14. Reimers, N., and Gurevych, I. "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks." EMNLP-IJCNLP, 2019.

15. Project artifacts and implementation files from the local repository:
    - `artifacts/classification/model_report.json`
    - `artifacts/processed/cleaning_summary.json`
    - `artifacts/feature_engineering/feature_manifest.json`
    - `artifacts/showcase/tables/`
    - `notebooks/arxiv_eda_showcase.ipynb`
    - `src/research_ai/platform.py`
    - `src/research_ai/api/main.py`
