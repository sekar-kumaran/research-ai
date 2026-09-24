# Storage and Persistence Audit

## READ-ONLY DEPLOYMENT ARTIFACTS
These artifacts are safely bundled into the HF Space and read statically into RAM. They require no persistence beyond the repository itself.
- `hf_full/artifacts/classifier.joblib`
- `hf_full/artifacts/tfidf_vectorizer.joblib`
- `hf_full/artifacts/labels.joblib`
- `hf_full/artifacts/paper_index.faiss`
- `hf_full/artifacts/paper_metadata.parquet`
- `hf_full/artifacts/embedding_model_name.joblib`

## MUTABLE APPLICATION STATE
These elements change dynamically during runtime. Because Hugging Face Space local storage is ephemeral, anything written locally will be lost on container restart.

1. **Chat History (`ConversationStore`)**
   - Current Location: In-memory dictionary mapped by `conversation_id`.
   - Action Required: Since the dictionary is only in RAM, chat sessions will be destroyed if the Space goes to sleep. For a true production system, this requires an external DB (e.g., PostgreSQL). However, for the context of this deployable HF Space experiment, keeping it in memory is acceptable as long as it's understood that sessions do not survive cold restarts.

2. **File Uploads (`/chat/upload`)**
   - Current Location: FastAPI `UploadFile.file` writes to standard `/tmp/` when files exceed the spooled memory size.
   - Action Required: The application must properly garbage collect and delete uploaded PDFs after parsing, as the ephemeral disk is limited. 

3. **Knowledge Graph (`KnowledgeGraph`)**
   - Current Location: In-memory networkx graph instance.
   - Action Required: Transitory by design. Safe to lose on restart.

## CONCLUSION
No permanent database integration is inherently *required* to make the backend function for a short-lived stateless API demonstration. However, real users will lose their conversation history if the Space sleeps. The ephemeral disk is only used securely for temporary PDF parsing.
