# Research Summary: RAG Chatbot for Physical AI & Humanoid Robotics Textbook

## Technology Decisions

### Decision: Backend Framework Choice
**Rationale:** FastAPI was selected as the backend framework based on the user's technical stack requirements. It offers excellent performance, built-in async support, and automatic OpenAPI documentation generation, which is ideal for the RAG chatbot system that needs to handle concurrent requests efficiently.

**Alternatives considered:** 
- Flask: Less performant, fewer built-in features
- Django: Overly complex for this microservice use case
- Node.js/Express: Would conflict with Python-based AI tooling

### Decision: AI Agent Framework
**Rationale:** OpenAI Agents SDK was selected as specified in the technical requirements. This provides advanced agentic capabilities, conversation management, and tool orchestration that are essential for the RAG system to interact with the textbook content and perform semantic search.

**Alternatives considered:**
- LangChain: Would require more custom development for conversation management
- Custom implementation: Higher complexity and maintenance burden
- Anthropic Claude: Doesn't align with specified OpenAI Agents SDK requirement

### Decision: Vector Database
**Rationale:** Qdrant Cloud was selected as specified in the technical requirements. It provides efficient semantic search capabilities essential for the RAG system to find relevant textbook passages based on student queries.

**Alternatives considered:**
- Pinecone: Similar capabilities but different pricing model
- Weaviate: Good alternative but Qdrant was specified
- ChromaDB: Open-source option but lacks cloud scalability

### Decision: LLM Provider
**Rationale:** GEMINI 2.5 flash was selected via OpenAI-compatible API as specified in the requirements. This provides cost-effective, high-quality responses with excellent reasoning capabilities for educational content.

**Alternatives considered:**
- OpenAI GPT models: More expensive than GEMINI flash
- Anthropic Claude: Different API compatibility requirements
- Open-source models: Would require self-hosting and optimization

### Decision: Database
**Rationale:** Neon Serverless Postgres was selected for structured data storage as specified in the requirements. It provides serverless scaling, built-in connection pooling, and compatibility with Python async frameworks.

**Alternatives considered:**
- AWS RDS: Would require more infrastructure management
- Supabase: Good alternative but Neon was specified
- SQLite: Insufficient for concurrent access requirements

## Implementation Approaches

### Approach: Textbook Content Processing Pipeline
**Rationale:** A one-time setup pipeline will be implemented to extract, parse, and embed textbook content. This ensures the RAG system has access to all 6 modules of content with proper hierarchy preservation.

**Key components:**
- Web scraping module to extract content from existing textbook website
- Content parser to handle HTML, code blocks, and images
- Intelligent chunking system to maintain context boundaries
- Embedding generator using GEMINI embeddings API

### Approach: Frontend Integration
**Rationale:** The chatbot will be integrated as a floating widget embeddable in the existing textbook website. This approach maintains the reading flow while providing interactive capabilities.

**Key components:**
- JavaScript/TypeScript widget for embedding
- Text selection capture mechanism
- WebSocket/SSE streaming for real-time responses
- Responsive design for mobile compatibility

### Approach: Conversation Management
**Rationale:** The system will maintain conversation history and context for each session, enabling follow-up questions and multi-turn dialogues as specified in the requirements.

**Key components:**
- Session state management in Postgres
- Context window management for multi-turn conversations
- Module-aware context for prioritizing relevant content