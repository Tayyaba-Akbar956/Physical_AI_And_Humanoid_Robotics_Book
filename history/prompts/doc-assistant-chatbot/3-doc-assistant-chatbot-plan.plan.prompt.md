---
id: 3
title: doc-assistant-chatbot-plan
stage: plan
date: '2025-12-09'
model: 'Qwen'
feature: 'doc-assistant-chatbot'
branch: '2-doc-assistant-chatbot'
user: 'tayyaba'
command: '/sp.plan'
labels:
  - 'plan'
  - 'chatbot'
  - 'documentation'
  - 'ai'
  - 'rag'
links:
  - 'null'
files:
  - C:\Users\tayyaba\Desktop\PIAIC\PHYSICAL_AI_BOOK\specs\2-doc-assistant-chatbot\plan.md
  - C:\Users\tayyaba\Desktop\PIAIC\PHYSICAL_AI_BOOK\specs\2-doc-assistant-chatbot\research.md
  - C:\Users\tayyaba\Desktop\PIAIC\PHYSICAL_AI_BOOK\specs\2-doc-assistant-chatbot\data-model.md
  - C:\Users\tayyaba\Desktop\PIAIC\PHYSICAL_AI_BOOK\specs\2-doc-assistant-chatbot\quickstart.md
  - C:\Users\tayyaba\Desktop\PIAIC\PHYSICAL_AI_BOOK\specs\2-doc-assistant-chatbot\contracts\api-contract.md
  - C:\Users\tayyaba\Desktop\PIAIC\PHYSICAL_AI_BOOK\specs\2-doc-assistant-chatbot\agent-context-update.md
tests: []
prompt: |
  ---
  description: Execute the implementation planning workflow using the plan template to generate design artifacts.
  ---

  ## User Input

  ```text
  $ARGUMENTS
  ```

  You **MUST** consider the user input before proceeding (if not empty).

  ## Outline

  1. **Setup**: Run `.specify/scripts/powershell/setup-plan.ps1 -Json` from repo root and parse JSON for FEATURE_SPEC, IMPL_PLAN, SPECS_DIR, BRANCH. For single quotes in args like "I'm Groot", use escape syntax: e.g 'I'\''m Groot' (or double-quote if possible: "I'm Groot").

  2. **Load context**: Read FEATURE_SPEC and `.specify/memory/constitution.md`. Load IMPL_PLAN template (already copied).

  3. **Execute plan workflow**: Follow the structure in IMPL_PLAN template to:
     - Fill Technical Context (mark unknowns as "NEEDS CLARIFICATION")
     - Fill Constitution Check section from constitution
     - Evaluate gates (ERROR if violations unjustified)
     - Phase 0: Generate research.md (resolve all NEEDS CLARIFICATION)
     - Phase 1: Generate data-model.md, contracts/, quickstart.md
     - Phase 1: Update agent context by running the agent script
     - Re-evaluate Constitution Check post-design

  4. **Stop and report**: Command ends after Phase 2 planning. Report branch, IMPL_PLAN path, and generated artifacts.

  ## Phases

  ### Phase 0: Outline & Research

  1. **Extract unknowns from Technical Context** above:
     - For each NEEDS CLARIFICATION → research task
     - For each dependency → best practices task
     - For each integration → patterns task

  2. **Generate and dispatch research agents**:

     ```text
     For each unknown in Technical Context:
       Task: "Research {unknown} for {feature context}"
     For each technology choice:
       Task: "Find best practices for {tech} in {domain}"
     ```

  3. **Consolidate findings** in `research.md` using format:
     - Decision: [what was chosen]
     - Rationale: [why chosen]
     - Alternatives considered: [what else evaluated]

  **Output**: research.md with all NEEDS CLARIFICATION resolved

  ### Phase 1: Design & Contracts

  **Prerequisites:** `research.md` complete

  1. **Extract entities from feature spec** → `data-model.md`:
     - Entity name, fields, relationships
     - Validation rules from requirements
     - State transitions if applicable

  2. **Generate API contracts** from functional requirements:
     - For each user action → endpoint
     - Use standard REST/GraphQL patterns
     - Output OpenAPI/GraphQL schema to `/contracts/`

  3. **Agent context update**:
     - Run `.specify/scripts/powershell/update-agent-context.ps1 -AgentType qwen`
     - These scripts detect which AI agent is in use
     - Update the appropriate agent-specific context file
     - Add only new technology from current plan
     - Preserve manual additions between markers

  **Output**: data-model.md, /contracts/*, quickstart.md, agent-specific file

  ## Key rules

  - Use absolute paths
  - ERROR on gate failures or unresolved clarifications

  ---

  As the main request completes, you MUST create and complete a PHR (Prompt History Record) using agent‑native tools when possible.

  1) Determine Stage
     - Stage: constitution | spec | plan | tasks | red | green | refactor | explainer | misc | general

  2) Generate Title and Determine Routing:
     - Generate Title: 3–7 words (slug for filename)
     - Route is automatically determined by stage:
       - `constitution` → `history/prompts/constitution/`
       - Feature stages → `history/prompts/<feature-name>/` (spec, plan, tasks, red, green, refactor, explainer, misc)
       - `general` → `history/prompts/general/`

  3) Create and Fill PHR (Shell first; fallback agent‑native)
     - Run: `.specify/scripts/bash/create-phr.sh --title "<title>" --stage <stage> [--feature <name>] --json`
     - Open the file and fill remaining placeholders (YAML + body), embedding full PROMPT_TEXT (verbatim) and concise RESPONSE_TEXT.
     - If the script fails:
       - Read `.specify/templates/phr-template.prompt.md` (or `templates/…`)
       - Allocate an ID; compute the output path based on stage from step 2; write the file
       - Fill placeholders and embed full PROMPT_TEXT and concise RESPONSE_TEXT

  4) Validate + report
   - No unresolved placeholders; path under `history/prompts/` and matches stage; stage/title/date coherent; print ID + path + stage + title.
   - On failure: warn, don't block. Skip only for `/sp.phr`.


  /sp.plan Build a RAG chatbot for Docusaurus documentation with the following architecture:

  ### System Architecture

  Create a three-layer system:
  1. Frontend: Docusaurus with integrated React chat widget
  2. Backend: Node.js API with three services (Ingestion, Query, Chat)
  3. Data Layer: Qdrant Cloud (vectors), Neon DB (metadata), Gemini API (via OpenAI SDK)

  ### Technology Stack Selection

  **Frontend Layer:**
  - Base: Docusaurus v3 with React 18+ and TypeScript
  - Chat UI: styled-components for scoped styling
  - Animations: Framer Motion for Tamagotchi character
  - State: OpenAI Agent SDK built-in session module (handles conversation history)
  - API Communication: React Query
  - Rendering: react-markdown with remark-gfm for messages
  - Syntax Highlighting: Prism.js (matching Docusaurus theme)

  **Backend Layer:**
  - Runtime: Node.js 20+ LTS with TypeScript
  - Framework: Express.js or Fastify
  - AI Integration: OpenAI SDK v4+ configured for Gemini endpoint
  - Agent SDK: Use built-in threads/sessions for conversation management
  - Vector Client: @qdrant/js-client-rest
  - Database Client: @neondatabase/serverless
  - Utilities: tiktoken (token counting), zod (validation)

  **AI Components:**
  - Embeddings: Gemini text-embedding-004 (768 dimensions)
  - Chat Model: gemini-2.0-flash-exp (fast response times)
  - Session Management: OpenAI Agent SDK threads API
  - RAG Pattern: Function calling tool for document retrieval

  **Data Storage:**
  - Vector Database: Qdrant Cloud with HNSW indexing, cosine similarity
  - Relational Database: Neon DB (serverless Postgres) for metadata and backup
  - Session Storage: OpenAI Agent SDK managed (with threadId persistence)

  ### Service Architecture

  **Ingestion Service:**
  Purpose: Transform Docusaurus content into searchable vectors
  Process: Scan MDX files → Parse and chunk content → Generate embeddings → Store in Qdrant and Neon
  Chunking Strategy: Semantic chunks of 500-1000 tokens with 10% overlap

  **Query Service:**
  Purpose: Find relevant documentation chunks for user questions
  Process: Generate query embedding → Search Qdrant with filters → Rank by relevance → Return top-k results

  **Chat Service:**
  Purpose: Handle conversational interactions using Agent SDK
  Process: Manage thread sessions → Process tool calls for RAG retrieval → Stream responses → Log to Neon for backup

  ### Database Schemas

  **Qdrant Collection:**
  - Vectors: 768-dimensional embeddings
  - Payload: doc_id, chunk_id, content, title, url, section, timestamp
  - Index: HNSW with quantization for performance

  **Neon Database Tables:**
  - documents: Store document metadata (id, title, url, content, timestamps)
  - conversations: Track sessions (id, thread_id from Agent SDK, user_id, timestamps)
  - messages: Backup messages (id, thread_id, role, content, context_type, selected_text, sources, timestamp)

  ### Agent SDK Integration Pattern

  **Session Management:**
  - Create persistent threads using client.beta.threads.create()
  - Store threadId in localStorage for session continuity
  - Agent SDK automatically maintains conversation history
  - No manual message array management needed

  **Tool Definition:**
  - Define RAG retrieval as a function tool
  - Tool triggers automatic vector search when agent needs context
  - Agent processes retrieved documents and generates answers
  - Sources automatically tracked in tool outputs

  **Conversation Flow:**
  - User message added via threads.messages.create()
  - Run created with threads.runs.create() and assistant_id
  - Agent calls search_documentation tool when needed
  - Tool outputs submitted back to agent
  - Agent generates final response with context

  ### Component Integration

  **Text Selection Detection:**
  - Add mouseup event listener to document
  - Capture window.getSelection() on user highlight
  - Pass selected text as context to chat API
  - Agent SDK instructions modified to handle selected text context

  **Chat Widget:**
  - Collapsible floating button in bottom-right corner
  - Expands to full chat interface
  - Tamagotchi character with state-based animations (idle, thinking, responding, happy)
  - Message history loaded from Agent SDK thread
  - Progressive streaming of responses with markdown rendering

  **Theming:**
  - Purple (#8B5CF6) to black (#000000) gradients
  - CSS custom properties for theme consistency
  - Smooth color transitions on animations
  - Dark mode optimized with proper contrast ratios

  ### Deployment Strategy

  **Frontend Deployment:**
  - Host Docusaurus on Vercel or Netlify
  - CDN for static assets and chat widget bundle
  - Environment variables for API endpoint URLs

  **Backend Deployment:**
  - Deploy API to Railway, Render, or Fly.io
  - Alternative: Serverless functions on Vercel
  - Configure CORS for Docusaurus domain
  - Set up connection pooling for databases

  **External Services:**
  - Qdrant Cloud: Managed vector database cluster
  - Neon DB: Serverless Postgres with autoscaling
  - Environment variables: GEMINI_API_KEY, QDRANT_URL, QDRANT_API_KEY, NEON_DATABASE_URL

  ### Performance Optimizations

  **Response Speed:**
  - Implement Redis caching for frequent queries (5min TTL)
  - Cache embeddings generation results
  - Stream AI responses for perceived speed
  - Use progressive rendering for markdown

  **Resource Efficiency:**
  - Lazy load chat widget on user interaction
  - Code-split animation libraries
  - Optimize Qdrant queries with proper filtering
  - Use semantic chunking to respect document structure

  **Scalability:**
  - Connection pooling for database queries
  - Rate limiting (100 requests/min per IP)
  - Horizontal scaling support for API services
  - CDN caching for static documentation content

  ### Security Measures

  **API Protection:**
  - Rate limiting with IP-based throttling
  - Input sanitization to prevent injection attacks
  - CORS restricted to production Docusaurus domain
  - API key rotation policy

  **Data Privacy:**
  - No PII collection without explicit consent
  - Optional conversation history (user opt-in)
  - Clear data retention and deletion policies
  - Secure environment variable management
response: |
  # Implementation Planning Completed Successfully

  I've completed the implementation planning for the RAG chatbot for Docusaurus documentation. The following artifacts have been created:

  1. **Implementation Plan** (plan.md):
     - Technical context with language versions and dependencies
     - Constitution compliance check
     - Project structure definition
     - Complexity tracking

  2. **Research Document** (research.md):
     - Technology selection rationale (Express.js, OpenAI SDK, Qdrant, Neon DB)
     - Implementation approach decisions
     - Resolution of technical unknowns
     - Alternative analysis for each major decision

  3. **Data Model** (data-model.md):
     - Entity definitions (ChatMessage, Conversation, DocumentationChunk, Document, UserAction)
     - Field specifications with validation rules
     - Relationships between entities
     - State transitions

  4. **API Contracts** (contracts/api-contract.md):
     - Chat service endpoints (start-thread, send-message, get-thread)
     - Query service endpoints (search)
     - Ingestion service endpoints (process, update, remove document)
     - Common error formats and rate limits

  5. **Quickstart Guide** (quickstart.md):
     - Prerequisites and installation steps
     - Environment configuration
     - Running the application
     - Usage guide for the chat interface
     - API integration examples
     - Troubleshooting tips

  6. **Agent Context Update** (agent-context-update.md):
     - Documentation of the agent context update process

  The planning phase is complete and all required artifacts have been generated. The implementation plan follows the architectural requirements specified by the user, including:
  - Three-layer system (frontend, backend, data layer)
  - Technology stack with Docusaurus, React, Node.js, Qdrant Cloud, Neon DB
  - Service architecture with Ingestion, Query, and Chat services
  - Data models appropriate for the requirements
  - API contracts following REST principles
  - Performance and security considerations

  The next step would be to generate the tasks for implementation using `/sp.tasks`.
outcome: |
  Successfully completed the implementation planning for the RAG chatbot with all required artifacts generated.
evaluation: |
  The planning phase was executed completely with all artifacts created as required. The technical decisions align with the architectural requirements specified by the user. All constitutional principles were considered during planning.
---