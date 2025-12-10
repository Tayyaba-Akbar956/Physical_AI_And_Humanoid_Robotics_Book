# Tasks: RAG Chatbot for Physical AI & Humanoid Robotics Textbook

Feature: Implementation of RAG (Retrieval-Augmented Generation) Chatbot embedded within the Physical AI & Humanoid Robotics online textbook that serves as an interactive study companion for students

## Dependencies

- User Story 1 (General Question Answering) must be complete before other stories can be fully tested
- User Story 2 (Text Selection-Based Queries) requires foundational infrastructure but is independent of other stories
- User Story 3 (Conversational Context) and User Story 4 (Module-Aware Context) are independent but benefit from shared foundational components

## Parallel Execution Examples

Per User Story 1:
- [P] Backend API development (models, services, endpoints) can run in parallel
- [P] Frontend widget development can run in parallel with backend
- [P] Content embedding pipeline can run in parallel with API development

Per User Story 2:
- [P] Text selection detection service can be developed in parallel with UI components
- [P] API development can run in parallel with frontend work

## Implementation Strategy

1. **MVP**: Implement User Story 1 with basic RAG functionality (T001-T040)
2. **Phase 2**: Add text selection functionality (User Story 2) (T041-T070)
3. **Phase 3**: Add conversational context (User Story 3) (T071-T095)
4. **Phase 4**: Add module-aware context (User Story 4) (T096-T120)
5. **Phase 5**: Polish and cross-cutting concerns (T121-T135)

## Phase 1: Setup

### Goals
- Initialize project structure
- Set up development environment
- Configure dependencies

### Tasks
- [X] T001 Create backend directory structure: /backend/src/models, /backend/src/services, /backend/src/api
- [X] T002 Create frontend directory structure: /frontend/rag-widget
- [X] T003 Create requirements.txt with FastAPI, OpenAI Agents SDK (with GEMINI), Qdrant client, Neon Postgres driver, Google GenAI client
- [X] T004 Create .env file structure for API keys and database connections
- [X] T005 Create .gitignore for Python/JavaScript project with venv, node_modules, .env exclusions

## Phase 2: Foundational Components

### Goals
- Set up database models and schema
- Establish database connections
- Set up content processing pipeline
- Implement basic agent framework

### Tasks
- [X] T006 [P] Create Student model in backend/src/models/student.py
- [X] T007 [P] Create TextbookContent model in backend/src/models/textbook_content.py
- [X] T008 [P] Create ChatSession model in backend/src/models/chat_session.py
- [X] T009 [P] Create Message model in backend/src/models/message.py
- [X] T010 [P] Create SelectedText model in backend/src/models/selected_text.py
- [X] T011 [P] Create VectorIndexEntry model in backend/src/models/vector_index_entry.py
- [X] T012 [P] Set up Neon Postgres connection in backend/src/db/connection.py
- [X] T013 [P] Implement database initialization in backend/src/db/init_db.py
- [X] T014 [P] Set up Qdrant client connection in backend/src/db/qdrant_client.py
- [X] T015 [P] Create content scraper module in backend/src/content_scraper.py
- [X] T016 [P] Create content parser module in backend/src/content_parser.py
- [X] T017 [P] Create content chunker module for GEMINI embeddings in backend/src/content_chunker.py
- [X] T018 [P] Create Gemini embedding generator module in backend/src/embedding_generator.py
- [X] T019 [P] Create Qdrant uploader module for GEMINI embeddings in backend/src/qdrant_uploader.py
- [X] T020 [P] Create content processing pipeline with Gemini embeddings in backend/src/content_pipeline.py
- [X] T021 [P] Create basic RAG agent service with GEMINI integration in backend/src/services/rag_agent.py
- [X] T022 [P] Create semantic search service using GEMINI embeddings in backend/src/services/semantic_search.py
- [X] T023 [P] Create session management service in backend/src/services/session_manager.py
- [X] T024 [P] Create text selection service in backend/src/services/text_selection.py

## Phase 3: User Story 1 - General Question Answering (P1)

### Story Goal
As a student studying the Physical AI & Humanoid Robotics textbook, I want to ask questions about textbook content and receive accurate, book-grounded answers so that I can understand the concepts without searching through multiple chapters.

### Independent Test Criteria
Students can ask any question about the textbook content and receive a response that cites specific modules/chapters from the book. If the information isn't in the book, the system clearly states this.

### Tasks
- [X] T025 [P] [US1] Create chat query endpoint in backend/src/api/chat_endpoints.py
- [X] T026 [P] [US1] Implement validation for chat query requests in backend/src/api/chat_endpoints.py
- [X] T027 [P] [US1] Create basic response generation in backend/src/services/rag_agent.py
- [X] T028 [P] [US1] Implement citation generation in RAG agent service
- [X] T029 [P] [US1] Add hallucination prevention to RAG agent
- [X] T030 [P] [US1] Create semantic search functionality using GEMINI embeddings in backend/src/services/semantic_search.py
- [X] T031 [P] [US1] Integrate GEMINI-based semantic search with RAG agent
- [X] T032 [P] [US1] Implement textbook terminology usage in RAG agent
- [X] T033 [P] [US1] Create response formatting with citations in backend/src/services/rag_agent.py
- [X] T034 [P] [US1] Add response length control (150-300 words) in backend/src/services/rag_agent.py
- [X] T035 [P] [US1] Create fallback response when content not found in backend/src/services/rag_agent.py
- [X] T036 [P] [US1] Implement response quality validation in backend/src/services/rag_agent.py
- [X] T037 [P] [US1] Create response streaming endpoint in backend/src/api/chat_endpoints.py
- [X] T038 [P] [US1] Implement WebSocket connection handling in backend/src/api/chat_endpoints.py
- [X] T039 [P] [US1] Add performance monitoring to chat endpoints
- [X] T040 [P] [US1] Create basic chat widget UI in frontend/rag-widget/chat-widget.js

## Phase 4: User Story 2 - Text Selection-Based Queries (P1)

### Story Goal
As a student reading the Physical AI & Humanoid Robotics textbook, I want to highlight/select text passages and ask specific questions about that selected passage so that I can get targeted explanations without copy-pasting or describing what I'm reading.

### Independent Test Criteria
Students can select 20+ characters of text, see an intuitive interface appear, type their question about the selected text, and receive a response that prioritizes explaining the selected passage while potentially enriching with related content.

### Tasks
- [X] T041 [P] [US2] Create text selection detection endpoint in backend/src/api/text_selection_endpoints.py
- [X] T042 [P] [US2] Implement validation for selected text (20+ characters) in backend/src/api/text_selection_endpoints.py
- [X] T043 [P] [US2] Add selected text storage to SelectedText model
- [X] T044 [P] [US2] Create text selection API contract validation
- [X] T045 [P] [US2] Implement text selection query handling with GEMINI embeddings in RAG agent
- [X] T046 [P] [US2] Modify RAG agent to prioritize selected text content
- [X] T047 [P] [US2] Add selected text reference to Message model
- [X] T048 [P] [US2] Create text selection enrichment with related content in RAG agent
- [X] T049 [P] [US2] Create floating UI for "Ask about this" button in frontend/rag-widget/text-selector.js
- [X] T050 [P] [US2] Implement text selection capture mechanism in frontend/rag-widget/text-selector.js
- [X] T051 [P] [US2] Add text selection validation in frontend (20+ characters)
- [X] T052 [P] [US2] Create text selection UI display logic in frontend/rag-widget/text-selector.js
- [X] T053 [P] [US2] Implement communication between frontend and backend for text selection
- [X] T054 [P] [US2] Add loading indicators for text selection queries in frontend
- [X] T055 [P] [US2] Create error handling for text selection in frontend
- [X] T056 [P] [US2] Integrate text selection with chat widget interface
- [X] T057 [P] [US2] Add responsive design for text selection interface
- [X] T058 [P] [US2] Create text selection accessibility features
- [X] T059 [P] [US2] Add performance optimization for text selection detection
- [X] T060 [P] [US2] Create text selection styling in frontend/rag-widget/styles.css
- [X] T061 [P] [US2] Add text selection position calculation
- [X] T062 [P] [US2] Implement text selection persistence between selections
- [X] T063 [P] [US2] Create selected text highlighting visualization
- [X] T064 [P] [US2] Add text selection context preservation in frontend
- [X] T065 [P] [US2] Create text selection metadata tracking
- [X] T066 [P] [US2] Add text selection analytics tracking
- [X] T067 [P] [US2] Create text selection cache optimization
- [X] T068 [P] [US2] Implement boundary detection for text selection
- [X] T069 [P] [US2] Add text selection history in frontend interface
- [X] T070 [P] [US2] Create text selection accessibility keyboard navigation

## Phase 5: User Story 3 - Conversational Context (P2)

### Story Goal
As a student having an ongoing conversation with the chatbot, I want to ask follow-up questions without re-explaining the context so that I can explore concepts naturally like in a study group discussion.

### Independent Test Criteria
Students can engage in multi-turn conversations where follow-up questions like "explain that differently" or "what about for humanoid robots?" correctly reference previous exchanges without requiring re-explanation.

### Tasks
- [X] T071 [P] [US3] Enhance ChatSession model to track conversation context
- [X] T072 [P] [US3] Implement conversation history tracking in backend/src/services/session_manager.py
- [X] T073 [P] [US3] Add conversation context to Message model
- [X] T074 [P] [US3] Create conversation context management in RAG agent
- [X] T075 [P] [US3] Implement context window management (5-10 exchanges) in RAG agent
- [X] T076 [P] [US3] Add follow-up question resolution in RAG agent
- [X] T077 [P] [US3] Create context-aware response generation in RAG agent
- [X] T078 [P] [US3] Add "explain that differently" resolution in RAG agent
- [X] T079 [P] [US3] Implement pronoun resolution ("that", "this") in RAG agent
- [X] T080 [P] [US3] Add topic tracking across conversation in RAG agent
- [X] T081 [P] [US3] Create conversation state validation
- [X] T082 [P] [US3] Implement conversation history API endpoints
- [X] T083 [P] [US3] Add conversation history retrieval to frontend
- [X] T084 [P] [US3] Create conversation history display in frontend/rag-widget/chat-widget.js
- [X] T085 [P] [US3] Add conversation history persistence
- [X] T086 [P] [US3] Implement conversation context serialization
- [X] T087 [P] [US3] Create conversation context caching
- [X] T088 [P] [US3] Add conversation history navigation in frontend
- [X] T089 [P] [US3] Implement conversation memory optimization
- [X] T090 [P] [US3] Add conversation context boundaries (start/end)
- [X] T091 [P] [US3] Create conversation context reset functionality
- [X] T092 [P] [US3] Implement conversation analytics and tracking
- [X] T093 [P] [US3] Add conversation context debugging tools
- [X] T094 [P] [US3] Create conversation context visualization in frontend
- [X] T095 [P] [US3] Add performance monitoring for multi-turn conversations

## Phase 6: User Story 4 - Module-Aware Context (P3)

### Story Goal
As a student currently viewing a specific module of the textbook, I want the chatbot to prioritize content from the current module in its answers so that I get relevant information for what I'm currently studying.

### Independent Test Criteria
The chatbot knows which module/chapter the student is viewing and prioritizes answers from that module, mentioning when information comes from other modules and suggesting related concepts.

### Tasks
- [X] T096 [P] [US4] Add current module context tracking to ChatSession model
- [X] T097 [P] [US4] Implement module prioritization in GEMINI-based semantic search service
- [X] T098 [P] [US4] Modify RAG agent to prioritize current module content
- [X] T099 [P] [US4] Add module context to chat query endpoint
- [X] T100 [P] [US4] Create module cross-reference detection in RAG agent
- [X] T101 [P] [US4] Implement related concept suggestion in RAG agent
- [X] T102 [P] [US4] Add module context validation
- [X] T103 [P] [US4] Create module context API for frontend communication
- [X] T104 [P] [US4] Implement module context preservation across sessions
- [X] T105 [P] [US4] Add module-aware response formatting in RAG agent
- [X] T106 [P] [US4] Create module navigation suggestions in responses
- [X] T107 [P] [US4] Implement module context switching detection
- [X] T108 [P] [US4] Add module context tracking to frontend widget
- [X] T109 [P] [US4] Create module context visualization in frontend
- [X] T110 [P] [US4] Implement module context persistence in frontend
- [X] T111 [P] [US4] Add module-specific analytics tracking
- [X] T112 [P] [US4] Create module context debugging tools
- [X] T113 [P] [US4] Implement module content filtering in GEMINI-based semantic search
- [X] T114 [P] [US4] Add module-related concept detection in RAG agent
- [X] T115 [P] [US4] Create module cross-reference citations
- [X] T116 [P] [US4] Implement module context boundary detection
- [X] T117 [P] [US4] Add module content relevance scoring
- [X] T118 [P] [US4] Create module-specific response customization
- [X] T119 [P] [US4] Add module context performance optimization
- [X] T120 [P] [US4] Implement module context fallback handling

## Phase 7: Polish & Cross-Cutting Concerns

### Goals
- Add health checks and monitoring
- Implement security measures
- Add comprehensive error handling
- Create embed script for frontend integration
- Add testing and quality measures

### Tasks
- [X] T121 Create health check endpoint in backend/src/api/health_endpoints.py
- [X] T122 Add dependency status reporting to health check
- [X] T123 Create monitoring and metrics for response times
- [X] T124 Implement logging throughout the application
- [X] T125 Add error handling middleware in FastAPI
- [X] T126 Create comprehensive error responses in API endpoints
- [X] T127 Add security headers to API responses
- [X] T128 Implement rate limiting for API endpoints
- [X] T129 Create embed script for textbook website in frontend/rag-widget/embed-script.js
- [X] T130 Add responsive design to chat widget styling
- [X] T131 Create accessibility features for chat widget
- [X] T132 Add internationalization support if needed
- [X] T133 Create comprehensive tests for all user stories
- [X] T134 Add performance testing scenarios
- [X] T135 Document the API endpoints and usage