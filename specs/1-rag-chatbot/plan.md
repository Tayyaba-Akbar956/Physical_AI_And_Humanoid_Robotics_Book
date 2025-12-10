# Implementation Plan: RAG Chatbot for Physical AI & Humanoid Robotics Textbook

**Branch**: `1-rag-chatbot` | **Date**: 2025-12-09 | **Spec**: [link](spec.md)
**Input**: Feature specification from `/specs/1-rag-chatbot/spec.md`

**Note**: This template is filled in by the `/sp.plan` command. See `.specify/templates/commands/plan.md` for the execution workflow.

## Summary

Implementation of a RAG (Retrieval-Augmented Generation) Chatbot embedded within the Physical AI & Humanoid Robotics online textbook. This system uses OpenAI Agents SDK for conversational management, FastAPI for the backend, and semantic search capabilities to provide students with accurate, book-grounded answers. The implementation includes a frontend widget for text selection and question-asking, with responses grounded in textbook content with proper citations.

## Technical Context

**Language/Version**: Python 3.11+, JavaScript/TypeScript
**Primary Dependencies**: OpenAI Agents SDK (using GEMINI 2.5 flash via OpenAI-compatible API), FastAPI, Qdrant client library, Neon serverless PostgreSQL driver
**Storage**: Neon Serverless Postgres (structured data), Qdrant Cloud (vector embeddings)
**Testing**: pytest for backend/unit tests, integration tests for full query flow
**Target Platform**: Linux server (backend), Web browser (frontend widget)
**Project Type**: Web application with embedded frontend component
**Performance Goals**: Response time under 2 seconds for 90th percentile, handle 100+ concurrent users
**Constraints**: <1.5s text selection queries, <5s general queries, zero hallucinations, 95%+ accuracy
**Scale/Scope**: 500 total students across multiple cohorts, 6 textbook modules with content

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

**EDUCATIONAL CLARIFICATION** - PASSED: The chatbot will provide educational content from the textbook, explaining concepts with proper citations and using textbook terminology.

**TECHNICAL ACCURACY** - PASSED: Using OpenAI Agents SDK with GEMINI 2.5 flash through OpenAI-compatible API and GEMINI embeddings ensures current, accurate responses based on textbook content.

**STRUCTURED LEARNING PATH** - PASSED: The system maintains module awareness and can prioritize content from current module being studied.

**PRACTICAL ORIENTATION** - PASSED: The implementation includes a practical frontend widget that students can use while reading, with text selection capability.

**ACCESSIBILITY** - PASSED: The web-based implementation will work across different hardware setups, with responsive design for various devices.

**COMPREHENSIVE COVERAGE** - PASSED: The system processes all 6 modules of textbook content, preserving hierarchy, code blocks, and diagrams as descriptions.

**PROFESSIONAL QUALITY** - PASSED: Using industry-standard tools like FastAPI, PostgreSQL, and Qdrant ensures professional-grade infrastructure.

**DOCUSAURUS STANDARDS** - PASSED: The frontend widget will integrate with the existing Docusaurus-based textbook website with responsive design.

**GITHUB READINESS** - PASSED: Implementation will follow clean code practices and proper documentation standards.

**DEPLOYMENT REQUIREMENTS** - PASSED: The architecture separates frontend and backend, making it suitable for deployment to GitHub Pages with backend API endpoint.

## Post-Design Constitution Re-evaluation

After implementing the detailed design (data models, API contracts, and architecture), the system continues to comply with all constitutional principles:

**EDUCATIONAL CLARIFICATION** - CONFIRMED: Data models include proper citation tracking, API contracts support educational responses with module/chapter references.

**TECHNICAL ACCURACY** - CONFIRMED: The architecture with vector search and semantic matching ensures accurate retrieval of textbook content.

**STRUCTURED LEARNING PATH** - CONFIRMED: The module context field in sessions and API contracts ensure proper learning path maintenance.

**PRACTICAL ORIENTATION** - CONFIRMED: The text selection API and frontend widget implementation provide practical, hands-on interaction.

**ACCESSIBILITY** - CONFIRMED: The web-based architecture and responsive frontend design ensure accessibility across devices.

**COMPREHENSIVE COVERAGE** - CONFIRMED: The content processing pipeline and vector database ensure all textbook content is covered.

**PROFESSIONAL QUALITY** - CONFIRMED: Using FastAPI, PostgreSQL, and Qdrant ensures professional-grade backend infrastructure.

**DOCUSAURUS STANDARDS** - CONFIRMED: The JavaScript widget integrates seamlessly with the existing Docusaurus website.

**GITHUB READINESS** - CONFIRMED: Clean code structure with proper documentation and testing approach ensures GitHub readiness.

**DEPLOYMENT REQUIREMENTS** - CONFIRMED: Separation of frontend and backend with API-based communication supports deployment requirements.

## Project Structure

### Documentation (this feature)

```text
specs/1-rag-chatbot/
├── plan.md              # This file (/sp.plan command output)
├── research.md          # Phase 0 output (/sp.plan command)
├── data-model.md        # Phase 1 output (/sp.plan command)
├── quickstart.md        # Phase 1 output (/sp.plan command)
├── contracts/           # Phase 1 output (/sp.plan command)
└── tasks.md             # Phase 2 output (/sp.tasks command - NOT created by /sp.plan)
```

### Source Code (repository root)

```text
backend/
├── src/
│   ├── models/
│   ├── services/
│   │   ├── rag_agent.py          # OpenAI Agents SDK implementation
│   │   ├── text_selection.py     # Text selection capture service
│   │   ├── semantic_search.py    # Qdrant-based search service  
│   │   ├── content_processor.py  # Textbook content processing
│   │   └── session_manager.py    # Conversation session management
│   ├── api/
│   │   ├── chat_endpoints.py     # FastAPI chat endpoints
│   │   ├── text_selection_endpoints.py  # Text selection endpoints
│   │   └── session_endpoints.py  # Session management endpoints
│   └── main.py                   # Application entry point
├── tests/
│   ├── unit/
│   ├── integration/
│   └── data/
└── requirements.txt               # Python dependencies

frontend/
├── rag-widget/
│   ├── chat-widget.js            # Main chat widget implementation
│   ├── text-selector.js          # Text selection capture mechanism
│   ├── styles.css                # Widget styling
│   └── embed-script.js           # Script for embedding widget in textbook
└── contracts/                    # API contract definitions
```

**Structure Decision**: Web application structure selected with separate backend and frontend directories. The backend uses FastAPI to serve REST and WebSocket endpoints, while the frontend provides a JavaScript-based widget for embedding in the existing textbook website.

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| [e.g., 4th project] | [current need] | [why 3 projects insufficient] |
| [e.g., Repository pattern] | [specific problem] | [why direct DB access insufficient] |