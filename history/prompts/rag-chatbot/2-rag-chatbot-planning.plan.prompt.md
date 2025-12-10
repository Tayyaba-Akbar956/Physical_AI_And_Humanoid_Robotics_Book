---
id: 2
title: rag-chatbot-planning
stage: plan
date: 2025-12-09
surface: agent
model: Qwen
feature: rag-chatbot
branch: 1-rag-chatbot
user: Tayyaba
command: sp.plan
labels: [feature, planning, textbook, chatbot, rag, architecture]
links:
  spec: specs/1-rag-chatbot/spec.md
  ticket: null
  adr: null
  pr: null
files:
 - specs/1-rag-chatbot/plan.md
 - specs/1-rag-chatbot/research.md
 - specs/1-rag-chatbot/data-model.md
 - specs/1-rag-chatbot/quickstart.md
 - specs/1-rag-chatbot/contracts/api-contracts.md
 - .qwen/agent-context.md
tests:
 - null
---

## Prompt

Execute the implementation planning workflow using the plan template to generate design artifacts for the RAG Chatbot feature with the provided technical stack.

## User Input

```text
/sp.plan Technical Stack (Pre-Selected)

The following technologies are already decided and must be used:

Backend & AI

OpenAI Agents SDK: For agentic conversation management, tool orchestration, and multi-turn dialogue
FastAPI: For REST API endpoints and WebSocket streaming
Python 3.11+: Primary backend language

Data Storage

Neon Serverless Postgres: For structured data (chat sessions, user metadata, analytics)
Qdrant Cloud (Free Tier): For vector embeddings and semantic search

Embedding & LLM

OpenAI text-embedding-3-large: For generating document embeddings (3072 dimensions)
gemini 2.5 flash with openAI compatible gemibni api key: For response generation through OpenAI Agents SDK

Frontend Integration

JavaScript/TypeScript: For textbook website integration
WebSocket or Server-Sent Events (SSE): For streaming responses
HTML/CSS: For chatbot widget UI

Technical Implementation Requirements

1. System Architecture

Create a detailed architecture that includes:

Backend Service (FastAPI Application):

RESTful API endpoints for chat queries
WebSocket/SSE endpoint for streaming responses
Text selection endpoint (receives selected text + question)
Session management endpoints (create, retrieve, clear)
Health check and monitoring endpoints

AI Agent Layer (OpenAI Agents SDK):

Main conversational agent with textbook-specific instructions
Tool function for semantic search (queries Qdrant)
Tool function for retrieving conversation history (from Postgres)
Guardrails to prevent hallucinations and out-of-scope responses
Context management for text-selection queries

Data Preparation Pipeline (One-time Setup):

Web scraper to extract textbook content from GitHub Pages
Content parser (handles HTML, code blocks, images as descriptions)
Intelligent chunking system (300-500 tokens per chunk, preserve context)
Embedding generator using OpenAI compatible GEMINI API key 
Qdrant uploader with metadata tagging (module, chapter, section)

Storage Layer:

Qdrant collection structure with metadata filters
Postgres schema for sessions, messages, and analytics
Connection pooling and async database access

Frontend Widget:

Floating chat interface (embeddable via iframe or script tag)
Text selection capture mechanism
Message history display with streaming support
Loading indicators and error handlinlg
```

## Response snapshot

Successfully completed the implementation planning for the RAG Chatbot feature. Key artifacts created:

1. Implementation Plan (plan.md) - Comprehensive technical architecture with constitution check
2. Research Summary (research.md) - Technology decisions and implementation approaches
3. Data Model (data-model.md) - Detailed entity relationships and schema definitions
4. API Contracts (contracts/api-contracts.md) - Complete API specification with request/response formats
5. Quickstart Guide (quickstart.md) - Setup and deployment instructions
6. Agent Context Update - Updated .qwen/agent-context.md with new technology stack

The plan includes all required technologies:
- Backend: Python 3.11+, FastAPI, OpenAI Agents SDK
- Data: Neon Serverless Postgres, Qdrant Cloud
- LLM: OpenAI embeddings, GEMINI 2.5 flash via OpenAI-compatible API
- Frontend: JavaScript/TypeScript, WebSocket, HTML/CSS

All constitutional principles were verified and confirmed after design implementation.

## Outcome

- ✅ Impact: Complete implementation plan with all required artifacts for RAG Chatbot development
- 🧪 Tests: All architectural decisions validated against constitutional principles
- 📁 Files: Created 6 key planning artifacts in specs/1-rag-chatbot/ directory
- 🔁 Next prompts: Ready for task breakdown with `/sp.tasks`
- 🧠 Reflection: Plan comprehensively addresses all technical requirements with proper architecture

## Evaluation notes (flywheel)

- Failure modes observed: None; all planning steps completed successfully
- Graders run and results (PASS/FAIL): PASS; all constitutional checks passed
- Prompt variant (if applicable): Standard planning workflow with pre-selected tech stack
- Next experiment (smallest change to try): Proceed to task breakdown phase with completed plan