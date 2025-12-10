# API Contracts: RAG Chatbot for Physical AI & Humanoid Robotics Textbook

## Chat Endpoint API

### POST /api/chat/query
Initiates a new chat or continues an existing conversation with the RAG chatbot.

**Request:**
```json
{
  "session_id": "uuid-string",
  "message": "How do ROS 2 nodes communicate with each other?",
  "module_context": "module-2-ros2",
  "selected_text": null
}
```

**Response:**
```json
{
  "session_id": "uuid-string",
  "response_id": "uuid-string",
  "message": "According to Module 2, Section 2.3, ROS 2 nodes communicate using a publish-subscribe model where nodes exchange messages through topics...",
  "citations": [
    {
      "module": "module-2-ros2",
      "chapter": "ch-2-3",
      "section": "pub-sub-model"
    }
  ],
  "timestamp": "2025-12-09T10:30:00Z"
}
```

**Error Responses:**
- 400: Invalid request format
- 404: Session not found
- 500: Internal server error

### POST /api/chat/text-selection-query
Handles queries based on selected text from the textbook.

**Request:**
```json
{
  "session_id": "uuid-string",
  "selected_text": "The Quantum Stabilization Matrix uses inverse kinematic chains",
  "question": "Can you explain this in simpler terms?",
  "module_context": "module-5-humanoids"
}
```

**Response:**
```json
{
  "session_id": "uuid-string",
  "response_id": "uuid-string",
  "message": "Based on the passage you selected from Module 5, the Quantum Stabilization Matrix refers to... [simplified explanation]. This concept relates to the balance control systems described earlier in Module 5.2...",
  "citations": [
    {
      "module": "module-5-humanoids",
      "chapter": "ch-5-2",
      "section": "balance-control"
    }
  ],
  "timestamp": "2025-12-09T10:32:00Z"
}
```

**Error Responses:**
- 400: Selected text less than 20 characters or invalid format
- 404: Session not found
- 500: Internal server error

## Session Management API

### POST /api/session/create
Creates a new chat session for a student.

**Request:**
```json
{
  "student_id": "uuid-string",
  "module_context": "module-3-simulation"
}
```

**Response:**
```json
{
  "session_id": "uuid-string",
  "created_at": "2025-12-09T10:30:00Z",
  "module_context": "module-3-simulation"
}
```

**Error Responses:**
- 400: Invalid request format
- 500: Internal server error

### GET /api/session/{session_id}
Retrieves the details of a specific session.

**Response:**
```json
{
  "session_id": "uuid-string",
  "student_id": "uuid-string",
  "created_at": "2025-12-09T10:30:00Z",
  "updated_at": "2025-12-09T10:45:00Z",
  "current_module_context": "module-3-simulation",
  "is_active": true
}
```

**Error Responses:**
- 404: Session not found
- 500: Internal server error

### POST /api/session/{session_id}/clear
Clears the conversation history for a session, starting a fresh conversation.

**Response:**
```json
{
  "session_id": "uuid-string",
  "message": "Conversation history cleared. Starting new conversation."
}
```

**Error Responses:**
- 404: Session not found
- 500: Internal server error

## WebSocket API for Streaming Responses

### WebSocket /ws/chat/{session_id}
Establishes a WebSocket connection for streaming chat responses.

**Message Format (Client to Server):**
```json
{
  "type": "chat_message",
  "session_id": "uuid-string",
  "message": "What are the key concepts in this chapter?",
  "module_context": "module-2-ros2"
}
```

**Message Format (Server to Client):**
```json
{
  "type": "streaming_response",
  "response_id": "uuid-string",
  "session_id": "uuid-string",
  "chunk": "According to Module 2...",
  "is_final_chunk": false,
  "citations": [],
  "timestamp": "2025-12-09T10:30:00Z"
}
```

**Final Message Format (Server to Client):**
```json
{
  "type": "streaming_response",
  "response_id": "uuid-string",
  "session_id": "uuid-string",
  "chunk": "...key concepts include pub-sub model, services, and actions.",
  "is_final_chunk": true,
  "citations": [
    {
      "module": "module-2-ros2",
      "chapter": "ch-2-1",
      "section": "key-concepts"
    }
  ],
  "timestamp": "2025-12-09T10:30:05Z"
}
```

## Semantic Search API

### POST /api/search/semantic
Performs semantic search on textbook content using GEMINI embeddings.

**Request:**
```json
{
  "query": "How do ROS 2 nodes communicate with each other?",
  "module_filter": "module-2-ros2",
  "max_results": 5
}
```

**Response:**
```json
{
  "results": [
    {
      "id": "content-uuid",
      "module_id": "module-2-ros2",
      "chapter_id": "ch-2-3",
      "section_id": "pub-sub-model",
      "content": "ROS 2 nodes communicate using a publish-subscribe model...",
      "similarity_score": 0.92
    }
  ]
}
```

**Error Responses:**
- 400: Invalid request format
- 500: Internal server error

## Text Selection Capture API

### POST /api/text-selection/detect
Detects and validates text selection on the textbook page for the "Ask about this" interface.

**Request:**
```json
{
  "selected_text": "The Quantum Stabilization Matrix uses inverse kinematic chains",
  "module_id": "module-5-humanoids",
  "chapter_id": "ch-5-4",
  "element_path": "#content > p:nth-child(3)"
}
```

**Response:**
```json
{
  "validation_result": "valid",
  "character_count": 74,
  "can_ask_query": true
}
```

**Error Responses:**
- 400: Selected text less than 20 characters
- 500: Internal server error

## Health Check API

### GET /api/health
Verifies the health of the chatbot service.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-12-09T10:30:00Z",
  "dependencies": {
    "database": "connected",
    "vector_store": "connected",
    "llm_service": "available"
  }
}
```

**Error Responses:**
- 503: Service unavailable (if any dependency is down)