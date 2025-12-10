# Data Model: RAG Chatbot for Physical AI & Humanoid Robotics Textbook

## Core Entities

### Student
**Description:** Primary user of the chatbot system
**Fields:**
- `id` (UUID): Unique identifier for the student
- `created_at` (DateTime): Timestamp when student record was created
- `updated_at` (DateTime): Timestamp of last update
- `preferences` (JSON): Student preferences for the chatbot interaction
- `last_module_accessed` (String): Module ID of the last module accessed

### Textbook Content
**Description:** Structured educational material from the 6 textbook modules
**Fields:**
- `id` (UUID): Unique identifier for the content chunk
- `module_id` (String): Identifier for the textbook module
- `chapter_id` (String): Identifier for the textbook chapter
- `section_id` (String): Identifier for the textbook section
- `content_type` (Enum): Type of content (text, code, diagram_description, etc.)
- `content` (Text): The actual content text
- `hierarchy_path` (String): Full path in the textbook hierarchy
- `embedding` (Vector): Vector embedding of the content for semantic search (generated using GEMINI embeddings)
- `metadata` (JSON): Additional metadata (word count, reading level, etc.)

### Chat Session
**Description:** Represents a single conversation session between student and chatbot
**Fields:**
- `id` (UUID): Unique identifier for the session
- `student_id` (UUID): References the student participating
- `created_at` (DateTime): Session start time
- `updated_at` (DateTime): Last interaction time
- `current_module_context` (String): Module the student is currently viewing
- `is_active` (Boolean): Whether the session is currently active
- `session_metadata` (JSON): Additional session-specific data

### Message
**Description:** Individual message within a chat session
**Fields:**
- `id` (UUID): Unique identifier for the message
- `session_id` (UUID): References the chat session
- `sender_type` (Enum): Type of sender (student, ai_agent)
- `content` (Text): The message content
- `timestamp` (DateTime): When the message was sent
- `message_metadata` (JSON): Additional message-specific data
- `citations` (JSON): References to textbook modules/chapters used in response
- `selected_text_ref` (UUID): Reference to selected text if this is a text selection query

### Selected Text
**Description:** Portion of textbook content highlighted by the student
**Fields:**
- `id` (UUID): Unique identifier for the selected text
- `content` (Text): The actual selected text (minimum 20 characters)
- `module_id` (String): Module where text was selected
- `chapter_id` (String): Chapter where text was selected
- `section_id` (String): Section where text was selected
- `hierarchy_path` (String): Full path in textbook hierarchy where selection occurred
- `created_at` (DateTime): When the selection was made

### Vector Index Entry
**Description:** Entry in the vector database for semantic search
**Fields:**
- `id` (UUID): Unique identifier for the vector entry
- `text_content_id` (UUID): References the Textbook Content entity
- `text_chunk` (Text): The text that was embedded
- `embedding_vector` (Vector): The actual embedding vector (dimensions based on GEMINI embedding model)
- `metadata` (JSON): Metadata for filtering (module, chapter, etc.)

## Relationships

1. **Student** *(1)* → *(Many)* **Chat Session**: A student can have multiple chat sessions over time
2. **Chat Session** *(1)* → *(Many)* **Message**: A session contains multiple messages
3. **Message** *(Many)* → *(1)* **Selected Text** (Optional): Messages may reference selected text
4. **Textbook Content** *(1)* → *(Many)* **Vector Index Entry**: One content chunk can have multiple vector entries if chunked differently

## Validation Rules

1. **Student**: 
   - `id` must be a valid UUID
   - `created_at` must be before `updated_at` if both present

2. **Textbook Content**:
   - `module_id`, `chapter_id`, `section_id` must follow the textbook hierarchy
   - `content` must not be empty
   - `embedding` must be a valid vector of appropriate dimensions
   - `content_type` must be one of the allowed values

3. **Chat Session**:
   - `student_id` must reference an existing student
   - `is_active` can only be true for one session per student at a time
   - `current_module_context` must be a valid textbook module

4. **Message**:
   - `session_id` must reference an active session
   - `sender_type` must be either 'student' or 'ai_agent'
   - `timestamp` must be within the session timeframe

5. **Selected Text**:
   - `content` must be at least 20 characters
   - `module_id`, `chapter_id`, `section_id` must form a valid textbook path

## State Transitions (if applicable)

### Chat Session States
- `created` → `active` when first message is sent
- `active` → `inactive` after period of inactivity (30 minutes) or explicit session end
- `inactive` → `active` if student continues conversation within retention period (24 hours)
- `inactive` → `archived` after retention period expires

## Indexing Strategy

1. **Textbook Content**: 
   - Primary index on `id`
   - Composite index on `module_id`, `chapter_id`, `section_id` for hierarchical queries
   - Index on `content_type` for filtering by content type

2. **Chat Session**:
   - Primary index on `id`
   - Index on `student_id` for student-specific queries
   - Index on `is_active` for active session queries
   - Index on `current_module_context` for module-specific analytics

3. **Message**:
   - Primary index on `id`
   - Index on `session_id` for session-specific message retrieval
   - Index on `timestamp` for chronological ordering
   - Composite index on `sender_type` and `session_id` for conversation flow

4. **Vector Index Entry**:
   - Primary index on `id`
   - Index on `text_content_id` for content lookup
   - Vector index on `embedding_vector` for semantic search operations