# Quickstart Guide: RAG Chatbot for Physical AI & Humanoid Robotics Textbook

## Prerequisites

- Python 3.11+
- Node.js 18+ (for development and testing)
- Access to GEMINI API (for both embeddings and text generation)
- Neon Serverless Postgres account
- Qdrant Cloud account

## Setup Instructions

### 1. Clone and Initialize the Repository

```bash
git clone <repository-url>
cd <repository-name>
```

### 2. Backend Setup
1. Navigate to the backend directory:
```bash
cd backend-node
```

2. Install dependencies:
```bash
npm install
```

3. Set up environment variables by creating a `.env` file:
```env
GEMINI_API_KEY=your_gemini_api_key
DATABASE_URL=postgresql://user:password@host:port/database
QDRANT_URL=your_qdrant_cloud_url
QDRANT_API_KEY=your_qdrant_api_key
```

4. Initialize the database:
```bash
npm run prisma:push
```

### 3. Frontend Setup

The frontend widget will be integrated with the existing Docusaurus-based textbook website. The JavaScript embed script needs to be added to the textbook pages.

### 4. Content Processing Pipeline

1. Run the content extraction and embedding pipeline:
```bash
python -m src.scripts.content_pipeline
```

This will:
- Scrape content from the existing textbook website
- Parse HTML content, preserving code blocks and image descriptions
- Chunk content appropriately (300-500 tokens per chunk)
- Generate embeddings using OpenAI text-embedding-3-large
- Upload embeddings to Qdrant with proper metadata

### 5. Run the Backend Service
1. Start the development server:
```bash
npm run dev
```

2. Verify the service is running:
```bash
http://localhost:3001/api/health
```

### 6. Embed the Chat Widget

Add the following script to the textbook pages to embed the chat widget:

```html
<script src="/path/to/chat-widget.js"></script>
<div id="rag-chatbot-container"></div>
<script>
  // Initialize the chatbot widget
  initializeRagChatbot({
    apiUrl: "http://localhost:3001",  // Update with your backend URL
    containerId: "rag-chatbot-container",
    initialModule: "module-1-introduction"  // Set based on current page
  });
</script>
```

## API Endpoints

### Chat Endpoints
- `POST /api/chat/query` - General chat queries
- `POST /api/chat/text-selection-query` - Queries based on selected text

### Session Management
- `POST /api/session/create` - Create new chat session
- `GET /api/session/{session_id}` - Get session details
- `POST /api/session/{session_id}/clear` - Clear conversation history

### Other Endpoints
- `GET /api/health` - Health check
- `POST /api/search/semantic` - Semantic search

## Testing
1. Run all tests:
```bash
npm test
```

## Development Workflow

1. Make changes to the backend in `backend-node/src/`
2. Update API contracts in `specs/1-rag-chatbot/contracts/` if needed
3. Update data models in `specs/1-rag-chatbot/data-model.md` if needed
4. Test changes using the test suite
5. For frontend changes, update files in `frontend/rag-widget/`

## Common Issues and Troubleshooting

### Embedding Generation Fails
- Verify OpenAI API key is correctly set
- Check internet connectivity
- Ensure rate limits are not exceeded

### Vector Search Not Returning Results
- Confirm content pipeline completed successfully
- Verify Qdrant connection and API key
- Check that textbook content was properly embedded

### Frontend Widget Not Appearing
- Verify script is properly included on textbook pages
- Check console for JavaScript errors
- Ensure backend API URL is correctly configured

## Next Steps

After completing the setup:

1. Customize the chatbot's instructions in the RAG agent service for specific textbook content
2. Fine-tune the UI/UX based on user feedback
3. Set up monitoring and analytics for usage patterns
4. Prepare for production deployment