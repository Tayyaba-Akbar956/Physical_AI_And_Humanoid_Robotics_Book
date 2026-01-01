# Physical AI & Humanoid Robotics Textbook with RAG Chatbot

An intelligent chatbot that helps students with the Physical AI & Humanoid Robotics textbook using Retrieval-Augmented Generation.

## Deployment

For detailed instructions on how to deploy this project (Frontend on Vercel, Backend on Render), please see the [Deployment Guide](DEPLOYMENT.md).

## Features

- **Conversational Context**: Maintains multi-turn conversations with context awareness
- **Module-Aware Context**: Prioritizes content from the current module being studied
- **Text Selection Queries**: Allows students to select text and ask questions about it
- **Semantic Search**: Uses GEMINI embeddings for accurate content retrieval
- **Citation Tracking**: Properly cites textbook modules, chapters, and sections
- **Comprehensive Logging**: Full monitoring and debugging capabilities

## Prerequisites

- Python 3.11+
- Node.js (for frontend development)

## Setup Instructions

### 1. Clone the Repository
```bash
git clone <repository-url>
cd <repository-name>
```

### 2. Backend Setup

1. Navigate to the backend directory:
```bash
cd backend
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables by creating a `.env` file:
```env
GEMINI_API_KEY=your_gemini_api_key
NEON_DB_URL=your_neon_postgres_url
QDRANT_URL=your_qdrant_cloud_url
QDRANT_API_KEY=your_qdrant_api_key
```

### 3. Frontend Integration

The frontend widget is located in `frontend/rag-widget/` and includes:
- `chat-widget.js` - Main chat widget implementation
- `text-selector.js` - Text selection functionality
- `styles.css` - Styling for the widget
- `embed-script.js` - Script to embed the widget in the textbook

## Running the Application

### 1. Start the Backend Service
```bash
cd backend
uvicorn src.main:app --reload --port 8000
```

### 2. Verify the Service
Visit http://localhost:8000 to see the API root endpoint.

## Testing the Implementation

### 1. Run the Comprehensive Test Suite
```bash
cd backend
python test_implementation.py
```

### 2. API Endpoints

The application provides the following API endpoints:

#### Chat Endpoints
- `POST /api/chat/query` - General chat queries
- `GET /api/chat/stream/{session_id}` - Response streaming
- `GET /api/chat/health` - Chat service health check

#### Session Management
- `POST /api/session/create` - Create new chat session
- `GET /api/session/{session_id}` - Get session details
- `POST /api/session/{session_id}/clear` - Clear conversation history

#### Text Selection
- `POST /api/text-selection/detect` - Detect text selection
- `POST /api/text-selection/query` - Query about selected text

#### Conversation Management
- `GET /api/conversation/history/{session_id}` - Get conversation history
- `GET /api/conversation/context/{session_id}` - Get conversation context
- `GET /api/conversation/summary/{session_id}` - Get conversation summary
- `POST /api/conversation/reset-context/{session_id}` - Reset context

#### Module Context
- `POST /api/module-context/set` - Set module context for session
- `GET /api/module-context/current/{session_id}` - Get current module
- `POST /api/module-context/relevance` - Get module relevance scores

#### System
- `GET /api/system/health` - System health check
- `GET /api/system/status` - System status
- `GET /api/system/metrics` - Performance metrics

## Using the Frontend Widget

To embed the chatbot in the textbook website:

1. Include the embed script in your HTML:
```html
<script src="/frontend/rag-widget/embed-script.js"></script>
```

2. The widget will automatically appear on the page with conversation history preservation.

## Configuration

### Environment Variables
Create a `.env` file with the following variables:

```env
# GEMINI API Configuration
GEMINI_API_KEY=your_gemini_api_key_here

# Qdrant Vector Database Configuration
QDRANT_URL=your_qdrant_url_here
QDRANT_API_KEY=your_qdrant_api_key_here

# Neon Database Configuration
NEON_DB_URL=your_neon_db_url_here

# Application Configuration
PORT=8000
DEBUG=false
ALLOWED_ORIGINS=http://localhost:3000,https://yourdomain.com
```

### Module Context
The system automatically detects the current module from the URL path. To manually set the module context, use the `/api/module-context/set` endpoint.

## Deployment

### Production Deployment
For production deployment, see the [DEPLOYMENT.md](DEPLOYMENT.md) guide for complete instructions.

### Frontend Integration
The frontend widget can be configured with different API endpoints:

1. **Via Script Data Attribute:**
   ```html
   <script src="/frontend/rag-widget/embed-script.js" data-api-url="https://yourdomain.com/api"></script>
   ```

2. **Via Meta Tag:**
   ```html
   <meta name="rag-chatbot-api-url" content="https://yourdomain.com/api">
   <script src="/frontend/rag-widget/embed-script.js"></script>
   ```

3. **Via Global Configuration:**
   ```html
   <script>
     window.RAG_CHATBOT_CONFIG = {
       apiUrl: 'https://yourdomain.com/api'
     };
   </script>
   <script src="/frontend/rag-widget/embed-script.js"></script>
   ```

## Architecture

The application is structured as follows:

```
backend/
├── src/
│   ├── api/              # API endpoints
│   ├── services/         # Business logic services
│   ├── models/           # Data models
│   ├── db/               # Database connections
│   └── middleware/       # Request processing middleware
├── requirements.txt      # Python dependencies
└── test_implementation.py  # Test suite

frontend/
└── rag-widget/           # Frontend widget components
```

## Development

### Running Tests
The test suite can be run with:
```bash
python test_implementation.py
```

### Adding New Content
To add new textbook content to the knowledge base:
1. Use the content processing pipeline in `src/content_pipeline.py`
2. Generate embeddings using the embedding generator
3. Upload to the Qdrant vector database

## Troubleshooting

If you encounter issues:
1. Verify all environment variables are set correctly
2. Check that external services (GEMINI, Qdrant, Neon) are accessible
3. Review logs for error messages
4. Run the test suite to verify functionality

## Performance Monitoring

The system includes:
- Response time monitoring
- API request metrics
- Performance optimization tools
- System health checks

Access metrics at `/api/system/metrics` endpoint.