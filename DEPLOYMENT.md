# Deployment Guide for Physical AI & Humanoid Robotics Textbook with RAG Chatbot

## Overview
This guide provides instructions for deploying the RAG Chatbot application for the Physical AI & Humanoid Robotics Textbook.

## Prerequisites

### System Requirements
- Python 3.11+
- Node.js 18+ (for frontend)
- Git
- Access to external services (GEMINI, Qdrant, NeonDB)

### External Services Required
1. **GEMINI API Key**: For AI model access
2. **Qdrant Vector Database**: For semantic search
3. **NeonDB (PostgreSQL)**: For structured data storage

## Environment Configuration

### 1. Create Environment File
Copy the example environment file and fill in your credentials:

```bash
cp .env.example .env
```

### 2. Update Environment Variables
Edit the `.env` file with your specific configuration:

```env
# GEMINI API Configuration
GEMINI_API_KEY=your_actual_gemini_api_key_here

# Qdrant Vector Database Configuration
QDRANT_URL=your_actual_qdrant_url_here
QDRANT_API_KEY=your_actual_qdrant_api_key_here

# Neon Database Configuration
NEON_DB_URL=your_actual_neon_db_url_here

# Application Configuration
PORT=8000
DEBUG=false
ALLOWED_ORIGINS=https://yourdomain.com,https://www.yourdomain.com
```

## Backend Deployment

### 1. Install Dependencies
```bash
cd backend
pip install -r requirements.txt
```

### 2. Run the Application
```bash
# For development
uvicorn src.main:app --reload --port 8000

# For production (using gunicorn)
gunicorn src.main:app --workers 4 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

## Frontend Integration

### 1. Configure API URL
The frontend widget can be configured in multiple ways:

#### Option A: Via Script Data Attribute
```html
<script src="/frontend/rag-widget/embed-script.js" data-api-url="https://yourdomain.com/api"></script>
```

#### Option B: Via Meta Tag
```html
<meta name="rag-chatbot-api-url" content="https://yourdomain.com/api">
<script src="/frontend/rag-widget/embed-script.js"></script>
```

#### Option C: Via Global Configuration
```html
<script>
  window.RAG_CHATBOT_CONFIG = {
    apiUrl: 'https://yourdomain.com/api'
  };
</script>
<script src="/frontend/rag-widget/embed-script.js"></script>
```

### 2. Embed the Widget
Include the following script in your HTML to embed the chatbot:

```html
<script src="/frontend/rag-widget/embed-script.js"></script>
```

## Production Configuration

### 1. Security Considerations
- Update CORS settings in `.env` with your actual domains
- Set `DEBUG=false` in production
- Use HTTPS for all API calls
- Implement proper API rate limiting

### 2. Environment Variables for Production
```env
# Production environment
DEBUG=false
ALLOWED_ORIGINS=https://yourdomain.com,https://www.yourdomain.com
PORT=8000

# API keys (do not commit to version control)
GEMINI_API_KEY=your_production_gemini_api_key
QDRANT_URL=your_production_qdrant_url
QDRANT_API_KEY=your_production_qdrant_api_key
NEON_DB_URL=your_production_neon_db_url
```

## Testing the Deployment

### 1. Backend Health Check
Verify the backend is running:
```bash
curl http://localhost:8000/health
```

### 2. API Endpoints
Test the main API endpoint:
```bash
curl http://localhost:8000/
```

### 3. Frontend Widget
Visit your website and verify the chatbot widget appears and functions correctly.

## Monitoring and Logging

### 1. Application Logs
The application logs to standard output. Configure your deployment platform to capture these logs.

### 2. Health Checks
The application provides health check endpoints:
- `/health` - Basic health check
- `/api/system/health` - System health check
- `/api/system/metrics` - Performance metrics

## Troubleshooting

### Common Issues

1. **Environment Variables Not Set**
   - Error: "GEMINI_API_KEY is not set in environment variables"
   - Solution: Ensure `.env` file is properly configured and loaded

2. **CORS Errors**
   - Error: "Access to fetch at ... from origin ... has been blocked by CORS policy"
   - Solution: Update `ALLOWED_ORIGINS` in your environment variables

3. **API Connection Issues**
   - Error: Cannot connect to external services
   - Solution: Verify API keys and service URLs are correct

### Debugging Tips
- Set `DEBUG=true` temporarily to get more detailed error messages
- Check that all required environment variables are properly set
- Verify external services (GEMINI, Qdrant, NeonDB) are accessible

## Scaling Recommendations

1. **Backend Scaling**
   - Use multiple workers when running with gunicorn
   - Implement load balancing for high-traffic scenarios
   - Monitor resource usage and scale accordingly

2. **Database Scaling**
   - Monitor Qdrant performance for semantic search
   - Consider NeonDB scaling options for high usage

3. **Frontend Optimization**
   - Cache the embed script for better performance
   - Implement lazy loading for the chatbot widget