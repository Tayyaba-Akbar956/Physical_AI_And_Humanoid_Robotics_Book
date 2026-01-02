# Physical AI & Humanoid Robotics Book - Vercel Deployment Guide

This project is configured for separate deployment of frontend and backend on Vercel.

## Deployment Strategy

- **Frontend**: Docusaurus site deployed as a static site on Vercel
- **Backend**: FastAPI RAG chatbot deployed as a serverless function on Vercel

## Deployment Steps

### 1. Backend Deployment (Vercel Serverless)

1. Go to [Vercel Dashboard](https://vercel.com/dashboard)
2. Click **New Project** and import your repository
3. In the project configuration:
   - **Root Directory**: Set to `backend`
   - **Framework Preset**: Auto (Vercel will detect Python)
4. Add Environment Variables:
   ```
   GEMINI_API_KEY=your_google_gemini_api_key
   QDRANT_URL=your_qdrant_instance_url
   QDRANT_API_KEY=your_qdrant_api_key
   NEON_DB_URL=your_neon_db_connection_string
   ALLOWED_ORIGINS=https://your-frontend-domain.vercel.app
   ```
5. Deploy the project

### 2. Frontend Deployment (Vercel Static)

1. Go to [Vercel Dashboard](https://vercel.com/dashboard)
2. Click **New Project** and import the same repository
3. In the project configuration:
   - **Root Directory**: Keep as root (`./`)
   - **Build Command**: `npm run build` (should auto-detect)
   - **Output Directory**: `build`
4. Add Environment Variables:
   ```
   REACT_APP_API_URL=https://your-backend-domain.vercel.app
   ```
5. Deploy the project

## API Endpoints

After deployment, your backend will be available at:
- Root: `https://your-backend-domain.vercel.app/`
- Health check: `https://your-backend-domain.vercel.app/api/health`
- Chat endpoint: `https://your-backend-domain.vercel.app/api/chat/query`

## Configuration Notes

- The frontend uses multiple methods to detect the backend API URL:
  1. `REACT_APP_API_URL` environment variable (build-time)
  2. `<meta name="rag-chatbot-api-url" content="...">` tag
  3. Runtime detection

- The chatbot widget is embedded automatically in all pages via the Docusaurus plugin

## Troubleshooting

- **CORS Errors**: Verify that `ALLOWED_ORIGINS` in your backend includes your frontend URL
- **API Not Found**: Ensure the backend is deployed and the URL is correctly configured in the frontend
- **Chatbot Not Loading**: Check browser console for errors and verify API URL configuration

## Vercel Configuration Files

- `backend/vercel.json`: Configures the Python serverless backend
- Root `vercel.json`: Configures the Docusaurus static frontend