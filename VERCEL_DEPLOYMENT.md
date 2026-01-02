# Vercel Deployment Guide: Separate Frontend & Backend

This guide explains how to deploy the Physical AI & Humanoid Robotics Book with a clear separation between the Docusaurus frontend and the FastAPI backend on Vercel.

---

## 1. Backend Deployment (Serverless API)

The backend is located in the `/backend` folder and is configured to run as a Vercel Serverless Function.

### Steps
1. In the Vercel Dashboard, click **Add New** > **Project**.
2. Select your repository.
3. **IMPORTANT**: Change the **Root Directory** to `backend`.
4. Vercel will detect the Python environment.
5. In **Environment Variables**, add the following:
   - `GEMINI_API_KEY`: Your Google Gemini API key.
   - `QDRANT_URL`: URL of your Qdrant instance.
   - `QDRANT_API_KEY`: API key for Qdrant.
   - `NEON_DB_URL`: PostgreSQL connection string (Neon DB).
   - `ALLOWED_ORIGINS`: `https://your-frontend-domain.vercel.app` (The frontend URL).
6. Click **Deploy**.

**Note**: The backend will be accessible at `https://your-backend-service.vercel.app/`.

---

## 2. Frontend Deployment (Static Site)

The frontend is the root Docusaurus project.

### Steps
1. In the Vercel Dashboard, click **Add New** > **Project**.
2. Select the same repository.
3. Keep the **Root Directory** as the root (`./`).
4. Vercel will auto-detect Docusaurus settings:
   - **Build Command**: `docusaurus build`
   - **Output Directory**: `build`
5. In **Environment Variables**, add:
   - `REACT_APP_API_URL`: `https://your-backend-service.vercel.app` (The URL from Step 1).
6. Click **Deploy**.

---

## 3. Connecting Them Together

1. **API URL Detection**: The frontend is configured to detect the backend URL in this order:
   - `REACT_APP_API_URL` environment variable (build-time).
   - `<meta name="rag-chatbot-api-url" content="...">` tag in HTML (added automatically by Docusaurus plugin).
2. **CORS**: Ensure the backend's `ALLOWED_ORIGINS` includes your frontend's production URL.

---

## Known Limitations on Vercel

### 1. Request Timeouts (Hobby Plan)
Vercel Hobby plan has a **10-second** execution limit for serverless functions. RAG queries (vector search + LLM generation) can sometimes exceed this.
- **Solution**: We've implemented an internal timeout in the backend and a loading state/retry logic in the frontend.

### 2. Cold Starts
If the backend hasn't been used for a while, the first request may take 5-10 seconds to start up.
- **Solution**: The UI shows a "Cold start" warning during the first interaction.

### 3. Persistent Connections
Serverless functions do not maintain persistent database connections.
- **Solution**: We use SQLAlchemy connection pooling with `pool_pre_ping` to handle ephemeral connections effectively.

## Troubleshooting

- **403 CORS Error**: Double check the `ALLOWED_ORIGINS` in your backend deployment.
- **504 Gateway Timeout**: The query took longer than Vercel's limit. Try asking a simpler question.
- **404 Not Found**: Ensure you are calling `/api/...` and that the backend's `vercel.json` is correctly routing to `index.py`.
