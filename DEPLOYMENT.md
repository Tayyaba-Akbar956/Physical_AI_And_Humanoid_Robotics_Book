# Deployment Guide: Split Hosting (Vercel + Render)

This project uses a split deployment strategy:
- **Frontend**: Docusaurus site on **Vercel**.
- **Backend**: FastAPI RAG Chatbot on **Render**.

---

## 1. Backend Deployment (Render)

### Prerequisites
- Create a [Render](https://render.com/) account.
- Connect your GitHub repository.

### Steps
1. Click **New +** and select **Web Service**.
2. Connect your repository.
3. Configure the service:
   - **Name**: `physical-ai-backend` (or your choice)
   - **Root Directory**: `backend`
   - **Runtime**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn api:app --host 0.0.0.0 --port $PORT`
4. Add **Environment Variables**:
   - `GEMINI_API_KEY`: Your Google Gemini API key.
   - `QDRANT_URL`: URL of your Qdrant instance.
   - `QDRANT_API_KEY`: API key for Qdrant.
   - `NEON_DB_URL`: PostgreSQL connection string (Neon DB).
   - `ALLOWED_ORIGINS`: `https://your-vercel-domain.vercel.app`
   - `PORT`: 8000 (Render sets this automatically).

---

## 2. Frontend Deployment (Vercel)

### Prerequisites
- Create a [Vercel](https://vercel.com/) account.
- Connect your GitHub repository.

### Steps
1. Click **Add New** and select **Project**.
2. Select your repository.
3. Vercel will auto-detect Docusaurus. Ensure the settings are:
   - **Build Command**: `npm run build` (or `docusaurus build`)
   - **Output Directory**: `build`
4. Add **Environment Variables**:
   - `REACT_APP_API_URL`: `https://your-render-service.onrender.com`

---

## 3. Post-Deployment Configuration

1. **Update Website Configuration**:
   - Ensure `docusaurus.config.js` or your meta tags reference the correct Render backend URL.
   
2. **CORS Validation**:
   - Verify that `ALLOWED_ORIGINS` in your Render backend includes your final Vercel domain.

3. **Verification**:
   - Visit your Vercel URL.
   - Open the chatbot and ask a question to verify it can reach the Render backend.

---

## Environment Variables Summary

### Backend (Render)
| Variable | Description |
| --- | --- |
| `GEMINI_API_KEY` | API key for RAG generation |
| `QDRANT_URL` | Vector DB URL |
| `QDRANT_API_KEY` | Vector DB Key |
| `NEON_DB_URL` | PostgreSQL URL |
| `ALLOWED_ORIGINS` | CORS whitelist (Vercel URL) |

### Frontend (Vercel)
| Variable | Description |
| --- | --- |
| `REACT_APP_API_URL` | The URL of your Render backend |