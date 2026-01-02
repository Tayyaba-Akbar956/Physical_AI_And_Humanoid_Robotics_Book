# Deployment Guide: Split Hosting (Vercel + Vercel)

This project uses a split deployment strategy:
- **Frontend**: Docusaurus site on **Vercel**.
- **Backend**: FastAPI RAG Chatbot on **Vercel**.

---

## 1. Backend Deployment (Vercel Serverless Function)

### Prerequisites
- Create a [Vercel](https://vercel.com/) account.
- Connect your GitHub repository.

### Steps
1. Click **New +** and select **Project**.
2. Connect your repository.
3. Configure the project:
   - **Root Directory**: `backend`
   - **Framework Preset**: Auto (Vercel will detect Python)
4. Vercel will auto-detect settings:
   - **Build Command**: `pip install -r requirements.txt`
   - **Output Directory**: (leave empty for Python projects)
5. Add **Environment Variables**:
   - `GEMINI_API_KEY`: Your Google Gemini API key.
   - `QDRANT_URL`: URL of your Qdrant instance.
   - `QDRANT_API_KEY`: API key for Qdrant.
   - `NEON_DB_URL`: PostgreSQL connection string (Neon DB).
   - `ALLOWED_ORIGINS`: `https://your-frontend-domain.vercel.app`
6. Click **Deploy**.

---

## 2. Frontend Deployment (Vercel Static Site)

### Prerequisites
- Create a [Vercel](https://vercel.com/) account.
- Connect your GitHub repository.

### Steps
1. Click **New +** and select **Project**.
2. Connect the same repository.
3. Configure the project:
   - **Root Directory**: `.` (root of repository)
   - **Framework Preset**: Auto (Vercel will detect Docusaurus)
4. Vercel will auto-detect settings:
   - **Build Command**: `npm run build` (or `docusaurus build`)
   - **Output Directory**: `build`
5. Add **Environment Variables**:
   - `REACT_APP_API_URL`: `https://your-backend-domain.vercel.app`
6. Click **Deploy**.

---

## 3. Post-Deployment Configuration

1. **Update Website Configuration**:
   - The `docusaurus.config.js` will automatically use the `REACT_APP_API_URL` environment variable.

2. **CORS Validation**:
   - Verify that `ALLOWED_ORIGINS` in your backend deployment includes your frontend domain.

3. **Verification**:
   - Visit your frontend Vercel URL.
   - Open the chatbot and ask a question to verify it can reach the backend.

---

## Environment Variables Summary

### Backend (Vercel)
| Variable | Description |
| --- | --- |
| `GEMINI_API_KEY` | API key for RAG generation |
| `QDRANT_URL` | Vector DB URL |
| `QDRANT_API_KEY` | Vector DB Key |
| `NEON_DB_URL` | PostgreSQL URL |
| `ALLOWED_ORIGINS` | CORS whitelist (frontend Vercel URL) |

### Frontend (Vercel)
| Variable | Description |
| --- | --- |
| `REACT_APP_API_URL` | The URL of your backend Vercel deployment |

---

## API Endpoints

After deployment, your backend will be available at:
- Root: `https://your-backend-domain.vercel.app/`
- Health check: `https://your-backend-domain.vercel.app/api/health`
- Chat endpoint: `https://your-backend-domain.vercel.app/api/chat/query`

## Troubleshooting

- **CORS Errors**: Verify that `ALLOWED_ORIGINS` in your backend includes your frontend URL
- **API Not Found**: Ensure the backend is deployed and the URL is correctly configured in the frontend
- **Chatbot Not Loading**: Check browser console for errors and verify API URL configuration