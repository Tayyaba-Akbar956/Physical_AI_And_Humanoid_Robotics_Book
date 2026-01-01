export const getApiUrl = () => {
    // Check build-time environment variable
    if (typeof process !== 'undefined' && process.env.REACT_APP_API_URL) {
        return process.env.REACT_APP_API_URL;
    }

    // Check runtime meta tag
    if (typeof document !== 'undefined') {
        const metaTag = document.querySelector('meta[name="rag-chatbot-api-url"]');
        if (metaTag?.content) {
            return metaTag.content;
        }

        // Check script data attribute
        const script = document.querySelector('script[data-api-url]');
        if (script?.dataset?.apiUrl) {
            return script.dataset.apiUrl;
        }
    }

    // Check global config
    if (typeof window !== 'undefined' && window.RAG_CHATBOT_CONFIG?.apiUrl) {
        return window.RAG_CHATBOT_CONFIG.apiUrl;
    }

    // Default fallback
    return 'http://localhost:8000';
};

export const API_URL = getApiUrl();
