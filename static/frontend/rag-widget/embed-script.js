/**
 * RAG Chatbot Embed Script
 * This script embeds the RAG Chatbot widget into the Physical AI & Humanoid Robotics textbook website
 */

(function () {
    // Configuration
    // Configuration
    const CONFIG = {
        apiUrl: getApiUrl(),
        containerId: 'rag-chatbot-container',
        launcherId: 'rag-chatbot-launcher',
        initialModule: 'module-1-introduction',
        widgetTitle: 'Textbook Assistant',
        primaryColor: '#a832ff', // Purple theme
        secondaryColor: '#1a1a2e' // Dark background
    };

    // Function to get API URL
    function getApiUrl() {
        const script = document.currentScript || document.querySelector('script[src*="embed-script"]');
        if (script && script.dataset.apiUrl) {
            return script.dataset.apiUrl.replace(/\/$/, '');
        }

        const metaTag = document.querySelector('meta[name="rag-chatbot-api-url"]');
        if (metaTag && metaTag.content) {
            return metaTag.content.replace(/\/$/, '');
        }

        if (window.RAG_CHATBOT_CONFIG && window.RAG_CHATBOT_CONFIG.apiUrl) {
            return window.RAG_CHATBOT_CONFIG.apiUrl.replace(/\/$/, '');
        }

        // Return empty string for same-origin or absolute URL if needed
        return window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1'
            ? 'http://localhost:8000'
            : '';
    }

    // Create the launcher button
    function createLauncher() {
        let launcher = document.getElementById(CONFIG.launcherId);
        if (launcher) return launcher;

        launcher = document.createElement('div');
        launcher.id = CONFIG.launcherId;
        launcher.style.cssText = `
            position: fixed;
            bottom: 20px;
            right: 20px;
            width: 60px;
            height: 60px;
            background: ${CONFIG.primaryColor};
            border-radius: 50%;
            display: flex;
            justify-content: center;
            align-items: center;
            cursor: pointer;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
            z-index: 10001;
            transition: transform 0.3s ease;
        `;
        launcher.innerHTML = `
            <svg width="30" height="30" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path>
            </svg>
        `;

        launcher.addEventListener('click', toggleChat);
        launcher.addEventListener('mouseover', () => launcher.style.transform = 'scale(1.1)');
        launcher.addEventListener('mouseout', () => launcher.style.transform = 'scale(1)');

        document.body.appendChild(launcher);
        return launcher;
    }

    // Toggle chat visibility
    function toggleChat() {
        const container = document.getElementById(CONFIG.containerId);
        if (container.style.display === 'none') {
            container.style.display = 'flex';
        } else {
            container.style.display = 'none';
        }
    }

    // Create the chatbot container
    function createChatbotContainer() {
        let container = document.getElementById(CONFIG.containerId);
        if (container) return container;

        container = document.createElement('div');
        container.id = CONFIG.containerId;
        container.style.cssText = `
            position: fixed;
            bottom: 20px;
            right: 20px;
            width: 380px;
            height: min(500px, 60vh);
            max-height: calc(100vh - 40px);
            z-index: 10000;
            box-shadow: 0 8px 24px rgba(0,0,0,0.2);
            border-radius: 12px;
            overflow: hidden;
            display: none;
            flex-direction: column;
            background: white;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif;
            border: 1px solid #eee;
            transition: all 0.3s ease;
        `;

        const header = document.createElement('div');
        header.style.cssText = `
            background: ${CONFIG.primaryColor};
            color: white;
            padding: 15px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            cursor: move;
        `;
        header.innerHTML = `
            <span style="font-weight: 600;">${CONFIG.widgetTitle}</span>
            <button id="rag-close-btn" style="background: none; border: none; color: white; font-size: 20px; cursor: pointer;">&times;</button>
        `;

        const chatArea = document.createElement('div');
        chatArea.id = 'rag-chat-area';
        chatArea.style.cssText = `flex: 1; overflow-y: auto; padding: 15px; background: #fff;`;

        const inputArea = document.createElement('div');
        inputArea.style.cssText = `padding: 15px; border-top: 1px solid #eee; display: flex; background: #f8f9fa;`;
        inputArea.innerHTML = `
            <input type="text" id="rag-user-input" placeholder="Ask a question..." style="flex: 1; padding: 10px; border: 1px solid #ddd; border-radius: 6px; margin-right: 10px; outline: none;">
            <button id="rag-send-btn" style="background: ${CONFIG.primaryColor}; color: white; border: none; border-radius: 6px; padding: 10px 15px; cursor: pointer; font-weight: 600;">Send</button>
        `;

        container.appendChild(header);
        container.appendChild(chatArea);
        container.appendChild(inputArea);
        document.body.appendChild(container);

        document.getElementById('rag-close-btn').addEventListener('click', toggleChat);
        document.getElementById('rag-send-btn').addEventListener('click', sendMessage);
        document.getElementById('rag-user-input').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') sendMessage();
        });

        // Add CSS for mobile responsiveness
        const style = document.createElement('style');
        style.innerHTML = `
            @media (max-width: 480px) {
                #${CONFIG.containerId} {
                    width: calc(100% - 40px) !important;
                    right: 20px !important;
                    bottom: 80px !important;
                    height: calc(100vh - 120px) !important;
                }
            }
        `;
        document.head.appendChild(style);

        return container;
    }

    async function sendMessage() {
        const input = document.getElementById('rag-user-input');
        const message = input.value.trim();
        if (!message) return;

        addMessageToChat('user', message);
        input.value = '';

        try {
            const currentModule = getCurrentModuleContext();
            const response = await fetch(`${CONFIG.apiUrl}/api/chat/query`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    message: message,
                    module_context: currentModule,
                    session_id: window.ragChatbotSessionId || null
                })
            });

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({}));
                throw new Error(errorData.detail || `Server error: ${response.status}`);
            }

            const data = await response.json();
            if (data.session_id) window.ragChatbotSessionId = data.session_id;
            addMessageToChat('bot', data.message);
        } catch (error) {
            console.error('Error sending message:', error);
            let userError = 'Sorry, I encountered an error.';
            if (error.message.includes('HTTP error') || error.message.includes('Server error')) {
                userError += ` (Detail: ${error.message})`;
            } else if (error.name === 'TypeError') {
                userError += ' (Network error: Check your connection)';
            } else {
                userError += ` (${error.message})`;
            }
            addMessageToChat('bot', userError);
        }
    }

    function addMessageToChat(sender, message) {
        const chatArea = document.getElementById('rag-chat-area');
        const messageDiv = document.createElement('div');
        messageDiv.style.cssText = `margin-bottom: 12px; display: flex; ${sender === 'user' ? 'justify-content: flex-end;' : 'justify-content: flex-start;'}`;

        const bubble = document.createElement('div');
        bubble.style.cssText = `
            max-width: 85%;
            padding: 10px 14px;
            border-radius: 12px;
            font-size: 14px;
            line-height: 1.4;
            ${sender === 'user'
                ? `background: ${CONFIG.primaryColor}; color: white; border-bottom-right-radius: 2px;`
                : 'background: #f0f0f0; color: #1a1a2e; border-bottom-left-radius: 2px;'};
        `;
        bubble.textContent = message;
        messageDiv.appendChild(bubble);
        chatArea.appendChild(messageDiv);
        chatArea.scrollTop = chatArea.scrollHeight;
    }

    function getCurrentModuleContext() {
        const pathParts = window.location.pathname.split('/');
        return pathParts.find(part => part.startsWith('module')) || CONFIG.initialModule;
    }

    function init() {
        createLauncher();
        createChatbotContainer();
        setTimeout(() => {
            if (document.getElementById('rag-chat-area').children.length === 0) {
                addMessageToChat('bot', 'Hello! I\'m your Physical AI assistant. How can I help you today?');
            }
        }, 1000);
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }

})();