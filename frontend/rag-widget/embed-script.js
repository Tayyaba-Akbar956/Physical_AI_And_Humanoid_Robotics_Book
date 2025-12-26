/**
 * RAG Chatbot Embed Script
 * This script embeds the RAG Chatbot widget into the Physical AI & Humanoid Robotics textbook website
 */

(function() {
    // Configuration
    const CONFIG = {
        apiUrl: getApiUrl(), // Get API URL from data attribute or fallback to default
        containerId: 'rag-chatbot-container',
        initialModule: 'module-1-introduction', // Will be replaced based on current page
        widgetTitle: 'Textbook Assistant',
        primaryColor: '#3498db',
        secondaryColor: '#2c3e50'
    };

    // Function to get API URL from data attribute or environment
    function getApiUrl() {
        // Try to get from script data attribute first
        const script = document.currentScript || document.querySelector('script[src*="embed-script"]');
        if (script && script.dataset.apiUrl) {
            return script.dataset.apiUrl;
        }

        // Try to get from meta tag
        const metaTag = document.querySelector('meta[name="rag-chatbot-api-url"]');
        if (metaTag && metaTag.content) {
            return metaTag.content;
        }

        // Try to get from window object (for configuration via script)
        if (window.RAG_CHATBOT_CONFIG && window.RAG_CHATBOT_CONFIG.apiUrl) {
            return window.RAG_CHATBOT_CONFIG.apiUrl;
        }

        // Fallback to environment-specific URL or localhost
        return window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1'
            ? 'http://localhost:8000'
            : 'https://yourdomain.com/api'; // Replace with your production API URL
    }

    // Create the chatbot container
    function createChatbotContainer() {
        // Check if container already exists
        let container = document.getElementById(CONFIG.containerId);
        if (container) return container;

        // Create main container
        container = document.createElement('div');
        container.id = CONFIG.containerId;
        container.style.cssText = `
            position: fixed;
            bottom: 20px;
            right: 20px;
            width: 380px;
            height: 600px;
            z-index: 10000;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            border-radius: 12px;
            overflow: hidden;
            display: flex;
            flex-direction: column;
            background: white;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif;
        `;

        // Create header
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
            <button id="rag-close-btn" style="
                background: none;
                border: none;
                color: white;
                font-size: 18px;
                cursor: pointer;
            ">&times;</button>
        `;

        // Create chat area
        const chatArea = document.createElement('div');
        chatArea.id = 'rag-chat-area';
        chatArea.style.cssText = `
            flex: 1;
            overflow-y: auto;
            padding: 15px;
            background: #f9f9f9;
        `;

        // Create input area
        const inputArea = document.createElement('div');
        inputArea.style.cssText = `
            padding: 15px;
            border-top: 1px solid #eee;
            display: flex;
        `;
        inputArea.innerHTML = `
            <input 
                type="text" 
                id="rag-user-input" 
                placeholder="Ask a question about this textbook..."
                style="flex: 1; padding: 10px; border: 1px solid #ddd; border-radius: 4px; margin-right: 10px;"
            >
            <button id="rag-send-btn" style="
                background: ${CONFIG.primaryColor};
                color: white;
                border: none;
                border-radius: 4px;
                padding: 10px 15px;
                cursor: pointer;
            ">Send</button>
        `;

        // Assemble the widget
        container.appendChild(header);
        container.appendChild(chatArea);
        container.appendChild(inputArea);

        // Add to page
        document.body.appendChild(container);

        // Add event listeners
        document.getElementById('rag-close-btn').addEventListener('click', () => {
            container.style.display = 'none';
        });

        document.getElementById('rag-send-btn').addEventListener('click', sendMessage);
        document.getElementById('rag-user-input').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') sendMessage();
        });

        return container;
    }

    // Send message to backend
    async function sendMessage() {
        const input = document.getElementById('rag-user-input');
        const message = input.value.trim();
        if (!message) return;

        const chatArea = document.getElementById('rag-chat-area');
        
        // Add user message to chat
        addMessageToChat('user', message);
        input.value = '';

        try {
            // Get current module context from page
            const currentModule = getCurrentModuleContext();
            
            // Send to backend
            const response = await fetch(`${CONFIG.apiUrl}/api/chat/query`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    message: message,
                    module_context: currentModule,
                    session_id: window.ragChatbotSessionId || null
                })
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            
            // Store session ID for future requests
            if (data.session_id && !window.ragChatbotSessionId) {
                window.ragChatbotSessionId = data.session_id;
            }

            // Add bot response to chat
            addMessageToChat('bot', data.message);
        } catch (error) {
            console.error('Error sending message:', error);
            addMessageToChat('bot', 'Sorry, I encountered an error processing your request. Please try again.');
        }
    }

    // Add message to chat area
    function addMessageToChat(sender, message) {
        const chatArea = document.getElementById('rag-chat-area');
        const messageDiv = document.createElement('div');
        messageDiv.style.cssText = `
            margin-bottom: 15px;
            display: flex;
            ${sender === 'user' ? 'justify-content: flex-end;' : 'justify-content: flex-start;'}
        `;

        const bubble = document.createElement('div');
        bubble.style.cssText = `
            max-width: 80%;
            padding: 10px 15px;
            border-radius: 18px;
            ${sender === 'user' 
                ? `background: ${CONFIG.primaryColor}; color: white;` 
                : 'background: #e9ecef; color: #333;'};
        `;
        bubble.textContent = message;

        messageDiv.appendChild(bubble);
        chatArea.appendChild(messageDiv);
        chatArea.scrollTop = chatArea.scrollHeight;
    }

    // Get current module context from the page
    function getCurrentModuleContext() {
        // Try to extract module from URL or page metadata
        const pathParts = window.location.pathname.split('/');
        if (pathParts.length > 1) {
            const modulePart = pathParts.find(part => part.startsWith('module'));
            if (modulePart) {
                return modulePart;
            }
        }
        
        // Fallback to default module
        return CONFIG.initialModule;
    }

    // Initialize drag functionality
    function initDrag() {
        const container = document.getElementById(CONFIG.containerId);
        const header = container.querySelector('div[style*="background:"]');
        
        let pos1 = 0, pos2 = 0, pos3 = 0, pos4 = 0;
        
        header.onmousedown = dragMouseDown;

        function dragMouseDown(e) {
            e = e || window.event;
            e.preventDefault();
            pos3 = e.clientX;
            pos4 = e.clientY;
            document.onmouseup = closeDragElement;
            document.onmousemove = elementDrag;
        }

        function elementDrag(e) {
            e = e || window.event;
            e.preventDefault();
            pos1 = pos3 - e.clientX;
            pos2 = pos4 - e.clientY;
            pos3 = e.clientX;
            pos4 = e.clientY;
            
            const top = (container.offsetTop - pos2) + 'px';
            const left = (container.offsetLeft - pos1) + 'px';
            
            // Keep within viewport
            const maxTop = window.innerHeight - container.offsetHeight;
            const maxLeft = window.innerWidth - container.offsetWidth;
            
            container.style.top = Math.max(0, Math.min(parseInt(top), maxTop)) + 'px';
            container.style.left = Math.max(0, Math.min(parseInt(left), maxLeft)) + 'px';
        }

        function closeDragElement() {
            document.onmouseup = null;
            document.onmousemove = null;
        }
    }

    // Initialize the chatbot when DOM is loaded
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', () => {
            createChatbotContainer();
            initDrag();
            // Add initial bot message
            setTimeout(() => {
                addMessageToChat('bot', 'Hello! I\'m your textbook assistant. Ask me any questions about the content you\'re studying.');
            }, 500);
        });
    } else {
        createChatbotContainer();
        initDrag();
        // Add initial bot message
        setTimeout(() => {
            addMessageToChat('bot', 'Hello! I\'m your textbook assistant. Ask me any questions about the content you\'re studying.');
        }, 500);
    }

})();