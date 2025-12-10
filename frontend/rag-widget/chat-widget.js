/**
 * RAG Chatbot Widget
 * Handles conversation history, message display, and UI interactions
 */

class RAGChatWidget {
    constructor(containerId, options = {}) {
        this.containerId = containerId;
        this.options = {
            apiUrl: options.apiUrl || 'http://localhost:8000',
            initialModule: options.initialModule || 'module-1-introduction',
            ...options
        };
        
        this.sessionId = null;
        this.currentModule = this.options.initialModule;
        this.messages = [];
        this.conversationHistory = [];
        
        this.init();
    }
    
    init() {
        this.createWidgetStructure();
        this.setupEventListeners();
        this.loadConversationHistory();
    }
    
    createWidgetStructure() {
        const container = document.getElementById(this.containerId);
        if (!container) {
            console.error(`Container with ID ${this.containerId} not found`);
            return;
        }
        
        container.innerHTML = `
            <div id="chat-header" class="rag-chat-header">
                <h3>Textbook Assistant</h3>
                <div class="rag-module-context">Module: <span id="current-module">${this.currentModule}</span></div>
                <button id="clear-history-btn" class="rag-clear-btn">Clear History</button>
            </div>
            <div id="chat-messages" class="rag-chat-messages">
                <!-- Messages will be added here -->
            </div>
            <div id="chat-input-area" class="rag-chat-input">
                <input type="text" id="user-message" class="rag-user-input" placeholder="Ask a question about this textbook..." />
                <button id="send-message" class="rag-send-btn">Send</button>
            </div>
        `;
    }
    
    setupEventListeners() {
        document.getElementById('send-message').addEventListener('click', () => this.sendMessage());
        document.getElementById('user-message').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') this.sendMessage();
        });
        
        document.getElementById('clear-history-btn').addEventListener('click', () => this.clearConversationHistory());
    }
    
    async sendMessage() {
        const inputElement = document.getElementById('user-message');
        const message = inputElement.value.trim();
        
        if (!message) return;
        
        // Add user message to UI
        this.addMessageToUI('user', message);
        inputElement.value = '';
        
        try {
            // Show typing indicator
            this.showTypingIndicator();
            
            // Send to backend
            const response = await fetch(`${this.options.apiUrl}/api/chat/query`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    message: message,
                    module_context: this.currentModule,
                    session_id: this.sessionId || null
                })
            });
            
            const data = await response.json();
            
            // Store session ID for future requests
            if (data.session_id && !this.sessionId) {
                this.sessionId = data.session_id;
            }
            
            // Hide typing indicator
            this.hideTypingIndicator();
            
            // Add bot response to UI
            this.addMessageToUI('bot', data.response, data.citations);
            
            // Update conversation history
            this.conversationHistory.push({
                sender: 'user',
                content: message,
                timestamp: new Date().toISOString()
            });
            this.conversationHistory.push({
                sender: 'bot',
                content: data.response,
                citations: data.citations,
                timestamp: new Date().toISOString()
            });
            
        } catch (error) {
            console.error('Error sending message:', error);
            this.hideTypingIndicator();
            this.addMessageToUI('bot', 'Sorry, I encountered an error processing your request. Please try again.');
        }
    }
    
    addMessageToUI(sender, content, citations = []) {
        const messagesContainer = document.getElementById('chat-messages');
        const messageDiv = document.createElement('div');
        messageDiv.className = `rag-message rag-message-${sender}`;
        
        // Format citations if present
        let contentWithCitations = content;
        if (citations.length > 0) {
            contentWithCitations += '<div class="rag-citations">';
            contentWithCitations += '<strong>Sources:</strong><ul>';
            citations.forEach(citation => {
                contentWithCitations += `<li>Module: ${citation.module}, Chapter: ${citation.chapter}, Section: ${citation.section}</li>`;
            });
            contentWithCitations += '</ul></div>';
        }
        
        messageDiv.innerHTML = `
            <div class="rag-message-content">${contentWithCitations}</div>
            <div class="rag-message-timestamp">${new Date().toLocaleTimeString()}</div>
        `;
        
        messagesContainer.appendChild(messageDiv);
        
        // Scroll to bottom
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }
    
    showTypingIndicator() {
        const messagesContainer = document.getElementById('chat-messages');
        const typingDiv = document.createElement('div');
        typingDiv.id = 'typing-indicator';
        typingDiv.className = 'rag-typing-indicator';
        typingDiv.innerHTML = 'Assistant is typing...';
        messagesContainer.appendChild(typingDiv);
        
        // Scroll to bottom
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }
    
    hideTypingIndicator() {
        const typingIndicator = document.getElementById('typing-indicator');
        if (typingIndicator) {
            typingIndicator.remove();
        }
    }
    
    async loadConversationHistory() {
        // If we have a session, try to load its history
        if (this.sessionId) {
            try {
                const response = await fetch(`${this.options.apiUrl}/api/conversation/history/${this.sessionId}`);
                const data = await response.json();
                
                if (data.messages && data.messages.length > 0) {
                    this.conversationHistory = [...data.messages];
                    
                    // Clear current messages and load history
                    document.getElementById('chat-messages').innerHTML = '';
                    
                    this.conversationHistory.forEach(message => {
                        this.addMessageToUI(
                            message.sender_type === 'student' ? 'user' : 'bot',
                            message.content,
                            message.citations || []
                        );
                    });
                }
            } catch (error) {
                console.error('Error loading conversation history:', error);
            }
        }
    }
    
    async clearConversationHistory() {
        if (this.sessionId) {
            try {
                await fetch(`${this.options.apiUrl}/api/session/${this.sessionId}/clear`, {
                    method: 'POST'
                });
                
                // Clear UI
                document.getElementById('chat-messages').innerHTML = '';
                this.conversationHistory = [];
                
                // Add welcome message
                this.addMessageToUI('bot', 'Conversation history cleared. How can I help you today?');
            } catch (error) {
                console.error('Error clearing conversation history:', error);
            }
        }
    }
    
    // Method to update module context
    updateModuleContext(newModule) {
        this.currentModule = newModule;
        document.getElementById('current-module').textContent = newModule;
    }
    
    // Method to get conversation context for the RAG agent
    getRecentConversationContext(limit = 10) {
        return this.conversationHistory.slice(-limit).map(msg => ({
            role: msg.sender_type === 'student' ? 'user' : 'assistant',
            content: msg.content,
            timestamp: msg.timestamp
        }));
    }
}

// Export for use in other modules if needed
if (typeof module !== 'undefined' && module.exports) {
    module.exports = RAGChatWidget;
}