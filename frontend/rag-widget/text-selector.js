/**
 * Text Selection Handler for RAG Chatbot
 * Handles text selection, highlighting, and "Ask about this" UI
 */

class TextSelectionHandler {
    constructor(widget, options = {}) {
        this.widget = widget; // Reference to the main chat widget
        this.options = {
            minSelectionLength: 20,
            ...options
        };
        
        this.selectionPopup = null;
        this.currentSelection = null;
        
        this.init();
    }
    
    init() {
        this.createSelectionPopup();
        this.setupEventListeners();
    }
    
    createSelectionPopup() {
        // Create the popup element if it doesn't exist
        if (!document.getElementById('text-selection-popup')) {
            const popup = document.createElement('div');
            popup.id = 'text-selection-popup';
            popup.className = 'text-selection-popup';
            popup.style.cssText = `
                position: absolute;
                background: #3498db;
                color: white;
                padding: 8px 12px;
                border-radius: 4px;
                cursor: pointer;
                z-index: 10001;
                display: none;
                box-shadow: 0 2px 10px rgba(0,0,0,0.2);
                font-size: 14px;
                font-weight: bold;
                user-select: none;
            `;
            popup.textContent = 'Ask about this';
            
            document.body.appendChild(popup);
            this.selectionPopup = popup;
        } else {
            this.selectionPopup = document.getElementById('text-selection-popup');
        }
    }
    
    setupEventListeners() {
        // Listen for text selection
        document.addEventListener('mouseup', () => {
            setTimeout(() => { // Allow selection to complete
                this.handleTextSelection();
            }, 1);
        });
        
        // Listen for popup click
        this.selectionPopup.addEventListener('click', () => {
            this.handlePopupClick();
        });
        
        // Hide popup when clicking elsewhere
        document.addEventListener('click', (e) => {
            if (!this.selectionPopup.contains(e.target) && 
                !['text-selection-popup', 'user-message'].includes(e.target.id)) {
                this.hideSelectionPopup();
            }
        });
    }
    
    handleTextSelection() {
        const selection = window.getSelection();
        const selectedText = selection.toString().trim();
        
        if (selectedText.length >= this.options.minSelectionLength) {
            this.currentSelection = {
                text: selectedText,
                range: selection.getRangeAt(0),
                rect: selection.getRangeAt(0).getBoundingClientRect()
            };
            
            this.showSelectionPopup(this.currentSelection.rect);
        } else {
            this.hideSelectionPopup();
            this.currentSelection = null;
        }
    }
    
    showSelectionPopup(rect) {
        if (!this.selectionPopup) return;
        
        // Position the popup above the selection
        this.selectionPopup.style.display = 'block';
        this.selectionPopup.style.top = `${rect.top + window.scrollY - 40}px`;
        this.selectionPopup.style.left = `${rect.left + window.scrollX + (rect.width / 2) - (this.selectionPopup.offsetWidth / 2)}px`;
    }
    
    hideSelectionPopup() {
        if (this.selectionPopup) {
            this.selectionPopup.style.display = 'none';
        }
        this.currentSelection = null;
    }
    
    async handlePopupClick() {
        if (!this.currentSelection) return;
        
        const selectedText = this.currentSelection.text;
        
        // Hide the popup
        this.hideSelectionPopup();
        
        // Open the chat widget if it's minimized
        const container = document.getElementById(this.widget.containerId);
        if (container) {
            container.style.display = 'block';
        }
        
        // Focus the chat input and prefill with the selected text
        const inputElement = document.getElementById('user-message');
        if (inputElement) {
            // Clear any existing text and focus
            inputElement.value = '';
            inputElement.focus();
            
            // Show a prompt to the user
            const messagesContainer = document.getElementById('chat-messages');
            const messageDiv = document.createElement('div');
            messageDiv.className = 'rag-message rag-message-system';
            messageDiv.innerHTML = `
                <div class="rag-message-content">
                    <strong>You selected:</strong> "${selectedText.substring(0, 100)}${selectedText.length > 100 ? '...' : ''}"
                    <br><br>
                    What would you like to know about this text?
                </div>
                <div class="rag-message-timestamp">${new Date().toLocaleTimeString()}</div>
            `;
            messagesContainer.appendChild(messageDiv);
            
            // Scroll to bottom
            messagesContainer.scrollTop = messagesContainer.scrollHeight;
        }
        
        // Store the selected text context for the next query
        window.ragSelectedTextContext = selectedText;
    }
    
    // Method to get currently selected text (for API calls)
    getCurrentSelectedText() {
        return this.currentSelection ? this.currentSelection.text : null;
    }
    
    // Method to clear selection context
    clearSelectionContext() {
        this.currentSelection = null;
        if (window.ragSelectedTextContext) {
            delete window.ragSelectedTextContext;
        }
    }
}

// Export for use in other modules if needed
if (typeof module !== 'undefined' && module.exports) {
    module.exports = TextSelectionHandler;
}