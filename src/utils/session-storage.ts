/**
 * Utility functions for managing session storage of thread IDs
 * and conversation continuity across page navigation
 */

const THREAD_ID_KEY = 'doc-assistant-thread-id';
const CONVERSATION_HISTORY_KEY = 'doc-assistant-conversation-history';

// Define the structure for conversation history
interface ConversationHistoryItem {
  threadId: string;
  title: string;
  lastMessage: string;
  timestamp: string;
}

/**
 * Saves the thread ID to localStorage
 * @param threadId The thread ID to save
 */
export const saveThreadId = (threadId: string): void => {
  try {
    localStorage.setItem(THREAD_ID_KEY, threadId);
  } catch (error) {
    console.error('Error saving thread ID to localStorage:', error);
  }
};

/**
 * Retrieves the saved thread ID from localStorage
 * @returns The saved thread ID or null if not found
 */
export const getSavedThreadId = (): string | null => {
  try {
    return localStorage.getItem(THREAD_ID_KEY);
  } catch (error) {
    console.error('Error retrieving thread ID from localStorage:', error);
    return null;
  }
};

/**
 * Removes the saved thread ID from localStorage
 */
export const removeSavedThreadId = (): void => {
  try {
    localStorage.removeItem(THREAD_ID_KEY);
  } catch (error) {
    console.error('Error removing thread ID from localStorage:', error);
  }
};

/**
 * Saves conversation history item to localStorage
 * @param threadId The thread ID
 * @param title The conversation title
 * @param lastMessage The last message in the conversation
 */
export const saveConversationToHistory = (threadId: string, title: string, lastMessage: string): void => {
  try {
    // Get existing history
    const existingHistory = getConversationHistory();
    
    // Check if this thread already exists in history
    const existingIndex = existingHistory.findIndex(item => item.threadId === threadId);
    
    // Remove if it already exists to update its position
    if (existingIndex !== -1) {
      existingHistory.splice(existingIndex, 1);
    }
    
    // Add the new item at the beginning
    const newItem: ConversationHistoryItem = {
      threadId,
      title,
      lastMessage: lastMessage.substring(0, 50) + (lastMessage.length > 50 ? '...' : ''), // Truncate last message
      timestamp: new Date().toISOString()
    };
    
    existingHistory.unshift(newItem);
    
    // Keep only the last 10 conversations
    const recentHistory = existingHistory.slice(0, 10);
    
    localStorage.setItem(CONVERSATION_HISTORY_KEY, JSON.stringify(recentHistory));
  } catch (error) {
    console.error('Error saving conversation to history:', error);
  }
};

/**
 * Retrieves the conversation history from localStorage
 * @returns Array of conversation history items
 */
export const getConversationHistory = (): ConversationHistoryItem[] => {
  try {
    const historyString = localStorage.getItem(CONVERSATION_HISTORY_KEY);
    if (!historyString) {
      return [];
    }
    
    const history = JSON.parse(historyString);
    
    // Validate the structure of the history items
    if (Array.isArray(history)) {
      return history.map(item => ({
        threadId: item.threadId || '',
        title: item.title || 'Untitled Conversation',
        lastMessage: item.lastMessage || '',
        timestamp: item.timestamp || new Date().toISOString()
      }));
    }
    
    return [];
  } catch (error) {
    console.error('Error retrieving conversation history:', error);
    return [];
  }
};

/**
 * Clears the conversation history from localStorage
 */
export const clearConversationHistory = (): void => {
  try {
    localStorage.removeItem(CONVERSATION_HISTORY_KEY);
  } catch (error) {
    console.error('Error clearing conversation history:', error);
  }
};

/**
 * Checks if the current session has a saved thread
 * @returns True if there's a saved thread, false otherwise
 */
export const hasSavedThread = (): boolean => {
  return getSavedThreadId() !== null;
};

/**
 * Initializes the session with a new thread ID if none exists
 * @param newThreadId The new thread ID to save
 * @returns The thread ID that should be used
 */
export const initializeSession = (newThreadId: string): string => {
  const savedThreadId = getSavedThreadId();
  
  if (savedThreadId) {
    return savedThreadId;
  }
  
  saveThreadId(newThreadId);
  return newThreadId;
};