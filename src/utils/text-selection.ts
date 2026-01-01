/**
 * Utility functions for detecting and handling text selection on the page
 */

// Interface for text selection data
export interface TextSelection {
  text: string;
  element: Element | null;
  rect: DOMRect | null;
}

/**
 * Gets the currently selected text on the page
 * @returns The selected text and related information
 */
export const getSelectedText = (): TextSelection => {
  const selection = window.getSelection();
  const text = selection?.toString().trim() || '';
  
  if (!text) {
    return { text: '', element: null, rect: null };
  }
  
  // Get the anchor and focus nodes to determine where the selection is
  const range = selection?.getRangeAt(0);
  const rect = range?.getBoundingClientRect() || null;
  
  // Get the element that contains the selection
  let element: Element | null = null;
  if (selection?.anchorNode) {
    element = selection.anchorNode.nodeType === Node.ELEMENT_NODE 
      ? selection.anchorNode as Element 
      : selection.anchorNode.parentElement;
  }
  
  return { text, element, rect };
};

/**
 * Sets up a global event listener to detect text selection
 * @param callback Function to call when text is selected
 * @returns A function to remove the event listener
 */
export const setupTextSelectionListener = (callback: (selection: TextSelection) => void) => {
  let selectionTimeout: NodeJS.Timeout | null = null;
  
  const handleSelection = () => {
    // Clear any existing timeout
    if (selectionTimeout) {
      clearTimeout(selectionTimeout);
    }
    
    // Use a timeout to ensure selection is complete
    selectionTimeout = setTimeout(() => {
      const selectionData = getSelectedText();
      if (selectionData.text) {
        callback(selectionData);
      }
    }, 150); // Delay to ensure selection is complete
  };
  
  // Add event listeners
  document.addEventListener('mouseup', handleSelection);
  document.addEventListener('keyup', handleSelection);
  
  // Return cleanup function
  return () => {
    document.removeEventListener('mouseup', handleSelection);
    document.removeEventListener('keyup', handleSelection);
    
    if (selectionTimeout) {
      clearTimeout(selectionTimeout);
    }
  };
};

/**
 * Copies text to clipboard
 * @param text The text to copy
 * @returns Promise that resolves when text is copied
 */
export const copyToClipboard = async (text: string): Promise<boolean> => {
  try {
    if (navigator.clipboard) {
      await navigator.clipboard.writeText(text);
      return true;
    } else {
      // Fallback for older browsers
      const textArea = document.createElement('textarea');
      textArea.value = text;
      document.body.appendChild(textArea);
      textArea.select();
      
      const successful = document.execCommand('copy');
      document.body.removeChild(textArea);
      
      return successful;
    }
  } catch (err) {
    console.error('Failed to copy text: ', err);
    return false;
  }
};

/**
 * Gets the context around the selected text (e.g., paragraph or section)
 * @param selection The text selection data
 * @returns Context around the selected text
 */
export const getTextContext = (selection: TextSelection): string => {
  if (!selection.element) {
    return '';
  }
  
  // Try to get context from the parent element
  let contextElement = selection.element.closest('p, div, section, article');
  
  if (!contextElement) {
    // If no semantic element found, use the direct parent
    contextElement = selection.element.parentElement;
  }
  
  if (!contextElement) {
    return selection.text;
  }
  
  // Get the text content of the context element
  const contextText = contextElement.textContent?.trim() || '';
  
  // Return the context with the selected text highlighted
  return contextText;
};