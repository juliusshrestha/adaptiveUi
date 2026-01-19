/**
 * Adaptive UI Extension - Background Service Worker
 *
 * Maintains WebSocket connection to the Python backend server
 * and routes adaptation commands to content scripts.
 */

// Configuration
const DEFAULT_SERVER_URL = 'ws://127.0.0.1:8765';
const RECONNECT_DELAYS = [1000, 2000, 4000, 8000, 16000, 30000]; // Exponential backoff
const KEEPALIVE_INTERVAL = 25000; // 25 seconds

// State
let ws = null;
let isConnected = false;
let reconnectAttempt = 0;
let reconnectTimeout = null;
let keepaliveInterval = null;
let lastAdaptationData = null;
// Adaptations are disabled (read-only mode).
let adaptationsEnabled = false;

/**
 * Initialize the service worker
 */
async function init() {
  console.log('[AdaptiveUI] Service worker initializing...');

  // Load settings from storage (serverUrl only)
  const settings = await chrome.storage.local.get(['serverUrl']);

  // Ensure any previously applied adaptations are reverted on startup
  broadcastToContentScripts({ type: 'revert_adaptations' });

  // Connect to server
  connect(settings.serverUrl || DEFAULT_SERVER_URL);

  // Setup message listeners
  setupMessageListeners();
}

/**
 * Connect to the WebSocket server
 */
function connect(serverUrl = DEFAULT_SERVER_URL) {
  if (ws && ws.readyState === WebSocket.OPEN) {
    console.log('[AdaptiveUI] Already connected');
    return;
  }

  console.log(`[AdaptiveUI] Connecting to ${serverUrl}...`);

  try {
    ws = new WebSocket(serverUrl);

    ws.onopen = handleOpen;
    ws.onmessage = handleMessage;
    ws.onclose = handleClose;
    ws.onerror = handleError;

  } catch (error) {
    console.error('[AdaptiveUI] Connection error:', error);
    scheduleReconnect();
  }
}

/**
 * Handle WebSocket open event
 */
function handleOpen() {
  console.log('[AdaptiveUI] Connected to server');
  isConnected = true;
  reconnectAttempt = 0;

  // Update storage
  chrome.storage.local.set({ connectionStatus: 'connected' });

  // Notify all tabs
  broadcastToContentScripts({ type: 'connection_status', connected: true });

  // Read-only mode: ensure adaptations are reverted even while connected
  broadcastToContentScripts({ type: 'revert_adaptations' });

  // Start keepalive
  startKeepalive();

  // Request initial status
  sendCommand('status');
}

/**
 * Handle WebSocket message event
 */
function handleMessage(event) {
  try {
    const message = JSON.parse(event.data);

    switch (message.type) {
      case 'adaptation_update':
        handleAdaptationUpdate(message);
        break;

      case 'status':
        handleStatusUpdate(message);
        break;

      case 'pong':
        // Keepalive response, do nothing
        break;

      case 'command_response':
        console.log('[AdaptiveUI] Command response:', message);
        break;

      case 'config_response':
        console.log('[AdaptiveUI] Config applied:', message);
        break;

      default:
        console.log('[AdaptiveUI] Unknown message type:', message.type);
    }

  } catch (error) {
    console.error('[AdaptiveUI] Error parsing message:', error);
  }
}

/**
 * Handle adaptation update from server
 */
function handleAdaptationUpdate(message) {
  lastAdaptationData = message.data;

  // Store latest data
  chrome.storage.local.set({
    lastAdaptation: message.data,
    lastUpdateTime: Date.now()
  });

  // Adaptations disabled: never apply. Also ensure any prior changes are reverted.
  broadcastToContentScripts({ type: 'revert_adaptations' });
}

/**
 * Handle status update from server
 */
function handleStatusUpdate(message) {
  chrome.storage.local.set({ serverStatus: message.data });
}

/**
 * Handle WebSocket close event
 */
function handleClose(event) {
  console.log(`[AdaptiveUI] Disconnected (code: ${event.code})`);
  isConnected = false;
  ws = null;

  // Stop keepalive
  stopKeepalive();

  // Update storage
  chrome.storage.local.set({ connectionStatus: 'disconnected' });

  // Notify all tabs
  broadcastToContentScripts({ type: 'connection_status', connected: false });

  // Schedule reconnection
  scheduleReconnect();
}

/**
 * Handle WebSocket error event
 */
function handleError(error) {
  console.error('[AdaptiveUI] WebSocket error:', error);
}

/**
 * Schedule a reconnection attempt with exponential backoff
 */
function scheduleReconnect() {
  if (reconnectTimeout) {
    clearTimeout(reconnectTimeout);
  }

  const delay = RECONNECT_DELAYS[Math.min(reconnectAttempt, RECONNECT_DELAYS.length - 1)];
  console.log(`[AdaptiveUI] Reconnecting in ${delay}ms (attempt ${reconnectAttempt + 1})`);

  reconnectTimeout = setTimeout(() => {
    reconnectAttempt++;
    connect();
  }, delay);
}

/**
 * Start keepalive ping to prevent connection timeout
 */
function startKeepalive() {
  stopKeepalive();

  keepaliveInterval = setInterval(() => {
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: 'ping', timestamp: Date.now() }));
    }
  }, KEEPALIVE_INTERVAL);
}

/**
 * Stop keepalive ping
 */
function stopKeepalive() {
  if (keepaliveInterval) {
    clearInterval(keepaliveInterval);
    keepaliveInterval = null;
  }
}

/**
 * Send a command to the server
 */
function sendCommand(command) {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: 'command', command }));
  } else {
    console.warn('[AdaptiveUI] Cannot send command - not connected');
  }
}

/**
 * Send configuration to the server
 */
function sendConfig(config) {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: 'config', config }));
  } else {
    console.warn('[AdaptiveUI] Cannot send config - not connected');
  }
}

/**
 * Broadcast a message to all content scripts
 */
async function broadcastToContentScripts(message) {
  try {
    const tabs = await chrome.tabs.query({});

    for (const tab of tabs) {
      if (tab.id && tab.url && !tab.url.startsWith('chrome://')) {
        try {
          await chrome.tabs.sendMessage(tab.id, message);
        } catch (error) {
          // Tab may not have content script loaded, ignore
        }
      }
    }
  } catch (error) {
    console.error('[AdaptiveUI] Error broadcasting:', error);
  }
}

/**
 * Setup message listeners for popup and content scripts
 */
function setupMessageListeners() {
  chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    switch (message.type) {
      case 'get_status':
        sendResponse({
          connected: isConnected,
          lastAdaptation: lastAdaptationData,
          adaptationsEnabled: false
        });
        break;

      case 'send_command':
        sendCommand(message.command);
        sendResponse({ success: true });
        break;

      case 'send_config':
        sendConfig(message.config);
        sendResponse({ success: true });
        break;

      case 'reconnect':
        if (!isConnected) {
          reconnectAttempt = 0;
          connect();
        }
        sendResponse({ success: true });
        break;

      default:
        console.log('[AdaptiveUI] Unknown message from popup/content:', message);
    }

    return true; // Keep channel open for async response
  });
}

// Initialize on load
init();

// Re-initialize on service worker activation (after being idle)
self.addEventListener('activate', () => {
  console.log('[AdaptiveUI] Service worker activated');
  if (!isConnected) {
    reconnectAttempt = 0;
    connect();
  }
});
