/**
 * Adaptive UI Extension - Popup Script
 *
 * Handles the popup UI interactions and displays real-time status.
 */

// Elements
const statusDot = document.getElementById('statusDot');
const statusText = document.getElementById('statusText');
const loadFill = document.getElementById('loadFill');
const loadValue = document.getElementById('loadValue');
const loadLevel = document.getElementById('loadLevel');
const gazeScore = document.getElementById('gazeScore');
const emotionScore = document.getElementById('emotionScore');
const mouseScore = document.getElementById('mouseScore');
const dominantEmotion = document.getElementById('dominantEmotion');
const processingToggleBtn = document.getElementById('processingToggleBtn');
const reconnectBtn = document.getElementById('reconnectBtn');
const fpsDisplay = document.getElementById('fpsDisplay');

// State
let isConnected = false;
let updateInterval = null;
let lastStatusRequestAt = 0;

/**
 * Initialize popup
 */
async function init() {
  // Setup event listeners
  setupEventListeners();

  // Get initial status
  updateStatus();

  // Start polling for updates
  updateInterval = setInterval(updateStatus, 500);
}

/**
 * Setup event listeners
 */
function setupEventListeners() {
  // Start/stop processing (this starts/stops the backend "main.py" pipeline only)
  processingToggleBtn.addEventListener('click', async () => {
    if (!isConnected) {
      statusText.textContent = 'Disconnected';
      return;
    }

    const storage = await chrome.storage.local.get(['serverStatus']);
    const active = !!(storage.serverStatus && storage.serverStatus.processing_active);

    chrome.runtime.sendMessage({
      type: 'send_command',
      command: active ? 'stop_processing' : 'start_processing'
    });

    // Refresh status shortly after
    setTimeout(() => chrome.runtime.sendMessage({ type: 'send_command', command: 'status' }), 250);
  });

  // Reconnect button
  reconnectBtn.addEventListener('click', () => {
    chrome.runtime.sendMessage({ type: 'reconnect' });
    statusText.textContent = 'Reconnecting...';
    reconnectBtn.style.display = 'none';
  });
}

/**
 * Update status from background
 */
async function updateStatus() {
  try {
    // Get status from background
    const response = await chrome.runtime.sendMessage({ type: 'get_status' });

    if (response) {
      updateConnectionStatus(response.connected);
      updateAdaptationData(response.lastAdaptation);
    }

    // Get server status from storage
    const storage = await chrome.storage.local.get(['serverStatus', 'lastAdaptation']);

    if (storage.serverStatus) {
      fpsDisplay.textContent = `${storage.serverStatus.fps || '--'} FPS`;
    }
    updateBackendControls(storage.serverStatus);

    // Keep status fresh so the toggle reflects real state (does NOT start processing).
    if (isConnected && Date.now() - lastStatusRequestAt > 1000) {
      lastStatusRequestAt = Date.now();
      chrome.runtime.sendMessage({ type: 'send_command', command: 'status' });
    }

  } catch (error) {
    console.error('Error getting status:', error);
  }
}

function updateBackendControls(serverStatus) {
  if (!isConnected) {
    processingToggleBtn.textContent = '--';
    processingToggleBtn.disabled = true;
    return;
  }

  if (!serverStatus) {
    processingToggleBtn.textContent = '--';
    processingToggleBtn.disabled = true;
    return;
  }

  processingToggleBtn.disabled = false;

  const active = !!serverStatus.processing_active;
  processingToggleBtn.textContent = active ? 'Stop' : 'Start';
  processingToggleBtn.title = active ? 'Stop processing (keep server running)' : 'Start processing';
}

/**
 * Update connection status display
 */
function updateConnectionStatus(connected) {
  isConnected = connected;

  if (connected) {
    statusDot.className = 'status-dot connected';
    statusText.textContent = 'Connected';
    reconnectBtn.style.display = 'none';
  } else {
    statusDot.className = 'status-dot disconnected';
    statusText.textContent = 'Disconnected';
    reconnectBtn.style.display = 'block';
  }
}

/**
 * Update adaptation data display
 */
function updateAdaptationData(data) {
  if (!data) {
    loadValue.textContent = '--';
    loadLevel.textContent = 'Unknown';
    loadFill.style.width = '0%';
    gazeScore.textContent = '--';
    emotionScore.textContent = '--';
    mouseScore.textContent = '--';
    dominantEmotion.textContent = '--';
    return;
  }

  const cognitiveLoad = data.cognitive_load || {};
  const emotion = data.emotion || {};

  // Update cognitive load display
  const score = cognitiveLoad.score || 0;
  const level = cognitiveLoad.level || 'unknown';

  loadValue.textContent = (score * 100).toFixed(0);
  loadLevel.textContent = capitalizeFirst(level);
  loadLevel.className = `load-level ${level}`;

  // Update load bar
  const percentage = Math.min(100, score * 100);
  loadFill.style.width = `${percentage}%`;
  loadFill.className = `load-fill ${level}`;

  // Update metric scores
  gazeScore.textContent = formatScore(cognitiveLoad.gaze_score);
  emotionScore.textContent = formatScore(cognitiveLoad.emotion_score);
  mouseScore.textContent = formatScore(cognitiveLoad.mouse_score);
  dominantEmotion.textContent = capitalizeFirst(emotion.dominant || 'unknown');
}

/**
 * Format score as percentage
 */
function formatScore(score) {
  if (score === undefined || score === null) return '--';
  return `${(score * 100).toFixed(0)}%`;
}

/**
 * Capitalize first letter
 */
function capitalizeFirst(str) {
  if (!str) return '';
  return str.charAt(0).toUpperCase() + str.slice(1);
}

// Cleanup on popup close
window.addEventListener('unload', () => {
  if (updateInterval) {
    clearInterval(updateInterval);
  }
});

// Initialize
init();
