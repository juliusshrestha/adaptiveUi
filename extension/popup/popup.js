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
const enableToggle = document.getElementById('enableToggle');
const gazeModeSelect = document.getElementById('gazeModeSelect');
const calibrateCenterBtn = document.getElementById('calibrateCenterBtn');
const sensitivitySlider = document.getElementById('sensitivitySlider');
const sensitivityValue = document.getElementById('sensitivityValue');
const adaptationsSection = document.getElementById('adaptationsSection');
const adaptationsList = document.getElementById('adaptationsList');
const reconnectBtn = document.getElementById('reconnectBtn');
const fpsDisplay = document.getElementById('fpsDisplay');
const toggleTrackingBtn = document.getElementById('toggleTrackingBtn');
const startServerBtn = document.getElementById('startServerBtn');
const stopServerBtn = document.getElementById('stopServerBtn');

// State
let isConnected = false;
let updateInterval = null;

/**
 * Initialize popup
 */
async function init() {
  // Load saved settings
  const settings = await chrome.storage.local.get(['adaptationsEnabled', 'sensitivity', 'gazeMode']);
  enableToggle.checked = settings.adaptationsEnabled !== false;
  sensitivitySlider.value = settings.sensitivity || 50;
  sensitivityValue.textContent = `${sensitivitySlider.value}%`;
  gazeModeSelect.value = settings.gazeMode || 'direct';
  updateCalibrateButtonState();

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
  // Enable toggle
  enableToggle.addEventListener('change', async () => {
    const enabled = enableToggle.checked;
    chrome.runtime.sendMessage({ type: 'toggle_adaptations', enabled });
    chrome.storage.local.set({ adaptationsEnabled: enabled });
  });

  // Gaze mode selector
  gazeModeSelect.addEventListener('change', async () => {
    const mode = gazeModeSelect.value;
    chrome.runtime.sendMessage({ type: 'send_config', config: { gaze_mode: mode } });
    chrome.storage.local.set({ gazeMode: mode });
    updateCalibrateButtonState();
  });

  // Center calibration (only meaningful in monitor_plane mode)
  calibrateCenterBtn.addEventListener('click', async () => {
    chrome.runtime.sendMessage({ type: 'send_command', command: 'calibrate_center' });
  });

  // Start/stop tracking (pause/resume processing loop on backend)
  toggleTrackingBtn.addEventListener('click', async () => {
    if (!isConnected) {
      statusText.textContent = 'Disconnected (start server first)';
      return;
    }
    const storage = await chrome.storage.local.get(['serverStatus']);
    const paused = !!(storage.serverStatus && storage.serverStatus.paused);
    chrome.runtime.sendMessage({ type: 'send_command', command: paused ? 'resume' : 'pause' });
    // Ask for status refresh shortly after.
    setTimeout(() => chrome.runtime.sendMessage({ type: 'send_command', command: 'status' }), 250);
  });

  // Start server: extensions cannot launch local processes directly, so we copy the command.
  startServerBtn.addEventListener('click', async () => {
    const cmd = 'python -m src.server.run_server';
    try {
      await navigator.clipboard.writeText(cmd);
      statusText.textContent = 'Start command copied (run in terminal)';
    } catch (e) {
      statusText.textContent = 'Run in terminal: python -m src.server.run_server';
    }
    chrome.runtime.sendMessage({ type: 'reconnect' });
  });

  // Stop server: graceful shutdown via WebSocket command.
  stopServerBtn.addEventListener('click', async () => {
    if (!isConnected) {
      statusText.textContent = 'Disconnected';
      return;
    }
    chrome.runtime.sendMessage({ type: 'send_command', command: 'shutdown' });
    statusText.textContent = 'Stopping server...';
  });

  // Sensitivity slider
  sensitivitySlider.addEventListener('input', () => {
    sensitivityValue.textContent = `${sensitivitySlider.value}%`;
  });

  sensitivitySlider.addEventListener('change', async () => {
    const sensitivity = sensitivitySlider.value / 100;
    chrome.runtime.sendMessage({ type: 'send_config', config: { sensitivity } });
    chrome.storage.local.set({ sensitivity: sensitivitySlider.value });
  });

  // Reconnect button
  reconnectBtn.addEventListener('click', () => {
    chrome.runtime.sendMessage({ type: 'reconnect' });
    statusText.textContent = 'Reconnecting...';
    reconnectBtn.style.display = 'none';
  });
}

function updateCalibrateButtonState() {
  const mode = gazeModeSelect.value;
  // Only enable for monitor_plane mode (backend supports calibrate_center there)
  calibrateCenterBtn.disabled = mode !== 'monitor_plane';
  calibrateCenterBtn.title = mode === 'monitor_plane'
    ? 'Look at the center of your screen, then click to calibrate.'
    : 'Switch to Monitor Plane mode to use center calibration.';
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
      enableToggle.checked = response.adaptationsEnabled;
    }

    // Get server status from storage
    const storage = await chrome.storage.local.get(['serverStatus', 'lastAdaptation']);

    if (storage.serverStatus) {
      fpsDisplay.textContent = `${storage.serverStatus.fps || '--'} FPS`;
    }
    updateBackendControls(storage.serverStatus);

  } catch (error) {
    console.error('Error getting status:', error);
  }
}

function updateBackendControls(serverStatus) {
  // Tracking button label
  if (!isConnected) {
    toggleTrackingBtn.textContent = '--';
    toggleTrackingBtn.disabled = true;
    stopServerBtn.disabled = true;
    return;
  }

  toggleTrackingBtn.disabled = false;
  stopServerBtn.disabled = false;

  const paused = !!(serverStatus && serverStatus.paused);
  toggleTrackingBtn.textContent = paused ? 'Start' : 'Stop';
  toggleTrackingBtn.title = paused ? 'Resume processing' : 'Pause processing';
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
  const adaptations = data.adaptations || [];

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

  // Update adaptations list
  updateAdaptationsList(adaptations, data.adaptation_commands);
}

/**
 * Update adaptations list display
 */
function updateAdaptationsList(adaptations, commands) {
  if (!adaptations || adaptations.length === 0) {
    adaptationsSection.style.display = 'none';
    return;
  }

  adaptationsSection.style.display = 'block';

  // Map adaptation names to friendly labels
  const labels = {
    'simplification': 'Simplifying UI',
    'hide_decorative': 'Hiding decorative elements',
    'guidance': 'Showing guidance',
    'highlight_next_step': 'Highlighting next step',
    'show_tooltips': 'Showing tooltips',
    'layout_reorganization': 'Reorganizing layout',
    'whitespace_increase': 'Increasing whitespace',
    'font_size_increase': 'Increasing font size'
  };

  adaptationsList.innerHTML = adaptations
    .map(a => `<span class="adaptation-tag">${labels[a] || a}</span>`)
    .join('');
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
