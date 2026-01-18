/**
 * Adaptive UI Extension - Content Script
 *
 * Applies UI adaptations to web pages based on cognitive load data
 * received from the background service worker.
 */

// State
let adaptationActive = false;
let originalStyles = new Map();
let gazeIndicator = null;
let lastGazeCoords = null;
let animationFrameId = null;

// Selectors for different element types
const SELECTORS = {
  // Non-essential elements to hide/fade during simplification
  nonEssential: [
    '[role="complementary"]',
    '[role="banner"]:not(:first-of-type)',
    'aside:not([role="main"])',
    '.sidebar', '.side-bar', '[class*="sidebar"]',
    '.advertisement', '.ad', '[class*="-ad-"]', '[class*="_ad_"]', '[id*="-ad-"]',
    '.social-share', '.social-buttons', '[class*="social"]',
    '.related-posts', '.related-articles', '.recommended',
    '.comments-section', '#comments',
    '.newsletter', '.subscribe',
    'footer:not([role="contentinfo"])',
    '.cookie-banner', '.cookie-notice',
    '[class*="popup"]', '[class*="modal"]:not([role="dialog"])',
    '.promo', '.promotion', '[class*="promo"]'
  ],

  // Interactive elements to highlight during guidance
  interactive: [
    'button:not([disabled])',
    'a[href]:not([disabled])',
    'input:not([disabled]):not([type="hidden"])',
    'select:not([disabled])',
    'textarea:not([disabled])',
    '[role="button"]:not([disabled])',
    '[tabindex]:not([tabindex="-1"])',
    '[onclick]'
  ],

  // Text containers for layout adjustments
  textContainers: [
    'p', 'li', 'td', 'th', 'span', 'div', 'article', 'section',
    'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
    'label', 'figcaption'
  ]
};

/**
 * Initialize the content script
 */
function init() {
  console.log('[AdaptiveUI] Content script loaded');

  // Listen for messages from background
  chrome.runtime.onMessage.addListener(handleMessage);

  // Create gaze indicator element
  createGazeIndicator();
}

/**
 * Handle messages from background service worker
 */
function handleMessage(message, sender, sendResponse) {
  switch (message.type) {
    case 'apply_adaptations':
      applyAdaptations(message.adaptations, message.cognitiveLoad, message.gazeCoords);
      break;

    case 'revert_adaptations':
      revertAllAdaptations();
      break;

    case 'connection_status':
      handleConnectionStatus(message.connected);
      break;

    default:
      console.log('[AdaptiveUI] Unknown message:', message.type);
  }
}

/**
 * Apply adaptations based on commands from server
 */
function applyAdaptations(commands, cognitiveLoad, gazeCoords) {
  if (!commands) return;

  const params = commands.parameters || {};

  // Update gaze indicator position
  if (gazeCoords) {
    updateGazeIndicator(gazeCoords);
  }

  // Apply different adaptation modes
  if (commands.simplify) {
    applySimplification(params);
  } else if (adaptationActive) {
    revertSimplification();
  }

  if (commands.layout) {
    applyLayoutChanges(params);
  } else if (adaptationActive) {
    revertLayoutChanges();
  }

  if (commands.guidance) {
    applyGuidance(params);
  } else if (adaptationActive) {
    revertGuidance();
  }

  // Track if any adaptation is active
  adaptationActive = commands.simplify || commands.layout || commands.guidance;

  // Add/remove body class
  if (adaptationActive) {
    document.body.classList.add('adaptive-ui-active');
  } else {
    document.body.classList.remove('adaptive-ui-active');
  }
}

/**
 * Apply simplification - hide/fade non-essential elements
 */
function applySimplification(params) {
  const selector = SELECTORS.nonEssential.join(', ');

  try {
    const elements = document.querySelectorAll(selector);

    elements.forEach(el => {
      // Skip if already processed
      if (el.dataset.adaptiveSimplified) return;

      // Store original styles
      if (!originalStyles.has(el)) {
        originalStyles.set(el, {
          opacity: el.style.opacity,
          filter: el.style.filter,
          transition: el.style.transition
        });
      }

      // Apply fade effect
      el.style.transition = 'opacity 0.3s ease, filter 0.3s ease';
      el.style.opacity = params.reduce_visual_elements ? String(1 - params.reduce_visual_elements) : '0.3';
      el.style.filter = 'grayscale(50%)';
      el.dataset.adaptiveSimplified = 'true';
    });

    document.body.classList.add('adaptive-simplified');

  } catch (error) {
    console.error('[AdaptiveUI] Error applying simplification:', error);
  }
}

/**
 * Revert simplification changes
 */
function revertSimplification() {
  try {
    const elements = document.querySelectorAll('[data-adaptive-simplified]');

    elements.forEach(el => {
      const original = originalStyles.get(el);
      if (original) {
        el.style.opacity = original.opacity;
        el.style.filter = original.filter;
        el.style.transition = original.transition;
      } else {
        el.style.opacity = '';
        el.style.filter = '';
      }
      delete el.dataset.adaptiveSimplified;
    });

    document.body.classList.remove('adaptive-simplified');

  } catch (error) {
    console.error('[AdaptiveUI] Error reverting simplification:', error);
  }
}

/**
 * Apply layout changes - increase font size, spacing 
 */
function applyLayoutChanges(params) {
  const fontMultiplier = params.font_size_multiplier || 1.2;
  const spacingMultiplier = params.whitespace_multiplier || 1.5;

  // Set CSS custom properties on root
  document.documentElement.style.setProperty('--adaptive-font-scale', String(fontMultiplier));
  document.documentElement.style.setProperty('--adaptive-spacing-scale', String(spacingMultiplier));

  try {
    const selector = SELECTORS.textContainers.join(', ');
    const elements = document.querySelectorAll(selector);

    elements.forEach(el => {
      // Skip if already processed or too deeply nested
      if (el.dataset.adaptiveLayout) return;

      const computedStyle = window.getComputedStyle(el);
      const fontSize = parseFloat(computedStyle.fontSize);
      const lineHeight = parseFloat(computedStyle.lineHeight) || fontSize * 1.2;
      const marginBottom = parseFloat(computedStyle.marginBottom) || 0;

      // Skip elements with unusual font sizes
      if (fontSize <= 0 || fontSize > 100) return;

      // Store original styles
      if (!originalStyles.has(el)) {
        originalStyles.set(el, {
          fontSize: el.style.fontSize,
          lineHeight: el.style.lineHeight,
          marginBottom: el.style.marginBottom,
          transition: el.style.transition
        });
      }

      // Apply layout changes with smooth transition
      el.style.transition = 'font-size 0.3s ease, line-height 0.3s ease, margin 0.3s ease';
      el.style.fontSize = `${fontSize * fontMultiplier}px`;
      el.style.lineHeight = '1.6';
      el.style.marginBottom = `${marginBottom * spacingMultiplier}px`;
      el.dataset.adaptiveLayout = 'true';
    });

    document.body.classList.add('adaptive-layout');

  } catch (error) {
    console.error('[AdaptiveUI] Error applying layout changes:', error);
  }
}

/**
 * Revert layout changes
 */
function revertLayoutChanges() {
  // Remove CSS custom properties
  document.documentElement.style.removeProperty('--adaptive-font-scale');
  document.documentElement.style.removeProperty('--adaptive-spacing-scale');

  try {
    const elements = document.querySelectorAll('[data-adaptive-layout]');

    elements.forEach(el => {
      const original = originalStyles.get(el);
      if (original) {
        el.style.fontSize = original.fontSize;
        el.style.lineHeight = original.lineHeight;
        el.style.marginBottom = original.marginBottom;
        el.style.transition = original.transition;
      } else {
        el.style.fontSize = '';
        el.style.lineHeight = '';
        el.style.marginBottom = '';
      }
      delete el.dataset.adaptiveLayout;
    });

    document.body.classList.remove('adaptive-layout');

  } catch (error) {
    console.error('[AdaptiveUI] Error reverting layout changes:', error);
  }
}

/**
 * Apply guidance - highlight interactive elements
 */
function applyGuidance(params) {
  const highlightColor = params.highlight_color || '#FFD700';

  // Set highlight color as CSS variable
  document.documentElement.style.setProperty('--adaptive-highlight-color', highlightColor);

  try {
    const selector = SELECTORS.interactive.join(', ');
    const elements = document.querySelectorAll(selector);

    elements.forEach(el => {
      if (el.dataset.adaptiveGuidance) return;

      el.classList.add('adaptive-interactive');
      el.dataset.adaptiveGuidance = 'true';
    });

    // Show tooltips for form inputs if enabled
    if (params.show_tooltips) {
      enhanceFormFields();
    }

    document.body.classList.add('adaptive-guidance');

  } catch (error) {
    console.error('[AdaptiveUI] Error applying guidance:', error);
  }
}

/**
 * Revert guidance changes
 */
function revertGuidance() {
  document.documentElement.style.removeProperty('--adaptive-highlight-color');

  try {
    const elements = document.querySelectorAll('[data-adaptive-guidance]');

    elements.forEach(el => {
      el.classList.remove('adaptive-interactive');
      delete el.dataset.adaptiveGuidance;
    });

    // Remove any tooltips we added
    document.querySelectorAll('.adaptive-tooltip').forEach(el => el.remove());

    document.body.classList.remove('adaptive-guidance');

  } catch (error) {
    console.error('[AdaptiveUI] Error reverting guidance:', error);
  }
}

/**
 * Enhance form fields with helpful tooltips
 */
function enhanceFormFields() {
  const inputs = document.querySelectorAll('input:not([type="hidden"]), select, textarea');

  inputs.forEach(input => {
    // Skip if already has tooltip
    if (input.dataset.adaptiveTooltip) return;

    // Get label text
    let labelText = '';
    const label = input.labels?.[0] || document.querySelector(`label[for="${input.id}"]`);
    if (label) {
      labelText = label.textContent.trim();
    } else if (input.placeholder) {
      labelText = input.placeholder;
    } else if (input.name) {
      labelText = input.name.replace(/[_-]/g, ' ');
    }

    if (labelText) {
      input.setAttribute('title', `Fill in: ${labelText}`);
      input.dataset.adaptiveTooltip = 'true';
    }
  });
}

/**
 * Create gaze indicator element
 */
function createGazeIndicator() {
  if (gazeIndicator) return;

  gazeIndicator = document.createElement('div');
  gazeIndicator.className = 'adaptive-gaze-indicator';
  gazeIndicator.style.display = 'none';
  document.body.appendChild(gazeIndicator);
}

/**
 * Update gaze indicator position
 */
function updateGazeIndicator(coords) {
  if (!gazeIndicator || !coords) return;

  lastGazeCoords = coords;

  // Convert normalized coords [0,1] to screen position
  const x = coords[0] * window.innerWidth;
  const y = coords[1] * window.innerHeight;

  // Use requestAnimationFrame for smooth updates
  if (animationFrameId) {
    cancelAnimationFrame(animationFrameId);
  }

  animationFrameId = requestAnimationFrame(() => {
    gazeIndicator.style.left = `${x}px`;
    gazeIndicator.style.top = `${y}px`;
    gazeIndicator.style.display = 'block';
  });
}

/**
 * Hide gaze indicator
 */
function hideGazeIndicator() {
  if (gazeIndicator) {
    gazeIndicator.style.display = 'none';
  }
}

/**
 * Handle connection status changes
 */
function handleConnectionStatus(connected) {
  if (!connected) {
    // Revert all adaptations when disconnected
    setTimeout(() => {
      if (!adaptationActive) return;
      revertAllAdaptations();
      hideGazeIndicator();
    }, 5000); // Wait 5 seconds before reverting
  }
}

/**
 * Revert all adaptations
 */
function revertAllAdaptations() {
  revertSimplification();
  revertLayoutChanges();
  revertGuidance();
  hideGazeIndicator();

  // Clear stored styles
  originalStyles.clear();

  // Remove body classes
  document.body.classList.remove('adaptive-ui-active', 'adaptive-simplified', 'adaptive-layout', 'adaptive-guidance');

  adaptationActive = false;
  console.log('[AdaptiveUI] All adaptations reverted');
}

// Initialize
init();
