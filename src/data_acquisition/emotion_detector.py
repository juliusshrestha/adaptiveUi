"""
Emotion Detection Module
Uses deep learning models for accurate facial emotion recognition
Processes video frames via CNN to detect facial expressions
"""

import numpy as np
from typing import Dict, Optional, Tuple
import cv2
import os
from pathlib import Path


class EmotionDetector:
    """
    Emotion detection using deep learning models for facial expression recognition
    Uses face detection + emotion classification for accurate results
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        use_fer: bool = True,
        use_vit: bool = True,
        temperature_scaling: float = 0.7,
        enable_neutral_bias_reduction: bool = True,
        neutral_bias_threshold: float = 0.35,
        neutral_bias_reduction_amount: float = 0.15
    ):
        """
        Initialize emotion detector

        Args:
            model_path: Path to TensorFlow/Keras model file (optional)
            use_fer: Whether to try FER library first (default: True)
            use_vit: Whether to use ViT model from Hugging Face (default: True)
            temperature_scaling: Temperature for softmax scaling (lower = more confident)
            enable_neutral_bias_reduction: Whether to reduce neutral bias
            neutral_bias_threshold: Threshold above which to reduce neutral probability
            neutral_bias_reduction_amount: Amount to reduce neutral probability
        """
        self.model_path = model_path
        self.use_fer = use_fer
        self.use_vit = use_vit
        self.fer_model = None
        self.keras_model = None
        self.vit_model = None
        self.vit_processor = None
        self.face_detector = None
        self.initialized = False

        # Emotion labels (ViT model uses these)
        self.emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        self.negative_emotions = ['angry', 'sad', 'fear', 'disgust']  # Stress/frustration indicators

        # ViT model configuration
        self.temperature_scaling = temperature_scaling
        self.enable_neutral_bias_reduction = enable_neutral_bias_reduction
        self.neutral_bias_threshold = neutral_bias_threshold
        self.neutral_bias_reduction_amount = neutral_bias_reduction_amount
        
    def setup_vit(self):
        """Initialize Vision Transformer model from Hugging Face"""
        try:
            from transformers import AutoImageProcessor, AutoModelForImageClassification
            import torch
            
            model_name = "mo-thecreator/vit-Facial-Expression-Recognition"
            print(f"Loading ViT model: {model_name}...")
            
            # Load processor and model
            self.vit_processor = AutoImageProcessor.from_pretrained(model_name)
            self.vit_model = AutoModelForImageClassification.from_pretrained(model_name)
            self.vit_model.eval()  # Set to evaluation mode
            
            # Move to GPU if available
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.vit_model.to(self.device)
            
            self.initialized = True
            print(f"✓ ViT Facial Expression Recognition model initialized successfully (device: {self.device})")
            return True
        except ImportError as e:
            print(f"Warning: transformers library not installed. Install with: pip install transformers torch")
            return False
        except Exception as e:
            print(f"Error initializing ViT model: {e}")
            return False
    
    def setup_fer(self):
        """Initialize FER (Facial Expression Recognition) library"""
        try:
            from fer import FER
            self.fer_model = FER(mtcnn=True)  # Use MTCNN for better face detection
            self.initialized = True
            print("✓ FER deep learning model initialized successfully")
            return True
        except ImportError:
            return False
        except Exception as e:
            print(f"Error initializing FER model: {e}")
            return False
    
    def setup_face_detector(self):
        """Setup face detector using OpenCV"""
        try:
            # Use OpenCV's built-in Haar Cascade (always available)
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_detector = cv2.CascadeClassifier(cascade_path)
            
            if self.face_detector.empty():
                raise Exception("Failed to load Haar Cascade")
            
            print("✓ Face detector initialized")
            return True
        except Exception as e:
            print(f"Warning: Face detector initialization failed: {e}")
            self.face_detector = None
            return False
    
    def download_emotion_model(self) -> Optional[str]:
        """
        Download a pre-trained emotion recognition model
        Returns path to model file or None
        """
        # For now, return None - user should provide their own model
        # In production, you could download from a URL
        print("Note: For best results, provide a pre-trained emotion model.")
        print("You can download models from:")
        print("  - https://github.com/oarriaga/face_classification")
        print("  - https://www.kaggle.com/datasets/msambare/fer2013")
        return None
    
    def setup_keras_model(self):
        """Initialize custom Keras model if provided"""
        if self.model_path is None:
            return False
        
        if not os.path.exists(self.model_path):
            print(f"Model file not found: {self.model_path}")
            return False
        
        try:
            import tensorflow as tf
            from tensorflow import keras
            
            self.keras_model = keras.models.load_model(self.model_path)
            print(f"✓ Custom Keras model loaded from {self.model_path}")
            return True
        except Exception as e:
            print(f"Error loading Keras model: {e}")
            return False
    
    def initialize(self):
        """Initialize emotion detection system"""
        # Try ViT model first (best accuracy, Vision Transformer)
        if self.use_vit:
            if self.setup_vit():
                if self.setup_face_detector():
                    return
        
        # Try FER library (easiest, has pre-trained models)
        if self.use_fer:
            if self.setup_fer():
                return
        
        # Try custom Keras model
        if self.model_path:
            if self.setup_keras_model():
                if self.setup_face_detector():
                    self.initialized = True
                    return
        
        # Setup face detector for fallback (at least we can detect faces)
        self.setup_face_detector()
        print("\n" + "="*60)
        print("IMPORTANT: Deep learning emotion model not available.")
        print("="*60)
        print("To enable accurate facial emotion recognition:")
        print("1. Install transformers and torch: pip install transformers torch")
        print("   (Uses ViT model: mo-thecreator/vit-Facial-Expression-Recognition)")
        print("2. Or install FER library: pip install fer")
        print("3. Or provide a Keras model path when initializing EmotionDetector")
        print("="*60 + "\n")
    
    def setup_tflite(self):
        """Initialize TensorFlow Lite interpreter (legacy method)"""
        self.initialize()
    
    def detect_faces(self, frame: np.ndarray) -> list:
        """Detect faces in frame with optimized single-pass detection"""
        if self.face_detector is None:
            return []

        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Optimized single-pass detection with balanced parameters
            # (Removed second detection attempt for better performance)
            faces = self.face_detector.detectMultiScale(
                gray,
                scaleFactor=1.1,      # Balanced scale factor
                minNeighbors=3,       # Balanced threshold - detects faces reliably
                minSize=(30, 30),     # Reasonable minimum size
                flags=cv2.CASCADE_SCALE_IMAGE
            )

            return faces.tolist() if len(faces) > 0 else []
        except Exception as e:
            # Use logger if available, otherwise fall back to print
            try:
                from src.utils.logger import get_logger
                logger = get_logger("adaptive_ui")
                logger.error(f"Error in face detection: {e}")
            except:
                print(f"Error in face detection: {e}")
            return []
    
    def extract_face_region(self, frame: np.ndarray, face_box: list) -> Optional[np.ndarray]:
        """Extract face region from frame"""
        x, y, w, h = face_box
        # Add padding
        padding = 20
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(frame.shape[1] - x, w + 2 * padding)
        h = min(frame.shape[0] - y, h + 2 * padding)
        
        face_roi = frame[y:y+h, x:x+w]
        if face_roi.size == 0:
            return None
        
        # Resize to standard size for emotion model (48x48 is common)
        face_resized = cv2.resize(face_roi, (48, 48))
        return face_resized
    
    def preprocess_face(self, face_image: np.ndarray) -> np.ndarray:
        """Preprocess face image for emotion model"""
        # Convert to grayscale
        if len(face_image.shape) == 3:
            gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_image
        
        # Normalize to [0, 1]
        normalized = gray.astype(np.float32) / 255.0
        
        # Reshape for model input
        if self.keras_model:
            input_shape = self.keras_model.input_shape
            if len(input_shape) == 4:
                normalized = np.expand_dims(normalized, axis=0)
                if input_shape[3] == 1:
                    normalized = np.expand_dims(normalized, axis=-1)
        
        return normalized
    
    def detect_emotion_fer(self, frame: np.ndarray) -> Dict[str, float]:
        """Detect emotion using FER library"""
        try:
            # FER library handles face detection and emotion recognition
            emotions = self.fer_model.detect_emotions(frame)
            
            if not emotions:
                return self._default_emotion_result()
            
            # Get the first (most prominent) face
            top_emotion = emotions[0]
            emotion_scores = top_emotion['emotions']
            
            # Convert to our format
            emotion_dict = {}
            for emotion in self.emotions:
                emotion_dict[emotion] = float(emotion_scores.get(emotion, 0.0))
            
            # Find dominant emotion
            dominant_emotion = max(emotion_dict.items(), key=lambda x: x[1])[0]
            
            # Calculate negative affect score (raw sum, no threshold)
            negative_affect_score = sum(emotion_dict[emotion] for emotion in self.negative_emotions)
            
            return {
                'emotions': emotion_dict,  # All emotions with raw probabilities
                'dominant_emotion': dominant_emotion,
                'negative_affect_score': float(negative_affect_score),  # Raw score
                'is_stressed': False,  # Removed threshold
                'face_detected': True
            }
        except Exception as e:
            print(f"Error in FER emotion detection: {e}")
            return self._default_emotion_result()
    
    def detect_emotion_vit(self, frame: np.ndarray) -> Dict[str, float]:
        """Detect emotion using Vision Transformer model"""
        if self.vit_model is None or self.vit_processor is None:
            return self._default_emotion_result()
        
        try:
            import torch
            from PIL import Image
            
            # Detect faces
            faces = []
            face_detected = False
            if self.face_detector is not None:
                faces = self.detect_faces(frame)
                face_detected = len(faces) > 0
            
            # Process face region with better preprocessing
            if faces:
                # Process detected face (most accurate)
                face_box = faces[0]
                face_roi = self.extract_face_region(frame, face_box)
                if face_roi is None or face_roi.size == 0:
                    face_roi = frame  # Fallback to full frame
            else:
                # No face detected - try center crop (where face is likely to be)
                h, w = frame.shape[:2]
                center_x, center_y = w // 2, h // 2
                crop_size = min(w, h) // 2
                x1 = max(0, center_x - crop_size // 2)
                y1 = max(0, center_y - crop_size // 2)
                x2 = min(w, center_x + crop_size // 2)
                y2 = min(h, center_y + crop_size // 2)
                face_roi = frame[y1:y2, x1:x2]
                
                # If crop is too small, use full frame
                if face_roi.size == 0 or face_roi.shape[0] < 32 or face_roi.shape[1] < 32:
                    face_roi = frame
            
            # Convert to RGB and enhance image for better emotion detection
            face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)

            # Optimized preprocessing: CLAHE + brightness normalization only
            # (Removed sharpening and edge detection to improve performance)

            # 1. CLAHE (Contrast Limited Adaptive Histogram Equalization) for better contrast
            lab = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            enhanced = cv2.merge([l, a, b])
            face_rgb = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)

            # 2. Normalize brightness to reduce neutral bias from lighting
            face_rgb = face_rgb.astype(np.float32)
            mean_brightness = np.mean(face_rgb)
            target_brightness = 128.0
            if mean_brightness > 0:
                face_rgb = face_rgb * (target_brightness / mean_brightness)
            face_rgb = np.clip(face_rgb, 0, 255).astype(np.uint8)
            
            # Convert to PIL Image
            pil_image = Image.fromarray(face_rgb)
            
            # Resize to model's expected size (ViT models typically expect 224x224)
            # The processor will handle this, but we ensure it's a reasonable size
            if pil_image.size[0] < 32 or pil_image.size[1] < 32:
                # Too small, resize up
                pil_image = pil_image.resize((224, 224), Image.Resampling.LANCZOS)
            elif pil_image.size[0] > 512 or pil_image.size[1] > 512:
                # Too large, resize down
                pil_image.thumbnail((512, 512), Image.Resampling.LANCZOS)
            
            # Preprocess image (processor handles resizing to model input size)
            inputs = self.vit_processor(pil_image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Predict
            with torch.no_grad():
                outputs = self.vit_model(**inputs)
                logits = outputs.logits
                
                # Apply temperature scaling to reduce neutral bias and improve emotion diversity
                # Lower temperature (0.6-0.8) makes the model more confident in non-neutral predictions
                # This helps detect subtle emotions like angry, sad, and fear
                scaled_logits = logits / self.temperature_scaling
                probabilities = torch.nn.functional.softmax(scaled_logits, dim=-1)
            
            # Get probabilities with temperature scaling applied
            probs = probabilities[0].cpu().numpy()
            
            # Map to emotion labels (ViT model outputs in specific order)
            id2label = self.vit_model.config.id2label
            
            # Label mapping from model labels to our format
            label_map = {
                'anger': 'angry',
                'happy': 'happy',
                'neutral': 'neutral',
                'sad': 'sad',
                'surprise': 'surprise',
                'fear': 'fear',
                'disgust': 'disgust'
            }
            
            # Create emotion dictionary - use raw model probabilities directly
            emotion_dict = {emotion: 0.0 for emotion in self.emotions}
            
            for idx, prob in enumerate(probs):
                label = id2label.get(idx, f"class_{idx}")
                label_lower = label.lower()
                
                # Map model label to our format
                mapped_label = label_map.get(label_lower, None)
                if mapped_label and mapped_label in self.emotions:
                    # Use raw probability directly - no thresholds, no filtering
                    emotion_dict[mapped_label] = float(prob)
            
            # Ensure all emotions are present (even if 0.0)
            for emotion in self.emotions:
                if emotion not in emotion_dict:
                    emotion_dict[emotion] = 0.0
            
            # Optional: Reduce neutral bias and improve emotion diversity
            # The model tends to be neutral-heavy, so we apply a de-biasing step (if enabled)
            if self.enable_neutral_bias_reduction:
                neutral_prob = emotion_dict.get('neutral', 0.0)

                # If neutral is too dominant, reduce it and redistribute to other emotions
                if neutral_prob > self.neutral_bias_threshold:
                    # Reduce neutral probability
                    emotion_dict['neutral'] = max(0.0, neutral_prob - self.neutral_bias_reduction_amount)

                    # Redistribute the reduced probability to other emotions based on their current values
                    # This ensures underrepresented emotions (angry, sad, fear) get boosted
                    total_other = sum(v for k, v in emotion_dict.items() if k != 'neutral')
                    if total_other > 0:
                        redistribution_factor = self.neutral_bias_reduction_amount / total_other
                        for emotion in self.emotions:
                            if emotion != 'neutral':
                                emotion_dict[emotion] *= (1.0 + redistribution_factor)

                    # Renormalize to ensure probabilities sum to 1.0
                    total = sum(emotion_dict.values())
                    if total > 0:
                        emotion_dict = {k: v / total for k, v in emotion_dict.items()}
            
            # Find dominant emotion (highest probability, no threshold)
            dominant_emotion = max(emotion_dict.items(), key=lambda x: x[1])[0]
            
            # Calculate negative affect score (sum of probabilities, no threshold)
            negative_affect_score = sum(emotion_dict[emotion] for emotion in self.negative_emotions)
            
            # Return raw results - no threshold-based filtering
            return {
                'emotions': emotion_dict,
                'dominant_emotion': dominant_emotion,
                'negative_affect_score': float(negative_affect_score),
                'is_stressed': False,  # Removed threshold - use negative_affect_score directly
                'face_detected': face_detected
            }
        except Exception as e:
            print(f"Error in ViT emotion detection: {e}")
            import traceback
            traceback.print_exc()
            return self._default_emotion_result()
    
    def detect_emotion_keras(self, frame: np.ndarray) -> Dict[str, float]:
        """Detect emotion using custom Keras model"""
        if self.keras_model is None or self.face_detector is None:
            return self._default_emotion_result()
        
        # Detect faces
        faces = self.detect_faces(frame)
        
        if not faces:
            return self._default_emotion_result()
        
        # Process first face
        face_box = faces[0]
        face_roi = self.extract_face_region(frame, face_box)
        
        if face_roi is None:
            return self._default_emotion_result()
        
        # Preprocess
        processed_face = self.preprocess_face(face_roi)
        
        # Predict
        try:
            predictions = self.keras_model.predict(processed_face, verbose=0)
            probabilities = predictions[0] if len(predictions.shape) > 1 else predictions
            
            # Create emotion dictionary
            emotion_dict = {}
            for i, emotion in enumerate(self.emotions):
                if i < len(probabilities):
                    emotion_dict[emotion] = float(probabilities[i])
                else:
                    emotion_dict[emotion] = 0.0
            
            # Normalize probabilities
            total = sum(emotion_dict.values())
            if total > 0:
                emotion_dict = {k: v / total for k, v in emotion_dict.items()}
            
            # Find dominant emotion
            dominant_emotion = max(emotion_dict.items(), key=lambda x: x[1])[0]
            
            # Calculate negative affect score (raw sum, no threshold)
            negative_affect_score = sum(emotion_dict[emotion] for emotion in self.negative_emotions)
            
            return {
                'emotions': emotion_dict,  # All emotions with raw probabilities
                'dominant_emotion': dominant_emotion,
                'negative_affect_score': float(negative_affect_score),  # Raw score
                'is_stressed': False,  # Removed threshold - use negative_affect_score directly
                'face_detected': True
            }
        except Exception as e:
            print(f"Error in Keras emotion prediction: {e}")
            return self._default_emotion_result()
    
    def _default_emotion_result(self) -> Dict[str, float]:
        """Return default emotion result when no face/model is available"""
        return {
            'emotions': {emotion: 0.0 for emotion in self.emotions},
            'dominant_emotion': 'neutral',
            'negative_affect_score': 0.0,
            'is_stressed': False,
            'face_detected': False
        }
    
    def detect_emotion(self, frame: np.ndarray) -> Dict[str, float]:
        """
        Detect emotion from video frame using deep learning
        
        Args:
            frame: Input video frame (BGR format)
            
        Returns:
            Dictionary with emotion probabilities and detected emotion
        """
        # Initialize if not done
        if not self.initialized:
            self.initialize()
        
        # Use ViT model if available (best accuracy - Vision Transformer)
        if self.vit_model is not None:
            return self.detect_emotion_vit(frame)
        
        # Use FER library if available
        if self.fer_model is not None:
            return self.detect_emotion_fer(frame)
        
        # Use custom Keras model if available
        if self.keras_model is not None:
            return self.detect_emotion_keras(frame)
        
        # No model available - return default
        return self._default_emotion_result()
    
    def is_frustrated(self, emotion_result: Dict[str, float]) -> bool:
        """
        Check if user shows signs of frustration
        Uses raw emotion probabilities - no thresholds
        
        Args:
            emotion_result: Result from detect_emotion()
            
        Returns:
            True if frustrated, False otherwise
        """
        # Use raw emotion probabilities - check if angry or sad has high probability
        emotions = emotion_result.get('emotions', {})
        angry_prob = emotions.get('angry', 0.0)
        sad_prob = emotions.get('sad', 0.0)
        
        # Return True if either emotion has significant probability (no fixed threshold)
        return angry_prob > 0.3 or sad_prob > 0.3 or \
               emotion_result.get('dominant_emotion') in ['angry', 'sad']
