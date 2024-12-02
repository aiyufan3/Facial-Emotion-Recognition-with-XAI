import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import time
import os
from pathlib import Path
import pandas as pd
from collections import deque
import threading
from typing import Dict, List, Tuple, Optional
import logging
import plotly.express as px
import plotly.graph_objects as go
import tensorflow as tf
import subprocess
import platform

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# Add tensorflow import with error handling
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import load_model
except ImportError:
    raise ImportError(
        "TensorFlow is required but not installed. "
        "Please install tensorflow using: pip install tensorflow"
    )

# Set page config with improved styling
st.set_page_config(
    page_title="Facial Emotion Recognition XAI",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .emotion-chart {
        background-color: #f0f2f6;
        border-radius: 5px;
        padding: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state with type hints
if 'model' not in st.session_state:
    st.session_state.model: Optional[object] = None
if 'camera_active' not in st.session_state:
    st.session_state.camera_active: bool = False
if 'current_results' not in st.session_state:
    st.session_state.current_results: Optional[List[Dict]] = None
if 'camera_initialized' not in st.session_state:
    st.session_state.camera_initialized: bool = False
if 'camera_error' not in st.session_state:
    st.session_state.camera_error: Optional[str] = None
if 'fps_buffer' not in st.session_state:
    st.session_state.fps_buffer: deque = deque(maxlen=30)


class CameraManager:
    """Enhanced camera management with improved error handling and resource management"""

    def __init__(self):
        self.available_cameras: List[Dict] = []
        self.current_camera: Optional[cv2.VideoCapture] = None
        self.lock: threading.Lock = threading.Lock()
        self.detect_cameras()
        logger.info("CameraManager initialized")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release_camera()
        logger.info("Camera resources released")

    def detect_cameras(self) -> None:
        """Detect available cameras with enhanced error handling"""
        self.available_cameras = []
        tested_indices = set()

        # Test common webcam indices
        priority_indices = [0, 1]  # Most common built-in webcam indices
        for idx in priority_indices:
            if self.test_camera(idx):
                tested_indices.add(idx)

        # Test additional indices
        for idx in range(10):
            if idx not in tested_indices and self.test_camera(idx):
                tested_indices.add(idx)

        logger.info(f"Detected {len(self.available_cameras)} cameras")

    def detect_camera_name(self, index: int, cap: cv2.VideoCapture) -> str:
        """Detect actual camera name based on platform"""
        try:
            # Try to get device name from OpenCV properties first
            # Set backend before getting properties
            if platform.system().lower() == 'windows':
                cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
                # Get camera name through DirectShow
                name = cap.get(cv2.CAP_PROP_GSTREAMER)
                if isinstance(name, str) and name:
                    return name.strip()

            elif platform.system().lower() == 'linux':
                # Try multiple device info files on Linux
                for device_info in [f'/sys/class/video4linux/video{index}/name',
                                    f'/sys/class/video4linux/video{index}/model']:
                    if os.path.exists(device_info):
                        try:
                            with open(device_info, 'r') as f:
                                name = f.read().strip()
                                if name:
                                    return name
                        except:
                            pass

            elif platform.system().lower() == 'darwin':
                try:
                    # Use system_profiler with timeout and better parsing
                    output = subprocess.check_output(
                        ['system_profiler', 'SPCameraDataType'],
                        timeout=2  # 2 second timeout
                    ).decode('utf-8')

                    # More robust parsing of system_profiler output
                    cameras = []
                    for line in output.split('\n'):
                        if ':' in line and any(x in line.lower() for x in ['camera', 'facetime', 'webcam']):
                            name = line.split(':')[1].strip()
                            if name:
                                cameras.append(name)

                    if index < len(cameras):
                        return cameras[index]
                except (subprocess.TimeoutExpired, subprocess.SubprocessError):
                    pass

            # Fallback: Create descriptive name with device info
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))

            # Include more device info in fallback name
            if platform.system().lower() == 'windows':
                return f"USB Camera {index} ({width}x{height} @ {fps}fps)"
            elif platform.system().lower() == 'linux':
                return f"V4L2 Camera {index} ({width}x{height} @ {fps}fps)"
            elif platform.system().lower() == 'darwin':
                return f"macOS Camera {index} ({width}x{height} @ {fps}fps)"
            else:
                return f"Camera {index} ({width}x{height} @ {fps}fps)"

        except Exception as e:
            logger.warning(f"Failed to get camera name for index {index}: {e}")
            return f"Camera {index}"  # Basic fallback

    def test_camera(self, index: int) -> bool:
        """Test camera functionality with improved name detection"""
        try:
            cap = cv2.VideoCapture(index)
            if not cap.isOpened():
                return False

            # Test multiple frame captures for reliability
            successful_reads = 0
            for _ in range(3):
                ret, frame = cap.read()
                if ret and frame is not None and frame.size > 0:
                    successful_reads += 1

            if successful_reads >= 2:
                # Get camera properties and actual device name
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = int(cap.get(cv2.CAP_PROP_FPS))

                if width > 0 and height > 0:
                    name = self.detect_camera_name(index, cap)

                    camera_info = {
                        'index': index,
                        'name': name,
                        'resolution': f"{width}x{height}",
                        'fps': fps
                    }
                    self.available_cameras.append(camera_info)
                    logger.info(f"Successfully tested camera {index}: {name}")
                    return True

            return False

        except Exception as e:
            logger.error(f"Error testing camera {index}: {str(e)}")
            return False

        finally:
            if 'cap' in locals():
                cap.release()

    def initialize_camera(self, index: int) -> bool:
        """Initialize camera with enhanced error recovery"""
        with self.lock:
            if self.current_camera is not None:
                self.release_camera()

            try:
                cap = cv2.VideoCapture(index)
                if not cap.isOpened():
                    raise RuntimeError("Failed to open camera")

                # Configure camera with optimal settings
                optimal_settings = {
                    cv2.CAP_PROP_FRAME_WIDTH: 640,
                    cv2.CAP_PROP_FRAME_HEIGHT: 480,
                    cv2.CAP_PROP_FPS: 30,
                    cv2.CAP_PROP_BUFFERSIZE: 1
                }

                for prop, value in optimal_settings.items():
                    cap.set(prop, value)

                # Verify settings
                ret, frame = cap.read()
                if not ret or frame is None or frame.size == 0:
                    raise RuntimeError("Failed to capture test frame")

                self.current_camera = cap
                st.session_state.camera_initialized = True
                st.session_state.camera_error = None
                logger.info(f"Successfully initialized camera {index}")
                return True

            except Exception as e:
                error_msg = f"Camera initialization error: {str(e)}"
                logger.error(error_msg)
                st.session_state.camera_error = error_msg
                st.session_state.camera_initialized = False
                if 'cap' in locals():
                    cap.release()
                return False

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Thread-safe frame capture with error handling"""
        with self.lock:
            try:
                if self.current_camera is not None:
                    ret, frame = self.current_camera.read()
                    if ret and frame is not None and frame.size > 0:
                        return True, frame
                    logger.warning("Invalid frame captured")
                return False, None
            except Exception as e:
                logger.error(f"Error reading frame: {str(e)}")
                return False, None

    def release_camera(self) -> None:
        """Safely release camera resources"""
        with self.lock:
            try:
                if self.current_camera is not None:
                    self.current_camera.release()
                    self.current_camera = None
                    time.sleep(0.5)  # Allow time for cleanup
                    st.session_state.camera_initialized = False
                    logger.info("Camera released successfully")
            except Exception as e:
                logger.error(f"Error releasing camera: {str(e)}")


class EmotionDetectorXAI:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        self.enable_gradients = False

        # Optimized thresholds based on validation data
        self.base_thresholds = {
            'Angry': 0.65,
            'Disgust': 0.65,
            'Fear': 0.65,
            'Happy': 0.5,
            'Sad': 0.65,
            'Surprise': 0.55,
            'Neutral': 0.50  # Adjusted Neutral threshold
        }

        # Enhanced smoothing with configurable parameters
        self.emotion_buffer = deque(maxlen=15)
        self.feature_buffer = deque(maxlen=5)

        # Configuration parameters
        self.config = {
            'confidence_threshold': 0.5,
            'smoothing_factor': 0.5,
            'edge_detection_threshold': 100,
            'feature_weight': 0.8
        }

        self.load_model()
        logger.info("EmotionDetectorXAI initialized")

    def load_model(self) -> None:
        """Load emotion recognition model with enhanced error handling"""
        if st.session_state.model is not None:
            self.model = st.session_state.model
            return

        try:
            model_path = os.path.join(os.path.dirname(__file__), 'fer_model.h5')
            if not os.path.exists(model_path):
                raise FileNotFoundError("Model file not found")

            self.model = load_model(model_path)
            st.session_state.model = self.model
            logger.info("Model loaded successfully")

        except Exception as e:
            error_msg = f"Error loading model: {str(e)}"
            logger.error(error_msg)
            st.error(error_msg)
            self.model = None

    def smooth_predictions(self, predictions: np.ndarray) -> np.ndarray:
        """Improved smoothing to respond better to dynamic expressions."""
        self.emotion_buffer.append(predictions)
        if len(self.emotion_buffer) < 2:
            return predictions

        # Exponential smoothing with lower inertia for Surprise
        weights = np.exp(np.linspace(-1, 0, len(self.emotion_buffer)))
        weights /= np.sum(weights)

        smoothed = np.zeros_like(predictions)
        for i, pred in enumerate(self.emotion_buffer):
            smoothed += pred * weights[i]

        # Boost Surprise slightly if detected
        surprise_idx = self.emotions.index('Surprise')
        smoothed[surprise_idx] *= 1.1

        return smoothed

    def analyze_face_regions(self, face_img: np.ndarray) -> Dict:
        h, w = face_img.shape[:2]

        # Ensure consistent dimensions by resizing regions
        regions = {
            'upper': cv2.resize(face_img[0:int(h * 0.35), :], (84, 58)),  # Fixed dimensions
            'middle': cv2.resize(face_img[int(h * 0.35):int(h * 0.65), :], (84, 50)),
            'lower': cv2.resize(face_img[int(h * 0.65):h, :], (84, 59))
        }

        analysis = {}
        for region_name, region in regions.items():
            try:
                if region.size == 0:
                    raise ValueError(f"Empty region: {region_name}")

                gray_region = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)

                # Calculate metrics with dimension checks
                analysis[f'{region_name}_intensity'] = float(np.mean(gray_region))
                analysis[f'{region_name}_variation'] = float(np.std(gray_region))

                edges = cv2.Canny(gray_region, 100, 200)
                analysis[f'{region_name}_edges'] = float(np.mean(edges))

                # Ensure shapes match before subtraction
                left = gray_region[:, :gray_region.shape[1] // 2]
                right = np.fliplr(gray_region[:, gray_region.shape[1] // 2:])
                if left.shape == right.shape:
                    analysis[f'{region_name}_symmetry'] = 1 - float(np.mean(np.abs(left - right))) / 255
                else:
                    analysis[f'{region_name}_symmetry'] = 0.0

                analysis[f'{region_name}_texture'] = float(
                    np.std(gray_region - cv2.GaussianBlur(gray_region, (5, 5), 0))
                )

            except Exception as e:
                logger.error(f"Error analyzing {region_name} region: {str(e)}")
                analysis[f'{region_name}_error'] = str(e)

        return analysis

    def adjust_predictions(self, predictions: np.ndarray, features: Dict) -> np.ndarray:
        """Improved prediction adjustment logic to avoid over-calibration."""
        adjusted = predictions.copy()
        try:
            # Apply base thresholds with smooth transition
            for i, emotion in enumerate(self.emotions):
                threshold = self.base_thresholds[emotion]
                if adjusted[i] < threshold:
                    factor = np.clip(adjusted[i] / threshold, 0.5, 1.0)  # Adjusted min factor
                    adjusted[i] *= factor

            # Neutral vs. Surprise/Sad balancing using features
            neutral_idx = self.emotions.index('Neutral')
            surprise_idx = self.emotions.index('Surprise')

            symmetry_score = np.mean([features.get(f'{region}_symmetry', 0) for region in ['upper', 'middle', 'lower']])
            variation_score = np.mean(
                [features.get(f'{region}_variation', 0) for region in ['upper', 'middle', 'lower']])

            # Surprise calibration
            if symmetry_score < 0.7 and variation_score > 30:
                adjusted[surprise_idx] *= 1.3  # Boost Surprise for high variation
                adjusted[neutral_idx] *= 0.8

            # Normalize predictions
            total = np.sum(adjusted)
            if total > 0:
                adjusted = adjusted / total

        except Exception as e:
            logger.error(f"Error adjusting predictions: {str(e)}")
            return predictions

        return adjusted

    def process_frame(self, frame: np.ndarray) -> List[Dict]:
        if frame is None or self.model is None:
            return []

        try:
            results = []
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=4,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )

            for (x, y, w, h) in faces:
                try:
                    # Extract and preprocess face for emotion detection
                    face_roi = frame[y:y + h, x:x + w]
                    face_gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
                    face_gray = cv2.resize(face_gray, (48, 48))  # Model expects 48x48
                    face_gray = face_gray.astype("float") / 255.0
                    face_gray = np.expand_dims(face_gray, axis=[0, -1])

                    # Get emotion predictions
                    predictions = self.model.predict(face_gray, verbose=0)[0]

                    # Get region features
                    features = self.analyze_face_regions(face_roi)

                    # Calculate detection confidence
                    detection_confidence = self._calculate_detection_confidence(face_roi)

                    # Smooth and calibrate predictions
                    smoothed = self.smooth_predictions(predictions)
                    calibrated = self.adjust_predictions(smoothed, features)

                    # Get attention weights
                    attention_weights = self._calculate_attention_weights(cv2.resize(face_gray[0, :, :, 0], (48, 48)))

                    # Create result dictionary with all required information
                    result = {
                        'bbox': (x, y, w, h),
                        'raw_emotions': predictions,
                        'calibrated_emotions': calibrated,
                        'features': features,
                        'detection_confidence': detection_confidence,
                        'attention_weights': attention_weights,
                        'inference_time': 0,  # Add proper timing if needed
                    }

                    results.append(result)

                except Exception as e:
                    logger.error(f"Error processing face: {str(e)}")
                    continue

            return results

        except Exception as e:
            logger.error(f"Error in frame processing: {str(e)}")
            return []
    def _calculate_detection_confidence(self, face_roi: np.ndarray) -> float:
        """Calculate confidence score for face detection"""
        try:
            # Multiple metrics for confidence calculation
            metrics = {
                'size': np.prod(face_roi.shape[:2]),
                'contrast': np.std(cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)),
                'blur': cv2.Laplacian(cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
            }

            # Normalize metrics
            normalized_metrics = {
                'size': min(1.0, metrics['size'] / (300 * 300)),
                'contrast': min(1.0, metrics['contrast'] / 80),
                'blur': min(1.0, metrics['blur'] / 500)
            }

            # Weighted combination
            weights = {'size': 0.3, 'contrast': 0.3, 'blur': 0.4}
            confidence = sum(normalized_metrics[k] * weights[k] for k in weights)

            return float(np.clip(confidence, 0.0, 1.0))

        except Exception as e:
            logger.error(f"Error calculating detection confidence: {str(e)}")
            return 0.5

    def _extract_enhanced_features(self, face_roi: np.ndarray) -> Dict:
        """Extract additional features for XAI"""
        try:
            gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            features = {}

            # Facial landmarks (if available)
            if hasattr(self, 'landmark_detector'):
                try:
                    landmarks = self.landmark_detector(gray)
                    features['landmarks'] = landmarks
                except:
                    pass

            # Texture analysis
            try:
                features['texture'] = {
                    'contrast': float(np.std(gray)),
                    'entropy': float(self._calculate_entropy(gray))
                }
            except:
                pass

            # Edge analysis
            try:
                edges = cv2.Canny(gray, 100, 200)
                features['edges'] = {
                    'density': float(np.mean(edges) / 255),
                    'strength': float(np.std(edges) / 255)
                }
            except:
                pass

            return features

        except Exception as e:
            logger.error(f"Error extracting enhanced features: {str(e)}")
            return {}

    def _calculate_attention_weights(self, face_gray: np.ndarray) -> Dict:
        """Calculate attention weights for different facial regions"""
        try:
            h, w = face_gray.shape
            regions = {
                'eyes': face_gray[int(0.2 * h):int(0.5 * h), :],
                'nose': face_gray[int(0.3 * h):int(0.7 * h), int(0.3 * w):int(0.7 * w)],
                'mouth': face_gray[int(0.6 * h):int(0.9 * h), :]
            }

            weights = {}
            for region_name, region in regions.items():
                # Calculate weight based on intensity variation
                weight = float(np.std(region))
                # Normalize
                weights[region_name] = weight

            # Normalize weights to sum to 1
            total = sum(weights.values())
            if total > 0:
                weights = {k: v / total for k, v in weights.items()}

            return weights

        except Exception as e:
            logger.error(f"Error calculating attention weights: {str(e)}")
            return {}

    def _calculate_entropy(self, gray: np.ndarray) -> float:
        """Calculate image entropy for texture analysis"""
        try:
            histogram = cv2.calcHist([gray], [0], None, [256], [0, 256])
            histogram = histogram.ravel() / histogram.sum()
            non_zero = histogram > 0
            return -np.sum(histogram[non_zero] * np.log2(histogram[non_zero]))
        except Exception as e:
            logger.error(f"Error calculating entropy: {str(e)}")
            return 0.0

    def _compute_gradients(self, face_gray: np.ndarray) -> Optional[np.ndarray]:
        """Compute gradients for XAI visualization with proper error handling"""
        if not hasattr(self, 'enable_gradients') or not self.enable_gradients:
            return None

        try:
            # Convert input for gradient computation
            input_tensor = np.expand_dims(face_gray, axis=[0, -1])
            input_tensor = tf.convert_to_tensor(input_tensor, dtype=tf.float32)

            # Use GradientTape for automatic differentiation
            with tf.GradientTape() as tape:
                tape.watch(input_tensor)
                predictions = self.model(input_tensor)
                top_pred_idx = tf.argmax(predictions[0])
                top_class_pred = predictions[:, top_pred_idx]

            # Compute gradients
            gradients = tape.gradient(top_class_pred, input_tensor)

            # Process gradients
            if gradients is not None:
                gradients = tf.math.abs(gradients)
                gradients = np.array(gradients[0, ..., 0])

                # Normalize gradients for visualization
                gradient_min = gradients.min()
                gradient_max = gradients.max()
                if gradient_max > gradient_min:
                    gradients = (gradients - gradient_min) / (gradient_max - gradient_min)
                else:
                    gradients = np.zeros_like(gradients)

                return gradients

            return None

        except (ImportError, AttributeError) as e:
            logger.warning(f"TensorFlow operations not available: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Error computing gradients: {str(e)}")
            return None


def main():
    st.title("Facial Emotion Recognition with XAI")

    # Initialize managers with better error handling
    try:
        camera_mgr = CameraManager()
        detector = EmotionDetectorXAI()
        emotions = detector.emotions
    except Exception as e:
        st.error(f"Failed to initialize system: {str(e)}")
        return

    # Enhanced sidebar controls - single camera selection
    # Camera selection with error handling
    available_cameras = [cam['name'] for cam in camera_mgr.available_cameras]
    if not available_cameras:
        st.error("No working cameras detected. Please check your camera connections.")
        return


    with st.sidebar:
        selected_camera = st.selectbox(
            "Select Camera",
            available_cameras,
            help="Choose which camera to use",
            key="camera_select"  # Added unique key
        )

        # Analysis settings
        st.header("Analysis Settings")
        confidence_threshold = st.slider(
            "Confidence Threshold",
            0.0, 1.0, 0.5,
            help="Minimum confidence level for emotion detection",
            key="conf_threshold"
        )

        show_features = st.checkbox(
            "Show Feature Analysis",
            True,
            help="Display facial feature analysis",
            key="show_features"
        )

        show_performance = st.checkbox(
            "Show Performance Metrics",
            False,
            help="Display FPS and processing times",
            key="show_performance"
        )

    try:
        camera_index = next(
            cam['index'] for cam in camera_mgr.available_cameras
            if cam['name'] == selected_camera
        )
    except Exception as e:
        st.error(f"Error selecting camera: {str(e)}")
        return

    # Main content area with improved layout
    col1, col2 = st.columns([2, 1])

    with col1:
        st.header("Live Feed")
        frame_placeholder = st.empty()
        status_placeholder = st.empty()
        fps_placeholder = st.empty()

        # Camera controls
        start_col, stop_col = st.columns(2)
        with start_col:
            start_button = st.button(
                "Start Camera",
                help="Begin emotion detection",
                key="start_camera"
            )
        with stop_col:
            stop_button = st.button(
                "Stop Camera",
                help="Stop emotion detection",
                key="stop_camera"
            )

    with col2:
        st.header("Analysis Results")
        emotion_chart = st.empty()
        features_chart = st.empty()
        metadata_container = st.empty()

    if start_button:
        if detector.model is None:
            st.error("Cannot start camera: Model not loaded")
            return

        st.session_state.camera_active = True

        # Initialize camera with status updates
        status_placeholder.info("Initializing camera...")
        if not camera_mgr.initialize_camera(camera_index):
            st.error(f"Failed to initialize camera: {st.session_state.camera_error}")
            st.session_state.camera_active = False
            return
        status_placeholder.success("Camera initialized successfully")

        try:
            last_time = time.time()
            frames_processed = 0
            fps_update_interval = 0.5

            while st.session_state.camera_active:
                ret, frame = camera_mgr.read_frame()
                if not ret or frame is None:
                    status_placeholder.warning("Camera connection lost. Attempting to reconnect...")
                    if not camera_mgr.initialize_camera(camera_index):
                        status_placeholder.error("Failed to reconnect to camera")
                        break
                    continue

                # Process frame and update results
                results = detector.process_frame(frame)
                st.session_state.current_results = results

                # Draw results
                annotated_frame = draw_results(
                    frame.copy(),
                    results,
                    confidence_threshold,
                    emotions,
                    show_features
                )

                # Update performance metrics
                current_time = time.time()
                frames_processed += 1
                if current_time - last_time >= fps_update_interval:
                    fps = frames_processed / (current_time - last_time)
                    if show_performance:
                        fps_placeholder.text(f"FPS: {fps:.1f}")
                    frames_processed = 0
                    last_time = current_time

                # Display frame
                frame_placeholder.image(
                    cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB),
                    caption="Live Feed with Emotion Detection",
                    use_container_width=True
                )

                if stop_button:
                    st.session_state.camera_active = False
                    break

                # Update analysis displays if results available
                if len(results) > 0:
                    result = results[0]

                    # Update emotion chart
                    emotions_df = pd.DataFrame({
                        'Emotion': emotions,
                        'Confidence': result['calibrated_emotions']
                    })
                    emotion_chart.plotly_chart(
                        create_emotion_chart(emotions_df),
                        use_container_width=True
                    )

                    if show_features:
                        # Update feature analysis
                        features_df = pd.DataFrame({
                            'Region': ['Upper', 'Middle', 'Lower'],
                            'Activity': [
                                result['features'].get('upper_variation', 0),
                                result['features'].get('middle_variation', 0),
                                result['features'].get('lower_variation', 0)
                            ],
                            'Symmetry': [
                                result['features'].get('upper_symmetry', 0),
                                result['features'].get('middle_symmetry', 0),
                                result['features'].get('lower_symmetry', 0)
                            ]
                        })
                        features_chart.plotly_chart(
                            create_features_chart(features_df),
                            use_container_width=True
                        )

                        # Update metadata
                        metadata = {
                            'Detection Confidence': f"{result.get('detection_confidence', 0):.2f}",
                            'Inference Time': f"{result.get('inference_time', 0) * 1000:.1f}ms",
                            'Face Size': f"{result['bbox'][2]}x{result['bbox'][3]}px"
                        }
                        metadata_container.json(metadata)

        except Exception as e:
            logger.error(f"Error in main loop: {str(e)}")
            st.error(f"An error occurred: {str(e)}")
            st.session_state.camera_active = False

        finally:
            camera_mgr.release_camera()
            status_placeholder.info("Camera stopped")

def draw_results(frame: np.ndarray, results: List[Dict],
                confidence_threshold: float, emotions: List[str],
                show_features: bool = True) -> np.ndarray:
    """Enhanced visualization of detection results"""
    try:
        # Visualization constants
        FONT = cv2.FONT_HERSHEY_SIMPLEX
        FONT_SCALE = 0.8
        THICKNESS = 2

        for result in results:
            x, y, w, h = result['bbox']
            emotion_probs = result['calibrated_emotions']

            # Enhanced confidence visualization
            detection_conf = result.get('detection_confidence', 0.5)
            box_color = (
                int(255 * (1 - detection_conf)),
                int(255 * detection_conf),
                0
            )

            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, THICKNESS)

            # Draw emotion label
            emotion_idx = np.argmax(emotion_probs)
            emotion = emotions[emotion_idx]
            confidence = emotion_probs[emotion_idx]

            if confidence >= confidence_threshold:
                label = f"{emotion}: {confidence:.2f}"

                # Enhanced label background
                (label_w, label_h), _ = cv2.getTextSize(
                    label,
                    FONT,
                    FONT_SCALE,
                    THICKNESS
                )

                cv2.rectangle(
                    frame,
                    (x, y - label_h - 10),
                    (x + label_w + 10, y),
                    box_color,
                    -1
                )

                cv2.putText(
                    frame,
                    label,
                    (x + 5, y - 5),
                    FONT,
                    FONT_SCALE,
                    (255, 255, 255),
                    THICKNESS
                )

            if show_features:
                # Draw region markers
                h_third = h // 3
                regions = [
                    ('Upper', y + h_third),
                    ('Middle', y + 2 * h_third),
                    ('Lower', y + h)
                ]

                for region_name, y_pos in regions:
                    cv2.line(
                        frame,
                        (x, y_pos),
                        (x + w, y_pos),
                        (255, 0, 0),
                        1
                    )

                    cv2.putText(
                        frame,
                        region_name,
                        (x + w + 5, y_pos),
                        FONT,
                        FONT_SCALE * 0.75,
                        (255, 0, 0),
                        1
                    )

                if 'attention_weights' in result:
                    weights = result['attention_weights']
                    for region, weight in weights.items():
                        y_pos = y + {
                            'eyes': h_third,
                            'nose': 2 * h_third,
                            'mouth': h
                        }[region]

                        radius = int(10 * weight)
                        cv2.circle(
                            frame,
                            (x + w + 20, y_pos),
                            radius,
                            (0, 255, 0),
                            -1
                        )

        return frame

    except Exception as e:
        logger.error(f"Error drawing results: {str(e)}")
        return frame


def create_emotion_chart(df):
    """Create interactive emotion distribution chart"""
    fig = px.bar(
        df,
        x='Emotion',
        y='Confidence',
        color='Confidence',
        color_continuous_scale='Blues',
        title='Emotion Distribution'
    )

    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=40, b=20),
        title_x=0.5
    )

    return fig


def create_features_chart(df):
    """Create interactive feature analysis chart"""
    fig = go.Figure()

    # Add activity bars
    fig.add_trace(
        go.Bar(
            name='Activity',
            x=df['Region'],
            y=df['Activity'],
            marker_color='rgb(55, 83, 109)'
        )
    )

    # Add symmetry line
    fig.add_trace(
        go.Scatter(
            name='Symmetry',
            x=df['Region'],
            y=df['Symmetry'],
            mode='lines+markers',
            line=dict(color='rgb(26, 118, 255)')
        )
    )

    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=40, b=20),
        title='Facial Region Analysis',
        title_x=0.5,
        barmode='group',
        showlegend=True
    )

    return fig


if __name__ == '__main__':
    main()