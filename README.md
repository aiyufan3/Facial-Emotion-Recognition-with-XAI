# Facial-Emotion-Recognition-with-XAI
### Real-time emotion detection with explainable AI capabilities

An advanced facial emotion recognition system that provides real-time emotion detection with explainable AI visualizations. The system analyzes facial expressions across multiple regions, provides confidence scoring, and offers detailed insights into its decision-making process, making it particularly valuable for psychological research and clinical applications.

![app gif](https://github.com/user-attachments/assets/8cbab127-8f1b-4d42-afeb-e19980ea4779)

## Features
- Real-time facial emotion detection
- Multi-region facial analysis with confidence scoring
- Interactive visualization of emotion distributions and facial features
- Explainable AI components for understanding model decisions
- Robust camera handling with automatic recovery
- Configurable analysis parameters
- Clinical-grade emotion tracking capabilities

## Installation

### Prerequisites
- Python 3.8+
- Webcam or compatible camera device

### Dependencies
```bash
pip install -r requirements.txt
```

Or using conda:
```bash
conda env create -f environment.yml
```

Required packages:
- tensorflow
- opencv-python
- streamlit
- plotly
- numpy
- pandas

## Usage

### Basic Startup
```bash
streamlit run app.py
```

### Configuration
The application provides several configuration options in the sidebar:
- Camera selection
- Confidence threshold adjustment
- Feature analysis toggles
- Performance metrics display

## Technical Details

### Architecture
The system consists of three main components:
1. Camera Management System
   - Robust device detection and initialization
   - Error recovery and resource management
   - Cross-platform compatibility

2. Emotion Detection Engine
   - Deep learning model for emotion classification
   - Multi-region facial analysis
   - Confidence scoring system

3. XAI Visualization Module
   - Real-time feature importance visualization
   - Region-specific attention mapping
   - Confidence metrics calculation

### XAI Components
The system provides multiple levels of explainability:
- Region-based facial analysis
- Attention weight visualization
- Feature importance mapping
- Confidence scoring mechanisms

## Clinical Applications

### Psychological Research
- Emotion pattern analysis
- Response monitoring
- Treatment effectiveness evaluation
- Patient progress tracking

### Therapeutic Settings
- Real-time emotion feedback
- Session recording and analysis
- Patient engagement monitoring
- Treatment response assessment

## Customization
The system can be extended through:
- Custom emotion detection models
- Additional facial feature analyzers
- Enhanced visualization components
- Clinical reporting modules

## Performance Optimization
- Efficient frame processing
- Memory management
- Resource cleanup
- Error recovery mechanisms

## Future Development
Planned enhancements include:
- Integration with more advanced emotion detection models
- Enhanced multi-person tracking
- Improved privacy features
- Extended clinical reporting capabilities
- Cross-cultural emotion recognition calibration

## Troubleshooting
Common issues and solutions:
- Camera detection problems
- Model loading errors
- Performance optimization
- System compatibility

