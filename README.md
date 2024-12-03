### Facial-Emotion-Recognition-with-XAI

Real-time emotion detection with explainable AI capabilities. This project provides a robust system for recognizing and analyzing facial emotions in real-time with integrated explainable AI (XAI) features. It offers detailed visualizations, confidence scoring, and interpretability, making it a valuable tool for psychological research, clinical applications, and human-computer interaction.

![app gif](https://github.com/user-attachments/assets/8cbab127-8f1b-4d42-afeb-e19980ea4779)

---

## Features
- **Real-time Facial Emotion Detection**: Analyze emotions such as Happy, Sad, Neutral, and more.
- **Explainable AI (XAI) Components**:
  - Gradient-based attention maps.
  - SHAP-based feature importance visualizations.
- **Multi-Region Facial Analysis**: Confidence scoring for different facial areas.
- **Interactive Visualizations**: Displays emotion distributions and explanations of model decisions.
- **Streamlit Interface**: User-friendly web app for real-time image and video analysis.

---

## Installation

### Prerequisites
- Python 3.8 or later.
- A webcam or compatible camera device (optional for real-time analysis).

### Dependencies
Install dependencies using pip:
```bash
pip install -r requirements.txt
```


### Required Packages
- TensorFlow
- OpenCV
- Streamlit
- SHAP
- LIME
- Hugging Face Transformers
- NumPy, Pandas, Matplotlib, Seaborn

---

## Usage

### Basic Startup
Run the application using Streamlit:
```bash
streamlit run app.py
```

### Configuration
Use the Streamlit sidebar to configure options:
- Camera selection (for live analysis).
- Confidence threshold adjustment.
- Enable/disable visualization features.

### Running the Jupyter Notebook
Open the `Facial_Emotion_Recognition_with_XAI.ipynb` notebook for step-by-step model training, evaluation, and explainability workflows.

---

## Technical Details

### Architecture
The system is divided into three main components:
1. **Camera Management System**:
   - Handles webcam or video feed initialization.
   - Includes error recovery and resource cleanup.

2. **Emotion Detection Engine**:
   - Uses a pre-trained deep learning model for emotion classification.
   - Provides multi-region facial analysis with confidence scoring.

3. **XAI Visualization Module**:
   - Generates SHAP values for feature importance.
   - Provides gradient-based attention maps for visual explainability.

### Explainability (XAI)
- SHAP explains feature-level importance for each prediction.
- Attention maps visualize region-specific contributions to the final emotion classification.

---

## Clinical Applications
This tool is designed for practical applications in psychological research and therapy:
- **Research**:
  - Track emotion patterns across different scenarios.
  - Monitor response to stimuli for experimental validation.
- **Therapeutic Settings**:
  - Provide real-time emotion feedback during sessions.
  - Monitor treatment response and patient engagement.

---

## Project Structure
```plaintext
Facial-Emotion-Recognition-with-XAI/
│
├── Facial_Emotion_Recognition_with_XAI.ipynb  # Jupyter Notebook for model training and XAI
├── app.py                                     # Streamlit app for real-time use
├── fer_model.h5                               # Pre-trained model weights
├── requirements.txt                           # Dependency list
├── LICENSE                                    # Project license (GPL-3.0)
└── README.md                                  # Project documentation
```

---

## Future Enhancements
- Support for multi-person emotion detection.
- Enhanced privacy controls for sensitive use cases.
- Integration with advanced emotion detection models.
- Calibration for cross-cultural emotion recognition.
- Expanded clinical reporting features.

---

## Troubleshooting
### Common Issues
1. **Camera Detection**: Ensure the webcam is properly connected and accessible.
2. **Model Loading Errors**: Verify that `fer_model.h5` is in the correct directory.
3. **Performance**: Use a GPU for faster processing if performance is slow.

---

## References
- [Facial Emotion Recognition GitHub Repository](https://github.com/SajalSinha/Facial-Emotion-Recognition)
- [Facial Emotion Recognition with XAI](https://github.com/aiyufan3/Facial-Emotion-Recognition-with-XAI)
- Smith, D., Johnson, R., & Lee, T. (2023). *Advancing facial expression recognition*. Scientific Reports, 13(1). [https://doi.org/10.1038/s41598-023-35446-4](https://doi.org/10.1038/s41598-023-35446-4)
- [Papers with Code: Facial Expression Recognition](https://paperswithcode.com/task/facial-expression-recognition)

---

## License
This project is licensed under the [GPL-3.0 License](LICENSE).
