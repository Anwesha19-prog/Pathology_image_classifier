# Breast Cancer Histopathology Image Classification

A deep learning project for classifying breast cancer histopathology images as benign or malignant using transfer learning with fastai.

## 🎯 Project Overview

This project demonstrates the application of deep learning to computational pathology, specifically for breast cancer diagnosis from histopathology images. Using the BreakHis dataset and fastai's powerful vision models, we achieve high accuracy in distinguishing between benign and malignant breast tissue samples.

## 🔬 Medical Context

Breast cancer is one of the most common cancers worldwide. Histopathological examination of tissue samples is the gold standard for diagnosis. This project aims to assist pathologists by providing an automated classification tool that can:
- Reduce diagnostic time
- Provide a second opinion
- Help in educational settings
- Support research in digital pathology

## 📊 Dataset

**BreakHis (Breast Cancer Histopathological Database)**
- Source: [BreakHis Dataset](https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/)
- Contains microscopic images of breast tumor tissue
- Two classes: Benign and Malignant
- Multiple magnification factors: 40X, 100X, 200X, 400X
- For this project, we use 400X magnification for consistency

## 🛠️ Technical Stack

- **Python 3.8+**
- **fastai** - Deep learning library built on PyTorch
- **Gradio** - Web interface for model deployment
- **PIL/OpenCV** - Image processing
- **Matplotlib/Seaborn** - Visualization

## 📁 Project Structure

```
breast-cancer-classifier/
│
├── data/
│   ├── train/
│   │   ├── benign/
│   │   └── malignant/
│   └── valid/
│       ├── benign/
│       └── malignant/
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   └── 02_model_training.ipynb
│
├── models/
│   └── breast_cancer_classifier.pkl
│
├── app.py                      # Gradio web interface
├── requirements.txt
├── README.md
└── results/
    └── visualizations/
```

## 🚀 Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/breast-cancer-classifier.git
cd breast-cancer-classifier
```

2. **Create a virtual environment (recommended)**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download the dataset**
- Visit [BreakHis Dataset](https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/)
- Download and extract to the `data/` folder
- Organize images into train/valid folders with benign/malignant subfolders

## 💻 Usage

### Training the Model

Run the training notebook:
```bash
jupyter notebook notebooks/02_model_training.ipynb
```

Or train directly with Python:
```bash
python train.py
```

### Running the Web Interface

Launch the Gradio app:
```bash
python app.py
```

Then open your browser to the provided local URL (typically `http://127.0.0.1:7860`)

## 📈 Results

### Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | ~90-95% |
| Precision | ~92% |
| Recall | ~91% |
| F1-Score | ~91% |

*Note: Actual results will be updated after training*

### Key Features

- **Transfer Learning**: Uses ResNet34/ResNet50 pre-trained on ImageNet
- **Data Augmentation**: Automatic augmentation to improve generalization
- **Learning Rate Finder**: Optimal learning rate selection
- **Grad-CAM Visualization**: Shows which regions the model focuses on
- **Interactive Demo**: Easy-to-use web interface for testing

## 🔍 Model Interpretation

The project includes Grad-CAM (Gradient-weighted Class Activation Mapping) visualizations that highlight which parts of the histopathology image the model considers most important for its prediction. This is crucial for:
- Building trust with medical professionals
- Understanding model behavior
- Identifying potential biases
- Educational purposes

## 🎓 Learning Outcomes

This project demonstrates:
- ✅ Understanding of medical imaging and pathology
- ✅ Deep learning for image classification
- ✅ Transfer learning techniques
- ✅ Model evaluation with medical metrics
- ✅ Model interpretability (Grad-CAM)
- ✅ Web deployment of ML models
- ✅ Best practices in ML project organization

## 🔮 Future Improvements

- [ ] Multi-class classification (different cancer subtypes)
- [ ] Multi-magnification ensemble models
- [ ] Nuclei segmentation and feature extraction
- [ ] Attention mechanisms for better interpretability
- [ ] Integration with whole slide imaging (WSI)
- [ ] Deployment to cloud platforms (Hugging Face Spaces, AWS)
- [ ] Mobile app development

## 📚 References

1. Spanhol, F. A., et al. "A Dataset for Breast Cancer Histopathological Image Classification." IEEE TBME, 2016.
2. Howard, J., & Gugger, S. "Deep Learning for Coders with fastai and PyTorch." O'Reilly, 2020.
3. Selvaraju, R. R., et al. "Grad-CAM: Visual Explanations from Deep Networks." ICCV, 2017.

## 📝 License

This project is for educational and portfolio purposes. Please ensure compliance with the BreakHis dataset license for any commercial use.

## 👤 Author

**Your Name**
- GitHub: [@yourusername](https://github.com/yourusername)
- LinkedIn: [Your LinkedIn](https://linkedin.com/in/yourprofile)
- Email: your.email@example.com

## 🙏 Acknowledgments

- BreakHis dataset creators at Federal University of Paraná
- fastai community and Jeremy Howard
- The computational pathology research community

---

*This project was created as a portfolio demonstration for computational pathology research positions.*
