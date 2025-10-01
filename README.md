# 🍽️ Food Classification with TensorFlow

A simple deep learning project to classify food images into **5 categories**:
- Breakfast
- Dessert
- Meat
- Rice
- Flour-based foods

The model is trained on **personal food images** collected from my iPhone, converted from HEIC to JPEG format, making this a **personalized dataset** project.  

---

## 🚀 Demo
*(Screenshot or GIF goes here)*

---

## 📦 Installation

Clone the repository and set up a Python virtual environment:

```bash
git clone https://github.com/<your-username>/food_classification.git
cd food_classification

python -m venv venv
source venv/bin/activate  # On Mac/Linux
venv\Scripts\activate     # On Windows

pip install -r requirements.txt
```
---

## Usage

# Run the web app
`python app.py`

# Make a quick prediction
`python predict_food.py --img test_image.jpeg

## 📊 Model Performance

* Base model: MobileNetV2
* Input size: 128x128
* Current accuracy: ~83% on test set (140 total images)

## ⚠️ Model is still small and trained on limited data. Performance will improve with:

* More balanced dataset
* Data augmentation
* Hyperparameter tuning

## 🤝 Future Improvements

* Increase dataset size
* Improve accuracy with augmentation and hyperparameter tuning
* Deploy app to Hugging Face Spaces / Streamlit Cloud