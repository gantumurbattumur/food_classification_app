# app.py
import os

# --- Set env vars BEFORE importing tensorflow (helps avoid some macOS threading issues) ---
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"   # reduce TF logging
# Optionally force CPU if GPU causes problems:
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# --- Now import TF and other libs ---
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import numpy as np
import gradio as gr
import traceback

# --- Config ---
MODEL_PATH = "food_classification_model.keras" 
CLASS_NAMES = ['breakfast', 'dessert', 'meat', 'rice', 'flour', 'other_main']  
IMG_SIZE = (128, 128)

# --- Load model once, safely ---
def load_model_safe(path):
    try:
        model = keras.models.load_model(path)
        return model
    except Exception as e:
        print("Error loading model:", e)
        traceback.print_exc()
        raise

print("Loading model (this can take a few seconds)...")
model = load_model_safe(MODEL_PATH)
print("Model loaded.")

# --- Prediction function ---
def predict_pil_image(img: Image.Image):
    try:
        # Ensure RGB
        if img.mode != "RGB":
            img = img.convert("RGB")
        img = img.resize(IMG_SIZE)
        arr = np.array(img).astype(np.float32) / 255.0
        arr = np.expand_dims(arr, axis=0)
        preds = model.predict(arr)[0]  # (num_classes,)
        # Build dict of label -> probability
        out = {CLASS_NAMES[i]: float(preds[i]) for i in range(len(CLASS_NAMES))}
        # Also provide sorted list for display
        sorted_list = sorted(out.items(), key=lambda x: x[1], reverse=True)
        return out, sorted_list
    except Exception as err:
        print("Prediction error:", err)
        traceback.print_exc()
        return {c: 0.0 for c in CLASS_NAMES}, []

# --- Gradio UI ---
with gr.Blocks(title="Food Classifier") as demo:
    gr.Markdown("## Food Classifier (upload an image)")
    with gr.Row():
        inp = gr.Image(type="pil", label="Upload image")
        out_label = gr.Label(label="Predicted probabilities")
    prob_table = gr.Dataframe(headers=["label", "probability"], interactive=False)
    btn = gr.Button("Predict")

    def run(img):
        probs, ranked = predict_pil_image(img)
        # prepare dataframe rows
        rows = [[lbl, f"{prob*100:.2f}%"] for lbl, prob in ranked]
        return probs, rows

    btn.click(fn=run, inputs=inp, outputs=[out_label, prob_table])

# --- Launch (use share=False for local only) ---
if __name__ == "__main__":
    # show some env info
    print("TF version:", tf.__version__)
    print("Num GPUs:", len(tf.config.list_physical_devices("GPU")))
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
