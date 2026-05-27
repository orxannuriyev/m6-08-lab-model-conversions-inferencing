import numpy as np, onnxruntime as ort
from PIL import Image

class FlowerClassifier:
    def __init__(self, onnx_path):
        # Load ONNX model once into a runtime session
        self.session = ort.InferenceSession(onnx_path)
        
        # ImageNet mean and std for normalization
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
        self.std  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)

    def preprocess(self, image_path):
        # Open image, convert to RGB, resize to 232x232
        img = Image.open(image_path).convert("RGB").resize((232, 232))
        
        # Center crop to 224x224
        left = (232 - 224) // 2
        img = img.crop((left, left, left + 224, left + 224))
        
        # Convert to numpy array, transpose to (C,H,W), add batch dimension
        x = np.asarray(img, dtype=np.float32).transpose(2, 0, 1)[None] / 255.0
        
        # Normalize with ImageNet mean/std
        return ((x - self.mean) / self.std).astype(np.float32)

    def predict(self, image_path, k=3):
        # Preprocess input image
        x = self.preprocess(image_path)
        
        # Run inference with ONNX Runtime
        logits = self.session.run(None, {"input": x})[0][0]
        
        # Convert logits to probabilities (softmax)
        probs = np.exp(logits - logits.max())
        probs /= probs.sum()
        
        # Get top-k predictions
        top = np.argsort(probs)[::-1][:k]
        return [(int(i), float(probs[i])) for i in top]
