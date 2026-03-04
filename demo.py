from tensorflow.keras.models import load_model
import numpy as np

model_path = r"D:\PROJECT\fruitvegapp_\fruitvegappoptimizedmodels.keras"
model = load_model(model_path, compile=False)
print("Model loaded successfully!")
print("Input shape:", model.input_shape)
print("Output shape:", model.output_shape)
print("Last layer:", model.layers[-1].__class__.__name__, getattr(model.layers[-1], "activation", None))

zeros = np.zeros((1, 299, 299, 3), dtype=np.float32)
pred = model.predict(zeros, verbose=0)[0]
print("Softmax sum:", float(np.sum(pred)))
print("Softmax max:", float(np.max(pred)))
print("Argmax index:", int(np.argmax(pred)))
