#%%import os
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from PIL import Image
import joblib

# Image settings
IMG_SIZE = (128, 128)
dataset_path =r"C:\Users\Maria Abitha\Desktop\Apple_leaf"

# Load images and labels
features = []
labels = []

for class_dir in os.listdir(dataset_path):
    class_path = os.path.join(dataset_path, class_dir)
    if os.path.isdir(class_path):
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            try:
                img = Image.open(img_path).resize(IMG_SIZE).convert('RGB')
                img_array = np.array(img).flatten()
                features.append(img_array)
                labels.append(class_dir)
            except:
                print("❌ Error loading image:", img_path)

features = np.array(features)
labels = np.array(labels)

# Encode labels
le = LabelEncoder()
labels_encoded = le.fit_transform(labels)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(features, labels_encoded, test_size=0.2, random_state=42)

# Train Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Accuracy
y_pred = rf_model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"🎯 Random Forest Accuracy: {acc * 100:.2f}%")

# Save model and encoder
joblib.dump(rf_model, "rf_model.pkl")
joblib.dump(le, "label_encoder.pkl")

print("✅ Random Forest model saved as rf_model.pkl")
