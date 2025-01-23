import torch
from transformers import BertForSequenceClassification, BertTokenizer
from sklearn.preprocessing import LabelEncoder
import pickle


model = BertForSequenceClassification.from_pretrained('saved_model')
tokenizer = BertTokenizer.from_pretrained('saved_tokenizer')


with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)


print("Label Mapping:")
for i, class_name in enumerate(label_encoder.classes_):
    print(f"{i}: {class_name}")


def predict(text):

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=64)


    with torch.no_grad():
        outputs = model(**inputs)


    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=-1).item()

    return predicted_class

sample_text = "i want to change my card"
predicted_label = predict(sample_text)


print(f"Predicted Label: {predicted_label}")
print(f"Predicted Class: {label_encoder.classes_[predicted_label]}")