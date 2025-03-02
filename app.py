from flask import Flask, request, jsonify
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch._dynamo
import os

app = Flask(__name__)

# Set device
torch._dynamo.config.suppress_errors = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_id = "nlptown/bert-base-multilingual-uncased-sentiment"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSequenceClassification.from_pretrained(model_id, torch_dtype=torch.float16)

model.to(device)
model.eval()
id2label = model.config.id2label


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        text = data.get("data", "")

        if not text:
            return jsonify({"error": "No text provided"}), 400

        inputs = tokenizer(text, return_tensors="pt").to(device)

        with torch.inference_mode():
            outputs = model(**inputs)
            predictions = outputs.logits.argmax(dim=-1)

        rating = id2label[predictions.item()]
        sentiment = ""
        if(int(rating[0])>2 and int(rating[0])<4):
            sentiment = "neutral"
        elif(int(rating[0])>=4):
            sentiment = "positive"
        else:
            sentiment = "negative"
       
        
        return jsonify({"sentiment": sentiment})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  
    app.run(host="0.0.0.0", port=port)