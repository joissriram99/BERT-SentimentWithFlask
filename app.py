from flask import Flask,request
import torch
import flask
from src import model,config
import torch.nn as nn

from src.model import BERTBaseUncased

app = Flask(__name__)
MODEL = None
DEVICE = config.DEVICE

def sentence_predictor(sentence):
    tokenizer = config.TOKENIZER
    max_len = config.MAX_LEN
    review = str(sentence)
    review = " ".join(review.split())

    inputs = tokenizer.encode_plus(
        review,
        None,
        add_special_tokens=True,
        max_length= max_len
    )

    ids = inputs["input_ids"]
    mask = inputs["attention_mask"]
    token_type_ids = inputs["token_type_ids"]

    padding_length = max_len - len(ids)
    ids = ids + ([0] * padding_length)
    mask = mask + ([0] * padding_length)
    token_type_ids = token_type_ids + ([0] * padding_length)

    ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
    mask = torch.tensor(mask, dtype=torch.long).unsqueeze(0)
    token_type_ids = torch.tensor(token_type_ids, dtype=torch.long).unsqueeze(0)

    ids = ids.to(DEVICE, dtype=torch.long)
    token_type_ids = token_type_ids.to(DEVICE, dtype=torch.long)
    mask = mask.to(DEVICE, dtype=torch.long)

    outputs = MODEL(ids=ids, mask=mask, token_type_ids=token_type_ids)
    outputs = torch.sigmoid(outputs).cpu().detach().numpy()

    return outputs[0][0]

@app.route("/predict")
def predict():
    sentence = request.args.get("sentence")
    positive_prediction = sentence_predictor(sentence)
    negative_prediction = 1 - positive_prediction
    response = {}
    response["responce"] = {
        "Positive_prediction" : str(positive_prediction),
        "Negative_prediction" : str(negative_prediction),
        "Sentence" : str(sentence)
    }

    return flask.jsonify(response)





if __name__ == "__main__":
    MODEL = BERTBaseUncased()
    MODEL.load_state_dict(torch.load(config.MODEL_PATH))
    MODEL.to(DEVICE)
    MODEL.eval()
    app.run(debug=True)

