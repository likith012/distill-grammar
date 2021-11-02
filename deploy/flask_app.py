import torch
from flask import Flask
from flask import request
import flask
import sys, os

import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(sys.path[0], '../'))
from configs import config

app = Flask(__name__)

DEVICE = torch.device('cuda:0')

def sentence_prediction(sentence):

    tokenizer = config.TOKENIZER.from_pretrained(config.MODEL_PATH, local_files_only = True)
    model = config.MODEL.from_pretrained(config.MODEL_PATH, local_files_only = True)
    max_len = config.MAX_LEN

    sentence = str(sentence)
    sentence = " ".join(sentence.split())

    inputs = tokenizer.encode_plus( 
            sentence,
            add_special_tokens = True,
            max_length = max_len,
            pad_to_max_length = True,
            return_attention_mask = True,
            return_tensors = 'pt',
            truncation  = True
        )

    ids = torch.LongTensor(inputs['input_ids'][0]).unsqueeze(0)
    mask = torch.LongTensor(inputs['attention_mask'][0]).unsqueeze(0)

    ids = ids.to(DEVICE)
    mask = mask.to(DEVICE)
    model.to(DEVICE)

    outputs = model(
        ids, 
        token_type_ids = None, 
        attention_mask = mask, 
        return_dict=True
    )
    print(outputs)
    outputs = torch.sigmoid(outputs.logits).cpu().detach().numpy()

    return outputs[0][0]

@app.route("/predict")
def predict():

    sentence = request.args.get('sentence')
    positive_prediction = sentence_prediction(sentence)
    negative_prediction = 1 - positive_prediction
    response = {}
    response['response'] = {
        'positive': str(positive_prediction),
        'negative': str(negative_prediction),
        'sentence': str(sentence)
    }
    return flask.jsonify(response)

if __name__ == '__main__':

    app.run()