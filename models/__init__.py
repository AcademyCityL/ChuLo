# -*- coding: utf-8 -*-

from models.bert import Bert
from models.clean_qdnn import CLQDNN
from models.bi_lstm import BI_LSTM
from models.neural_network import Simple_Text_NN
from models.my_transformer import Transformer_Encoder
from models.my_bert import BERT
def get_model(params,data,config={}):   
    if len(config) > 0:
        params['name'] = config['MODEL']['name']
    print("model name: ",params['name'],config['MODEL'].get('model_name',""))
    print("model params: ",list(params.keys()))
    if params['name'] == "qdnn":
        model = QDNN(data,**params['kwargs'])
    elif params['name'] == "bert":
        model = Bert(data,**params['kwargs'])
    elif params['name'] == "pl_qdnn":
        model = PLQDNN(data,config)
    elif params['name'] == "clean_qdnn":
        model = CLQDNN(config,params)
    elif params['name'] == "bi_lstm":
        model = BI_LSTM(config,params)
    elif params['name'] == "simple_text_nn":
        model = Simple_Text_NN(config,params)
    elif params['name'] == "transformer_encoder":
        model = Transformer_Encoder(config,params)
    elif params['name'] == "BERT":
        model = BERT(config,params)
    else:
        raise Exception("model not supported: {}".format(params['name']))
    return model
