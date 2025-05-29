from argparse import ArgumentParser
import os
import torch
import pdb
import json
from copy import copy
from transformers import AutoTokenizer

from main_loop import main_loop, prediction, main_parser
from models import ClassificationReasoner
from buffer import Buffer
from utils import CAPACITY
def conditional_trans_classification(qbuf, dbuf):
    assert len(qbuf) == 1
    new_qbuf = Buffer()
    new_qblk = copy(qbuf[0])
    new_qblk.ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(new_qblk.label_name.replace('.', ' ')))
    new_qbuf.blocks.append(new_qblk)
    return new_qbuf, dbuf
if __name__ == "__main__":
    print('Please confirm the LUN data are ready!')
    print('=====================================')
    root_dir = os.path.abspath(os.path.dirname(__file__))
    parser = ArgumentParser(add_help=False)
    # ------------ add dataset-specific argument ----------
    parser.add_argument('--reasoner_config_num_labels', type=int, default=3)
    parser.add_argument('--only_predict', action='store_true')
    # ---------------------------------------------
    parser = main_parser(parser)
    parser.set_defaults(
        train_source = os.path.join(root_dir, 'data', 'lun_train.pkl'),
        val_source = os.path.join(root_dir, 'data', 'lun_val.pkl'),
        test_source = os.path.join(root_dir, 'data', 'lun_test.pkl')
    )
    config = parser.parse_args()
    config.reasoner_cls_name = 'ClassificationReasoner'

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    config.conditional_transforms = [conditional_trans_classification]

    if not config.only_predict: # train 
       best_epoch =  main_loop(config, conditional_trans_classification)

    ans, acc, total, acc_long, total_long = {}, 0., 0, 0., 0
    for qbuf, dbuf, buf, relevance_score, ids, output in prediction(config, False, best_epoch):
        _id = qbuf[0]._id
        pred, gold = output[0].view(-1).argmax().item(), int(qbuf[0].label)
        ans[_id] = (pred, gold)
        total += 1.
        acc += pred == gold
        if dbuf.calc_size() + 2 > CAPACITY:
            acc_long += pred == gold
            total_long += 1
            # if pred != gold:
            #     import pdb; pdb.set_trace()
    acc /= total
    acc_long /= total_long
    print(f'accuracy: {acc}')
    print(f'for long text: accuray {acc_long}, total {total_long}')
    with open(os.path.join(config.tmp_dir, 'pred_lun.json'), 'w') as fout:
        json.dump(ans, fout)

