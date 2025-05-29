import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse
import logging
import time
import torch

from .data import data_process,data_process_st,data_process_all_kpcans_tags
from .inference import keyphrases_selection, keyphrases_selection_st
from torch.utils.data import DataLoader
from transformers import T5ForConditionalGeneration
import datasets

 
def get_setting_dict(max_len=512):
    setting_dict = {}
    setting_dict["max_len"] = max_len
    setting_dict["temp_en"] = "movie review:"
    setting_dict["temp_de"] = "this movie review mainly talks about "
    setting_dict["model"] = "base"
    setting_dict["enable_filter"] = False
    setting_dict["enable_pos"] = True
    setting_dict["position_factor"] = 1.2e8
    setting_dict["length_factor"] = 0.6
    return setting_dict

def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input dataset.")
    parser.add_argument("--dataset_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The input dataset name.")
    parser.add_argument("--batch_size",
                        default=None,
                        type=int,
                        required=True,
                        help="Batch size for testing.")
    parser.add_argument("--log_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="Path for Logging file")
    args = parser.parse_args()
    return args


def get_key_phrases(doc_list, dataname, max_len = 512, topk=-1, whole_doc=False):
    setting_dict = get_setting_dict(max_len=512) # T5-baseis 512
    # args = parse_argument()
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # log = Logger(args.log_dir + args.dataset_name + '.log')
    start = time.time()
    # log.logger.info("Start Testing ...")

    
    # s11 = datasets.load_from_disk('../../data/imdb/')['train']
    model = T5ForConditionalGeneration.from_pretrained("t5-"+ setting_dict["model"])
    model.to(device)
    # doc_list = [s11[i]['text'] for i in range(len(s11))]
    print("get_key_phrases,  dataname ",dataname)
    dataset, doc_list = data_process_st(setting_dict, doc_list, dataname, whole_doc=whole_doc)
    dataloader = DataLoader(dataset, num_workers=0, batch_size=4)
    # keyphrases_selection(setting_dict, doc_list, labels_stemed, labels, model, dataloader, device, log)
    all_kp,all_score = keyphrases_selection_st(setting_dict, doc_list, model, dataloader, device, topk)

    end = time.time()
    # log_setting(log, setting_dict)
    print("Processing time: {}".format(end-start))
    return all_kp,all_score

def get_key_phrases_cans_and_pos_tags(doc_list,max_len):
    setting_dict = get_setting_dict(max_len=max_len)
    return data_process_all_kpcans_tags(setting_dict, doc_list)
        


def main():
    setting_dict = get_setting_dict()
    # args = parse_argument()
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # log = Logger(args.log_dir + args.dataset_name + '.log')
    start = time.time()
    # log.logger.info("Start Testing ...")

    
    s11 = datasets.load_from_disk('../../data/imdb/')['train']
    model = T5ForConditionalGeneration.from_pretrained("t5-"+ setting_dict["model"])
    model.to(device)
    doc_list = [s11[i]['text'] for i in range(2)]
    dataset, doc_list = data_process_st(setting_dict, doc_list, 'imdb')
    dataloader = DataLoader(dataset, num_workers=4, batch_size=64)
    # keyphrases_selection(setting_dict, doc_list, labels_stemed, labels, model, dataloader, device, log)
    all_kp,all_score = keyphrases_selection_st(setting_dict, doc_list, model, dataloader, device)

    end = time.time()
    # log_setting(log, setting_dict)
    print("Processing time: {}".format(end-start))

def log_setting(log, setting_dict):
    for i, j in setting_dict.items():
        log.logger.info(i + ": {}".format(j))

class Logger(object):

    def __init__(self, filename, level='info'):

        level = logging.INFO if level == 'info' else logging.DEBUG
        self.logger = logging.getLogger(filename)
        self.logger.propagate = False
        # # format_str = logging.Formatter(fmt)  
        # if args.local_rank == 0 :
        #     level = level
        # else:
        #     level = 'warning'
        self.logger.setLevel(level)  # 

        th = logging.FileHandler(filename,'w')
        # formatter = logging.Formatter('%(asctime)s => %(name)s * %(levelname)s : %(message)s')
        # th.setFormatter(formatter)

        #self.logger.addHandler(sh)  # 
        self.logger.addHandler(th)  # 
        
if __name__ == "__main__":
    torch.multiprocessing.set_sharing_strategy('file_system')
    main()
    
