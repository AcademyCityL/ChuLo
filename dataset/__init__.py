# -*- coding: utf-8 -*-
from dataset.imdb import IMDB, ExperimentIMDB
from dataset.hyperpartisan import HyperParTisan, ExperimentHyperParTisan
# from dataset.wikiqa import WikiQA
from dataset.mr import MR, ExperimentMR
# from dataset.wikihop import WikiHop
# from dataset.hotpotqa import HotpotQA
from dataset.triviaqa import TriviaQA,ExperimentTriviaQA
from dataset.sst import SST,ExperimentSST
from dataset.bbc import BBC,ExperimentBBC
from dataset.bbcn import BBCNews,ExperimentBBCNews
from dataset.offenseval import OffensEval, ExperimentOffensEval
from dataset.twitter import Twitter, ExperimentTwitter
from dataset.msrp import MSRP, ExperimentMSRP
from dataset.r8 import R8,ExperimentR8
from dataset.r52 import R52,ExperimentR52
from dataset.ng20 import NG20,ExperimentNG20
from dataset.ohsumed import Ohsumed, ExperimentOhsumed
from dataset.lun import LUN, ExperimentLUN
from dataset.booksummary import BS, BSDataset, ExperimentBS
from dataset.eurlex import Eurlex, ExperimentEurlex
from dataset.gum import Gum, ExperimentGUM
from dataset.mashqa import MASHQA
from dataset.quac import QUAC, ExperimentQUAC
from dataset.conll import Conll, ExperimentConll    
import os
# def setup(opt):
#     dir_path = os.path.join(opt.datasets_dir, opt.dataset_name)
#     if opt.dataset_name == 'aclimdb':
#         reader = IMDB(dir_path, opt)
#     elif opt.dataset_name == 'hyperpartisan':
#         reader = HyperPartisan(dir_path, opt)
#     elif opt.dataset_name == 'wikiqa':
#         reader = WikiQA(dir_path, opt)
#     elif opt.dataset_type == 'qa':
#         reader = qa.setup(opt)
#     elif opt.dataset_type == 'classification':
#         reader = classification.setup(opt)
#     # else: # By default the type is classification
#     # print(opt.dataset_name)
 
#     return reader

def get_data(config,device,global_config={},pre_cache = True):
    dir_path = os.path.join(config['datasets_dir'], config['dataset_name'])
    print("data name: ",config['dataset_name'])
    # print("model params: ",list(params.keys()))
    if config['dataset_name'] == 'imdb':
        reader = IMDB(dir_path, global_config,pre_cache)
    elif config['dataset_name'] == 'hyperpartisan':
        reader = HyperParTisan(dir_path, global_config,pre_cache)
    elif config['dataset_name'] == 'gum':
        reader = Gum(dir_path, global_config,pre_cache)
    elif config['dataset_name'] == 'conll':
        reader = Conll(dir_path, global_config,pre_cache)
    elif config['dataset_name'] == 'mashqa':
        reader = MASHQA(dir_path, global_config,pre_cache)
    elif config['dataset_name'] == 'quac':
        reader = QUAC(dir_path, global_config,pre_cache)
    # elif config['dataset_name'] == 'wikiqa':
    #     reader = WikiQA(dir_path,device, **config['kwargs'])
    # # elif config['dataset_name'] == 'mr':
    # #     reader = old_MR(dir_path, device, **config['kwargs'])
    # elif config['dataset_name'] == 'wikihop':
    #     reader = WikiHop(dir_path, device, **config['kwargs'])
    # elif config['dataset_name'] == 'hotpotqa':
    #     reader = HotpotQA(dir_path, device, **config['kwargs'])
    elif config['dataset_name'] == 'triviaqa':
        reader = TriviaQA(dir_path, global_config,pre_cache)
    elif config['dataset_name'] == 'sst/binary' or config['dataset_name'] == 'sst/fine-grained':
        reader = SST(dir_path, global_config,pre_cache)
    elif config['dataset_name'] == 'bbc':
        reader = BBC(dir_path, global_config,pre_cache)
    elif config['dataset_name'] == 'bbcn':
        reader = BBCNews(dir_path, global_config,pre_cache)
    elif config['dataset_name'] == 'offenseval':
        reader = OffensEval(dir_path, global_config,pre_cache)
    elif config['dataset_name'] == 'twitter':
        reader = Twitter(dir_path, global_config,pre_cache)
    elif config['dataset_name'] == 'mr':
        reader = MR(dir_path, global_config,pre_cache)
    elif config['dataset_name'] == 'msrp':
        reader = MSRP(dir_path, global_config,pre_cache)
    elif config['dataset_name'] == 'r8':
        reader = R8(dir_path, global_config,pre_cache)
    elif config['dataset_name'] == 'r52':
        reader = R52(dir_path, global_config,pre_cache)
    elif config['dataset_name'] == 'ng20':
        reader = NG20(dir_path, global_config,pre_cache)
    elif config['dataset_name'] == 'ohsumed':
        reader = Ohsumed(dir_path, global_config,pre_cache)
    elif config['dataset_name'] == 'lun':
        reader = LUN(dir_path, global_config,pre_cache)
    elif config['dataset_name'] == 'bs':
        reader = BS(dir_path, global_config,pre_cache,pair = False)
    elif config['dataset_name'] == 'bs-pair':
        reader = BS('data/bs', global_config,pre_cache,pair = True)
    elif config['dataset_name'] == 'eurlex':
        reader = Eurlex(dir_path, global_config,pre_cache,inverse=False)
    elif config['dataset_name'] == 'eurlex-inverse':
        reader = Eurlex('data/eurlex', global_config,pre_cache,inverse=True)
    else:
        raise Exception("dataset not supported: {}".format(config['dataset_name']))
    return reader

def get_experiment_model(global_config):
    name = global_config['DATA']['dataset_name']
    if name == "triviaqa":
        return ExperimentTriviaQA(global_config)
    elif name == 'bbc':
        return ExperimentBBC(global_config)
    elif name == 'bbcn':
        return ExperimentBBCNews(global_config)
    elif name == 'offenseval':
        return ExperimentOffensEval(global_config)
    elif name == 'twitter':
        return ExperimentTwitter(global_config)
    elif name == 'mr':
        return ExperimentMR(global_config)
    elif name == 'sst/binary':
        return ExperimentSST(global_config)
    elif name == 'msrp':
        return ExperimentMSRP(global_config)
    elif name == 'r8':
        return ExperimentR8(global_config)
    elif name == 'r52':
        return ExperimentR52(global_config)
    elif name == 'ng20':
        return ExperimentNG20(global_config)
    elif name == 'ohsumed':
        return ExperimentOhsumed(global_config)
    elif name == 'imdb':
        return ExperimentIMDB(global_config)
    elif name == 'hyperpartisan':
        return ExperimentHyperParTisan(global_config)
    elif name == 'lun':
        return ExperimentLUN(global_config)
    elif name == 'bs':
        return ExperimentBS(global_config)
    elif name == 'eurlex':
        return ExperimentEurlex(global_config)
    elif name == 'bs-pair':
        return ExperimentBS(global_config)
    elif name == 'eurlex-inverse':
        return ExperimentEurlex(global_config)
    elif name == 'gum':
        return ExperimentGUM(global_config)
    elif name == 'quac':
        return ExperimentQUAC(global_config)
    elif  name == 'conll':
        return ExperimentConll(global_config)
    
    