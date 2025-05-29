import os
from tools.tokenizer import get_tokenizer
from customlayers.attention import BiAttention
from sklearn.preprocessing import LabelEncoder
from tools.triviaqa_utils import evaluation_utils
import json
from torch.optim.lr_scheduler import LambdaLR
import string
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import models
import dataset

FILES = {
    'web':{
        'v_val': 'verified-web-dev.json',
        'val':'web-dev.json',
        'test':'web-test-without-answers.json',
        'train':'web-train.json'
    },
    'wiki':{
        'v_val':'trivia_squad_verified_wikipedia_dev_4096.json',
        'val':'trivia_squad_wikipedia_dev_4096.json',
        'test':'trivia_squad_wikipedia_test_without_answers_4096.json',
        'train':'trivia_squad_wikipedia_train_4096.json'
    }
}
### ----- Instructions to train TriviaQA -----
# to use the dataset, you should transform the original data format to squad format
# the following instructions are copied from https://github.com/allenai/longformer/blob/master/scripts/cheatsheet.txt
# Relevant files:
# - scripts/triviaqa.py - our training code implemented in pytorch-lightning
# - scripts/triviaqa_utils - copied from https://github.com/mandarjoshi90/triviaqa with slight modifications

# Convert to a squad-like format. This is slighlty modified from the official scripts
# here tools/triviaqa_utils/convert_to_squad_format.py
# to keep all answers in the document, not just the first answer. It also added the list of
# textual answers to make evaluation easy.
# python -m scripts.triviaqa_utils.convert_to_squad_format  \
#   --triviaqa_file path/to/qa/wikipedia-dev.json  \
#   --wikipedia_dir path/to/evidence/wikipedia/   \
#   --web_dir path/to/evidence/web/  \
#   --max_num_tokens 4096  \   # only keep the first 4096 tokens
#   --squad_file path/to/output/squad-wikipedia-dev-4096.json

class TriviaQADataset(Dataset):
    """
    Largely based on
    https://github.com/allenai/allennlp/blob/master/allennlp/data/dataset_readers/reading_comprehension/triviaqa.py
    and
    https://github.com/huggingface/transformers/blob/master/examples/run_squad.py
    """
    def __init__(self, file_path, mode, max_seq_len, max_doc_len, doc_stride,
                 max_num_answers, ignore_seq_with_no_answers, max_question_len, ratio = 1,tokenizer = None):
        print(file_path)
        assert os.path.isfile(file_path)
        self.file_path = file_path
        with open(self.file_path, "r", encoding='utf-8') as f:
            print(f'reading file: {self.file_path}')
            self.data_json = json.load(f)['data']
            print(f'done reading file: {self.file_path}')
        self.mode = mode
        self.data_json = self.data_json[:int(len(self.data_json)*ratio)]

        self.flit_no_answers()

        self.tokenizer = tokenizer

        self.max_seq_len = max_seq_len # max question + context length
        self.max_doc_len = max_doc_len # max context length
        self.doc_stride = doc_stride
        self.max_num_answers = max_num_answers
        self.ignore_seq_with_no_answers = ignore_seq_with_no_answers
        self.max_question_len = max_question_len

        self.qid_string_to_int_map =  \
            {
                self._get_qid(entry["paragraphs"][0]['qas'][0]['id']): index
                for index, entry in enumerate(self.data_json)
            }
    def set_tokenizer(self,tokenizer):
        # only support Huggingface tokenizers
        self.tokenizer = tokenizer

    def _normalize_text(self, text: str) -> str:  # copied from the official triviaqa repo
        return " ".join(
            [
                token
                for token in text.lower().strip(self.STRIPPED_CHARACTERS).split()
                if token not in self.IGNORED_TOKENS
            ]
        )
    IGNORED_TOKENS = {"a", "an", "the"}
    STRIPPED_CHARACTERS = string.punctuation + "".join([u"‘", u"’", u"´", u"`", "_"])

    def __len__(self):
        return len(self.data_json)

    # def __getitem__(self, idx):
    #     entry = self.data_json[idx]
    #     tensors_list = self.one_example_to_tensors(entry, idx)
    #     assert len(tensors_list) == 1
    #     return tensors_list[0]
    def __getitem__(self, idx):
        return self.data_json[idx]

    def flit_no_answers(self):
        new_data_json = []
        for example in self.data_json:
            if len(example['paragraphs'][0]['qas'][0]['answers']) == 0:
                continue
            new_data_json.append(example)  
        self.data_json = new_data_json

    def _reconstruct(self, examples):
        '''
        answers:[ 
            [{'text': 'a copper statue of Christ', 'answer_start': 188}, 
            {'text': 'a copper statue of Christ', 'answer_start': 188},
            ] 
        ]
        '''
        data = {'questions':[],'context':[],'answers':[],'ids':[],'qids':[],'aliases':[]}
        for one_example in examples:
            one_example = one_example['paragraphs'][0]
            q_as= one_example['qas'][0]
            data['questions'].append(q_as['question'])
            data['answers'].append(q_as['answers'])
            data['ids'].append(q_as['id'])
            data['qids'].append(q_as['qid'])
            data['aliases'].append(q_as['aliases'])
            data['context'].append(one_example['context'])
        return data


    def _process_answers(self,inputs,examples,contex_type = 1):
        offset_mapping = inputs["offset_mapping"]
        sample_map = inputs["overflow_to_sample_mapping"]
        answers = examples["answers"]
        start_positions_list = []
        end_positions_list = []

        for i, offset in enumerate(offset_mapping):
            sample_idx = sample_map[i]
            one_answers = answers[sample_idx]
            start_positions = []
            end_positions = []
            sequence_ids = inputs.sequence_ids(i)
            # Find the start and end of the context
            idx = 0
            while sequence_ids[idx] != contex_type:
                idx += 1
            context_start = idx
            while sequence_ids[idx] == contex_type:
                idx += 1
            context_end = idx - 1

            for answer_dict in one_answers:
                start_char = answer_dict["answer_start"]
                end_char = answer_dict["answer_start"] + len(answer_dict["text"])

                # If the answer is not fully inside the splitted context, continue
                if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
                    continue
                else:
                    # Otherwise it's the start and end token positions
                    idx = context_start
                    while idx <= context_end and offset[idx][0] <= start_char:
                        idx += 1
                    start_positions.append(idx - 1)

                    idx = context_end
                    while idx >= context_start and offset[idx][1] >= end_char:
                        idx -= 1
                    end_positions.append(idx + 1)
       
            # answers from start_positions and end_positions if > self.max_num_answers
            start_positions = start_positions[:self.max_num_answers]
            end_positions = end_positions[:self.max_num_answers]

            # 0 padding up to self.max_num_answers, 0 is the [CLS] token in the inputs
            padding_len = self.max_num_answers - len(start_positions)
            start_positions.extend([0] * padding_len)
            end_positions.extend([0] * padding_len)

            # replace duplicate start/end positions with `0` because duplicates can result into -ve loss values
            found_start_positions = set()
            found_end_positions = set()
            for i, (start_position, end_position) in enumerate(
                    zip(start_positions, end_positions)
                    ):
                if start_position in found_start_positions:
                    start_positions[i] = 0
                if end_position in found_end_positions:
                    end_positions[i] = 0
                found_start_positions.add(start_position)
                found_end_positions.add(end_position)

            start_positions_list.append(start_positions)
            end_positions_list.append(end_positions)

        return start_positions_list,end_positions_list

    def collate_fn_q_a_style_test(self,examples):
        examples = self._reconstruct(examples)
        questions = [q.strip() for q in examples["questions"]]
        inputs_c = self.tokenizer(
            examples["context"],
            max_length=self.max_seq_len,
            truncation=True,
            stride=self.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            return_attention_mask = True,
            return_special_tokens_mask = True,
            return_token_type_ids = True,
            padding="max_length",
            return_tensors = 'pt',
        )
        # After split context, its batch size become larger than original quastion's
        # So we need to put in more questions
        new_questions = []
        for i in inputs_c['overflow_to_sample_mapping']:
            new_questions.append(examples["questions"][i])

        inputs_q = self.tokenizer(
            new_questions,
            max_length=self.max_seq_len,
            truncation=True,
            return_overflowing_tokens=False,
            return_offsets_mapping=True,
            return_attention_mask = True,
            return_special_tokens_mask = True,
            return_token_type_ids = True,
            padding="max_length",
            return_tensors = 'pt',
        )
        

        sample_map = inputs_c["overflow_to_sample_mapping"]
        example_ids = []
        for i in range(len(inputs_c["input_ids"])):
            sample_idx = sample_map[i]
            example_ids.append(examples["id"][sample_idx])

        batch = {
            'inputs_1': inputs_q,
            'inputs_2': inputs_c,
            'example_ids': example_ids,
            'context':examples["context"],
            'qids':examples['qids'],
        }
        return batch

    def collate_fn_q_a_style_train_val(self,examples):
        examples = self._reconstruct(examples)
        questions = [q.strip() for q in examples["questions"]]
        inputs_c = self.tokenizer(
            examples["context"],
            max_length=self.max_seq_len,
            truncation=True,
            stride=self.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            return_attention_mask = True,
            return_special_tokens_mask = True,
            return_token_type_ids = True,
            padding="max_length",
            return_tensors = 'pt',
        )

        # After split context, its batch size become larger than original quastion's
        # So we need to put in more questions
        new_questions = []
        for i in inputs_c['overflow_to_sample_mapping']:
            new_questions.append(examples["questions"][i])
        inputs_q = self.tokenizer(
            new_questions,
            max_length=self.max_seq_len,
            truncation=True,
            return_overflowing_tokens=False,
            return_offsets_mapping=True,
            return_attention_mask = True,
            return_special_tokens_mask = True,
            return_token_type_ids = True,
            padding="max_length",
            return_tensors = 'pt',
        )

        start_positions_list, end_positions_list = self._process_answers(inputs_c,examples,contex_type=0)

        batch = {
            'inputs_1': inputs_q,
            'inputs_2': inputs_c,
            'start_positions_list': torch.tensor(start_positions_list),
            'end_positions_list': torch.tensor(end_positions_list),
            'context':examples["context"],
            'qids':examples['qids'],
            'aliases':examples['aliases']
        }
        return batch

    def collate_fn_self_attn_style_test(self,examples):
        examples = self._reconstruct(examples)
        questions = [q.strip() for q in examples["questions"]]
        inputs = self.tokenizer(
            questions,
            examples["context"],
            max_length=self.max_seq_len,
            truncation="only_second",
            stride=self.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            return_special_tokens_mask = True,
            return_token_type_ids = True,
            padding="max_length",
            return_tensors = 'pt',
        )
        '''
        The only thing we’ll add here is a tiny bit of cleanup of the offset mappings. 
        They will contain offsets for the question and the context, but once we’re in 
        the post-processing stage we won’t have any way to know which part of the input 
        IDs corresponded to the context and which part was the question (the sequence_ids() 
        method we used is available for the output of the tokenizer only). So, 
        we’ll set the offsets corresponding to the question to None:
        '''
        sample_map = inputs["overflow_to_sample_mapping"]
        example_ids = []
        for i in range(len(inputs["input_ids"])):
            sample_idx = sample_map[i]
            example_ids.append(examples["id"][sample_idx])

            sequence_ids = inputs.sequence_ids(i)
            offset = inputs["offset_mapping"][i]
            inputs["offset_mapping"][i] = [
                o if sequence_ids[k] == 1 else None for k, o in enumerate(offset)
            ]

        batch = {
            'inputs_1': inputs,
            "example_ids": example_ids,
            'context': examples["context"],
            'qids': examples['qids'],
            'aliases': examples['aliases'],
        }
        return batch

    def collate_fn_self_attn_style_train_val(self,examples):
        examples = self._reconstruct(examples)
        questions = [q.strip() for q in examples["questions"]]
        inputs = self.tokenizer(
            questions,
            examples["context"],
            max_length=self.max_seq_len,
            truncation="only_second",
            stride=self.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            return_special_tokens_mask = True,
            return_token_type_ids = True,
            padding="max_length",
            return_tensors = 'pt',
        )

        start_positions_list, end_positions_list = self._process_answers(inputs,examples)

        batch = {
            'inputs_1': inputs,
            'start_positions_list': torch.tensor(start_positions_list),
            'end_positions_list': torch.tensor(end_positions_list),
            'context':examples["context"],
            'qids':examples['qids'],
            'aliases':examples['aliases']
        }
        return batch


    def _get_qid(self, qid):
        """all input qids are formatted uniqueID__evidenceFile, but for wikipedia, qid = uniqueID,
        and for web, qid = uniqueID__evidenceFile. This function takes care of this conversion.
        """
        if 'wiki' == self.mode:
            # for evaluation on wikipedia, every question has one answer even if multiple evidence documents are given
            return qid.split('--')[0]
        elif 'web' == self.mode:
            # for evaluation on web, every question/document pair have an answer
            return qid
        else:
            return qid

    # @staticmethod
    def collate_one_doc_and_lists(batch):
        num_metadata_fields = 0  # qids and aliases
        fields = [x for x in zip(*batch)]
        stacked_fields = [torch.stack(field) for field in fields[:-num_metadata_fields]]  # don't stack metadata fields
        stacked_fields.extend(fields[-num_metadata_fields:])  # add them as lists not torch tensors

        # always use batch_size=1 where each batch is one document
        # will use grad_accum to increase effective batch size
        assert len(batch) == 1
        fields_with_batch_size_one = [f[0] for f in stacked_fields]
        return fields_with_batch_size_one

class TriviaQA(pl.LightningDataModule):
    """
    Largely based on
    https://github.com/allenai/allennlp/blob/master/allennlp/data/dataset_readers/reading_comprehension/triviaqa.py
    and
    https://github.com/huggingface/transformers/blob/master/examples/run_squad.py
    """
    def __init__(self,data_path, config, pre_cache=True):
        super(TriviaQA,self).__init__()
        self.data_path =data_path
        self.global_config = config
        self.pre_cache = pre_cache
        #### common procedure
        self.init_attr_from_config()
        self.init_datasets()

    def init_attr_from_config(self):
        # customed method
        data_config = self.global_config['DATA']
        self.batch_size = data_config['batch_size']
        self.train_ratio = data_config['train_ratio']
        self.label_encoder = LabelEncoder()
        self.mode = data_config['mode']
        self.tokenizer_type = data_config['tokenizer_type']
        self.tokenizer_name = data_config['tokenizer_name']
        self.tokenizer_params = data_config['tokenizer_params']
        self.num_workers = data_config['num_workers']
        self.max_seq_len = data_config['max_seq_len'] # question + context
        self.max_doc_len = data_config['max_doc_len'] # context
        self.doc_stride = data_config['doc_stride']
        self.max_num_answers = self.global_config['MODEL']['max_answer_num']
        self.ignore_seq_with_no_answers = data_config['ignore_seq_with_no_answers']
        self.max_question_len = data_config['max_question_len']
        # self.verified = data_config['verified']
        self.input_style = data_config['input_style']
        self.set_datasets = False
        self.set_tokenizer = False
        self.datasets = {'train':None,'val':None,'test':None,'v_val':None}

    def init_datasets(self):
        file_names = FILES['wiki'] if self.mode == 'wiki' else FILES['web']
        for split in ['train','val','test','v_val']:
            ratio = self.train_ratio if split == 'train' else 1
            filepath = os.path.join(self.data_path, file_names[split])
            self.datasets[split] = TriviaQADataset(file_path=filepath, mode=self.mode,tokenizer=None,
                                max_seq_len=self.max_seq_len, max_doc_len=self.max_doc_len,
                                doc_stride=self.doc_stride,
                                max_num_answers=self.max_num_answers,
                                max_question_len=self.max_question_len,
                                ignore_seq_with_no_answers=self.ignore_seq_with_no_answers,ratio = ratio)
        self.init_tokenizer()
        # To avoid multithreading conflication???, reset the tokenizer
        self.init_collect_fn()
        self.set_datasets = True
    def prepare_data(self):
        '''
        Downloading and saving data with multiple processes (distributed settings) will 
        result in corrupted data. Lightning ensures the prepare_data() is called only within
         a single process on CPU, so you can safely add your downloading logic within.
         prepare_data is called from the main process. It is not recommended to assign state 
         here (e.g. self.x = y) since it is called on a single process and if you assign states 
         here then they won’t be available for other processes.
        '''
        
        pass

    def setup(self, stage):
        self.stage = stage
        if self.set_tokenizer == False:
            for split,dataset in self.datasets.items():
                tokenizer = get_tokenizer('bert',self.tokenizer_real_name)
                dataset.set_tokenizer(tokenizer)
            self.set_tokenizer = True


    
    def init_collect_fn(self):
        if self.input_style == 'self_attn':
            self.train_val_collect_fn = TriviaQADataset.collate_fn_self_attn_style_train_val
            self.test_collect_fn = TriviaQADataset.collate_fn_self_attn_style_test
        elif self.input_style == 'q_a_pair':
            self.train_val_collect_fn = TriviaQADataset.collate_fn_q_a_style_train_val
            self.test_collect_fn = TriviaQADataset.collate_fn_q_a_style_test
        else:
            print("ERROR! {} is not supported".format(self.input_style))

    def init_tokenizer(self):
        if self.tokenizer_type == 'non_bert':
            pass
        elif self.tokenizer_type == 'bert':
            if self.tokenizer_name == 'blank_en':
                corpus = []
                for split,dataset in self.datasets.items():
                    for raw_data in dataset.data_json:
                        corpus.append(raw_data['paragraphs'][0]['context'])
                        corpus.append(raw_data['paragraphs'][0]['qas'][0]['question'])
                self.tokenizer = get_tokenizer(self.tokenizer_type,self.tokenizer_name,self.tokenizer_params,\
                    corpus = corpus)
                self.tokenizer_real_name = 'files/tokenizers/{}/'.format(self.tokenizer_name)
                self.tokenizer.save_pretrained(self.tokenizer_real_name)
            else:
                self.tokenizer = get_tokenizer(self.tokenizer_type,self.tokenizer_name,self.tokenizer_params)
                self.tokenizer_real_name = 'files/tokenizers/{}/'.format(self.tokenizer_name)
                self.tokenizer.save_pretrained(self.tokenizer_real_name)

    def train_dataloader(self):
        return DataLoader(self.datasets['train'], batch_size=self.batch_size, shuffle=True,\
            num_workers=self.num_workers, collate_fn=lambda examples: \
                self.train_val_collect_fn(self.datasets['train'], examples))

    def val_dataloader(self):
        return [DataLoader(self.datasets['val'], batch_size=self.batch_size, shuffle=False,\
            num_workers=self.num_workers, collate_fn=lambda examples: \
                self.train_val_collect_fn(self.datasets['val'], examples)),
            DataLoader(self.datasets['v_val'], batch_size=self.batch_size, shuffle=False,\
            num_workers=self.num_workers, collate_fn=lambda examples: \
                self.train_val_collect_fn(self.datasets['v_val'], examples))]

    def test_dataloader(self):
        '''
        the triviaqa doesn't have the test dataset, it only has the predict dataset.
        the difference is that the test dataset also has labels,while predict dataset
        doesn't have
        '''
        pass

    def predict_dataloader(self):
        return DataLoader(self.datasets['test'], batch_size=self.batch_size, shuffle=False,\
            num_workers=self.num_workers, collate_fn=lambda examples: \
                self.test_collect_fn(self.datasets['test'], examples))
    
    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        if self.input_style == 'self_attn':
            for k,v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)
        elif self.input_style == 'q_a_pair':
            for input in ['inputs_1','inputs_2']:
                for k,v in batch[input].items():
                    if isinstance(v, torch.Tensor):
                        batch[input][k] = v.to(device)
            if self.stage != 'predict':
                batch['start_positions_list'] = batch['start_positions_list'].to(device)
                batch['end_positions_list'] = batch['end_positions_list'].to(device)
        return batch

class ExperimentTriviaQA(pl.LightningModule):
    '''
    Each dataset is also an experiment environment, with specific metrics, loss, and a head at the top of the model.
    In this way, it's more convenient to compare different models in the same setting. And also in this style, each model 
    only takes charge of the feature extraction.
    '''
    def __init__(self, config):
        super(ExperimentTriviaQA, self).__init__()
        self.global_config = config
        self.init_attr_from_config()
        self.init_model()

    
    def init_attr_from_config(self):
        # model_config = self.global_config['MODEL']
        # self.num_labels = model_config['num_labels']
        # self.ret_features = model_config['ret_features']
        # self.max_sequence_length = model_config['max_sequence_length']
        # self.measurement_size = model_config['measurement_size']
        # self.dropout = model_config['dropout']
        # self.use_lexicon_as_measurement = model_config['use_lexicon_as_measurement']
        # self.embedding_params = model_config['embedding']
        # ---------------------------------------------------------------- #
        data_config = self.global_config['DATA']
        self.dataset_name = data_config['dataset_name']
        # ---------------------------------------------------------------- #
        experiment_config = self.global_config['EXPERIMENT']
        self.do_warmup = experiment_config.get('do_warmup', False)
        self.steps = experiment_config.get('steps', 30000)
        self.warmup = experiment_config.get('warmup',1000)
        self.lr = experiment_config.get('lr', 1e-3)
        self.optimizer = experiment_config.get('optimizer', 'adam')
        self.optimizer_params = experiment_config.get('optimizer_params', {})
        self.loss = experiment_config.get('loss', "regular_softmax_loss")
        self.max_answer_length = experiment_config.get('max_answer_length', 20)
        self.max_answer_num = experiment_config.get('max_answer_num', 10)
        self.dropout = experiment_config.get('dropout', 0.1)
        self.head_type = experiment_config.get('head_type', "qcpair") #qaselfattn
        # embedding  = {initialization='gensim', kwargs={name='glove-wiki-gigaword-50', freeze=true}}
        data = dataset.get_data(data_config,experiment_config.get('accelerator','gpu'),self.global_config)
        self.data = data
        # ---------------------------------------------------------------- #
        model_config = self.global_config['MODEL']
        self.input_dim = model_config['output_dim']

    def init_model(self):
        params = {'vocab':"todo"}
        self.model = models.get_model(params,"",self.global_config)

    def init_head(self):
        if self.head_type == "qcpair":
            self.head = QCPairHead(self.input_dim,self.dropout)
        elif self.head_type == "qcpair":
            self.head = QCSelfAttnHead(self.input_dim,self.dropout)

    def forward(self, batch, batch_idx,loss=True):
        if self.head_type == "qcpair":
            input_q,input_c = batch['inputs_1'],batch['inputs_2']
            ques_output = self.model(input_q['input_ids'])
            context_output = self.model(input_c['input_ids'])
            output = self.head(input_q,input_c,ques_output,context_output)
            if loss == True:
                loss = self.calculate_loss(output,batch)
            else:
                loss = 0
        return loss, output

    def training_step(self, batch, batch_idx):
        loss,output = self.forward(batch, batch_idx)
        start_logits, end_logits = output
        f1_scores,em_scores,answer_scores,qids = self._generate_answer(batch, start_logits, end_logits)
        self.log_dict({'loss': loss, 'answer_scores': torch.tensor(answer_scores).float().mean(),\
            'f1': torch.tensor(f1_scores).float().mean(), \
            'em': torch.tensor(em_scores).float().mean()})
        return {'loss': loss}

    def validation_step(self, batch, batch_idx,dataloader_idx):
        loss,output = self.forward(batch, batch_idx)
        start_logits, end_logits = output
        f1_scores,em_scores,answer_scores,qids = self._generate_answer(batch, start_logits, end_logits)
        # return {'loss': loss,'qids': qids,'answer_scores': answer_scores,}
        return {'loss': loss,'qids': qids, 'answer_scores': answer_scores,'f1': f1_scores, 'em': em_scores,
        'dataloader_idx':dataloader_idx}

    def on_validation_epoch_end(self, epoch_outputs):
        # epoch_outputs contains two dataloader's outputs, val and vval
        ret = {}
        pre_fix = ['avg_val_','avg_verified_val_']
        logs = {}
        for i in range(len(epoch_outputs)):
            outputs = epoch_outputs[i]
            n_samples = 0
            avg_loss = 0
            for x in outputs:
                n_samples += len(x['qids'])
                avg_loss += x['loss']*len(x['qids'])
            avg_loss = avg_loss/n_samples
            string_qids = [item for sublist in outputs for item in sublist['qids']]
            answer_scores = [item for sublist in outputs for item in sublist['answer_scores']]
            f1_scores = [item for sublist in outputs for item in sublist['f1']]
            em_scores = [item for sublist in outputs for item in sublist['em']]
            print(f'before sync --> sizes: {len(string_qids)}, {len(answer_scores)}, {len(f1_scores)}, {len(em_scores)}')
            # if self.trainer.use_ddp:
            #     torch.distributed.all_reduce(avg_loss, op=torch.distributed.ReduceOp.SUM)
            #     avg_loss /= self.trainer.world_size
            #     torch.distributed.all_reduce(avg_em, op=torch.distributed.ReduceOp.SUM)
            #     avg_em /= self.trainer.world_size

            #     string_qids = self.sync_list_across_gpus(string_qids, avg_loss.device, torch.int)
            #     answer_scores = self.sync_list_across_gpus(answer_scores, avg_loss.device, torch.float)
            #     f1_scores = self.sync_list_across_gpus(f1_scores, avg_loss.device, torch.float)
            #     em_scores = self.sync_list_across_gpus(em_scores, avg_loss.device, torch.int)
            print(f'after sync --> sizes: {len(string_qids)}, {len(answer_scores)}, {len(f1_scores)}, {len(em_scores)}')

            # Because of having multiple documents per questions, some questions might have multiple corresponding answers
            # Here, we only keep the answer with the highest answer_score
            qa_with_duplicates = {}
            for qid, answer_score, f1_score, em_score in zip(string_qids, answer_scores, f1_scores, em_scores):
                if qid not in qa_with_duplicates:
                    qa_with_duplicates[qid] = []
                qa_with_duplicates[qid].append({'answer_score': answer_score, 'f1': f1_score, 'em': em_score})
            f1_scores = []
            em_scores = []
            for qid, answer_metrics in qa_with_duplicates.items():
                top_answer = sorted(answer_metrics, key=lambda x: x['answer_score'], reverse=True)[0]
                f1_scores.append(top_answer['f1'])
                em_scores.append(top_answer['em'])
            avg_val_f1 = sum(f1_scores) / len(f1_scores)
            avg_val_em = sum(em_scores) / len(em_scores)

            logs[pre_fix[i]+'loss'] = avg_loss
            logs[pre_fix[i]+'f1'] = avg_val_f1
            logs[pre_fix[i]+'em']: avg_val_em
            ret[pre_fix[i]+'loss']=avg_loss
            ret['log'] = logs
            ret['progress_bar'] = logs
        return ret

    def predict_step(self, batch, batch_idx):
        _,output = self.forward(batch, batch_idx,False)
        start_logits, end_logits = output
        answers,qids = self._generate_answer(batch, start_logits, end_logits)
        return {'answers': answers,'qids': qids}

    def predict_epoch_end(self, outputs):
        qids = [item for sublist in outputs for item in sublist['qids']]
        answers = [item for sublist in outputs for item in sublist['answers']]

        qa_with_duplicates = {}
        for qid, answer in zip(qids, answers):
            qa_with_duplicates[qid].append({'answer_score': answer['score'], 'answer_text': answer['text'], })

        qid_to_answer_text = {}
        for qid, answer_metrics in qa_with_duplicates.items():
            top_answer = sorted(answer_metrics, key=lambda x: x['answer_score'], reverse=True)[0]
            qid_to_answer_text[qid] = top_answer['answer_text']

        with open('predictions.json', 'w') as f:
            json.dump(qid_to_answer_text, f)

        return {'count': len(qid_to_answer_text)}

    def configure_optimizers(self):
        def lr_lambda(current_step):
            if current_step < self.warmup:
                return float(current_step) / float(max(1, self.warmup))
            return max(0.0, float(self.steps - current_step) / float(max(1, self.steps - self.warmup)))
        if self.optimizer == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        elif self.optimizer == 'rmsprop':
            optimizer = torch.optim.RMSprop(self.parameters(), lr=self.lr)
        scheduler = LambdaLR(optimizer, lr_lambda, last_epoch=-1)
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def get_output_mask(self, outer):
        S = outer.size(1)
        # S = 20
        if S <= self.cache_S:
            return torch.tensor(self.cache_mask[:S, :S], requires_grad=False)
        self.cache_S = S
        np_mask = np.tril(np.triu(np.ones((S, S)), 0), self.max_answer_length) # 15 is used as the  max span length
        # print(outer.dtype,np_mask.dtype,outer.device)
        self.cache_mask = torch.tensor(np_mask,dtype = outer.dtype).to(device = outer.device)
        return torch.tensor(self.cache_mask, requires_grad=False)

    def _or_softmax_cross_entropy_loss(self, logits, n_sample, target, ignore_index=0, dim=-1):
        """loss function suggested in section 2.2 here https://arxiv.org/pdf/1710.10723.pdf"""
        assert logits.ndim == 2
        assert target.ndim == 2
        assert logits.size(0) == target.size(0)

        # with regular CrossEntropyLoss, the numerator is only one of the logits specified by the target
        # here, the numerator is the sum of a few potential targets, where some of them is the correct answer

        # compute a target mask
        target_mask = target == ignore_index
        # replaces ignore_index with 0, so `gather` will select logit at index 0 for the msked targets
        masked_target = target * (1 - target_mask.long())
        # gather logits
        # print(logits.shape,dim,masked_target.shape,masked_target)
        gathered_logits = logits.gather(dim=dim, index=masked_target)
        # Apply the mask to gathered_logits. Use a mask of -inf because exp(-inf) = 0
        gathered_logits[target_mask] = float('-inf')

        # # each batch is one example
        # gathered_logits = gathered_logits.view(1, -1)
        # logits = logits.view(1, -1)

        # numerator = log(sum(exp(gathered logits)))
        log_score = torch.logsumexp(gathered_logits, dim=dim, keepdim=False)
        # denominator = log(sum(exp(logits)))
        log_norm = torch.logsumexp(logits, dim=dim, keepdim=False)

        # compute the loss
        loss = -(log_score - log_norm)

        # some of the examples might have a loss of `inf` when `target` is all `ignore_index`.
        # remove those from the loss before computing the sum. Use sum instead of mean because
        # it is easier to compute. and for one sample, loss sum. for multi-sample, the loss average
        return loss[~torch.isinf(loss)].sum()/n_sample

    def _cross_entropy_loss(self, logits, n_sample, target, ignore_index=0, dim=-1):
        assert logits.ndim == 2
        assert target.ndim == 2
        assert logits.size(0) == target.size(0)
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=0,reduction=sum)
        # with regular CrossEntropyLoss, the numerator is only one of the logits specified by the target
        # here, the numerator is the sum of a few potential targets, where some of them is the correct answer

        # compute a target mask
        target_mask = target == ignore_index
        # replaces ignore_index with 0, so `gather` will select logit at index 0 for the msked targets
        masked_target = target * (1 - target_mask.long())
        # gather logits
        gathered_logits = logits.gather(dim=dim, index=masked_target)
        # Apply the mask to gathered_logits. Use a mask of -inf because exp(-inf) = 0
        gathered_logits[target_mask] = float('-inf')

        
        # compute the loss
        loss = loss_fct(gathered_logits, 1 - target_mask.long())

        # for one sample, loss sum. for multi-sample, the loss average
        return loss/n_sample

    def calculate_loss(self,output,batch):
        start_logits, end_logits = output
        start_positions_list, end_positions_list = batch['start_positions_list'],batch['end_positions_list']
        overflow_to_sample_mapping = batch['inputs_2']['overflow_to_sample_mapping']
        n_sample = len(set(overflow_to_sample_mapping))
        # # If we are on multi-GPU, split add a dimension
        # if len(start_positions_list.size()) > 2:
        #     start_positions = start_positions.squeeze(-1)
        # if len(end_positions.size()) > 2:
        #     end_positions = end_positions.squeeze(-1)

        if self.loss == 'regular_softmax_loss':
            # loss function suggested in section 2.2 here https://arxiv.org/pdf/1710.10723.pdf
            # NOTE: this returns sum of losses, not mean, so loss won't be normalized across different batch sizes
            # but if batch size is always 1, this is not a problem
            start_loss = self._or_softmax_cross_entropy_loss(start_logits, n_sample, start_positions_list, ignore_index=0)
            end_loss = self._or_softmax_cross_entropy_loss(end_logits, n_sample, end_positions_list, ignore_index=0)
        elif self.loss == 'ce_1': # only calculate the first answer
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=0)
            start_positions = start_positions_list[:, 0:1]
            end_positions = end_positions_list[:, 0:1]
            start_loss = loss_fct(start_logits, start_positions[:, 0])
            end_loss = loss_fct(end_logits, end_positions[:, 0])
        elif self.loss == 'ce': 
            start_loss = self._cross_entropy_loss(start_logits, n_sample, start_positions_list, ignore_index=0)
            end_loss = self._cross_entropy_loss(end_logits, n_sample, end_positions_list, ignore_index=0)

        total_loss = (start_loss + end_loss) / 2
        return total_loss

    def _generate_answer(self, batch, start_logits, end_logits):
        outer = start_logits.unsqueeze(2) + end_logits.unsqueeze(1)
        # we process the logits above, hence we don't need to process the mask
        outer_mask = self.get_output_mask(outer)
        outer = outer - 1e30 * (1 - outer_mask)
        topk_answers_indices = outer.flatten(start_dim=1).topk(k=self.max_answer_num).indices
        # print(outer.size())
        start_index = topk_answers_indices.div(outer.size()[1],rounding_mode='floor') #start_index: bs*max_answer_num
        end_index = topk_answers_indices%outer.size()[1]
        answers = [] #n_samples*n_answers, not n_batch_indices
        qids = batch['qids']
        if self.head_type == "qcpair":
            input_name = "inputs_2"
        else:
            input_name = "inputs_1"
        # print("-------",outer.shape,start_index,end_index)
        if 'aliases' in batch: # train or val
            aliases = batch['aliases']
        else:
            aliases = None
        for i in range(len(start_index)):
            sample_id = batch[input_name]['overflow_to_sample_mapping'][i]
            if sample_id+1 > len(answers):
                answers.append([])
            for j in range(len(start_index[i])):
                start = start_index[i][j]
                end = end_index[i][j]
                sp_start = batch[input_name].token_to_chars(i,start)
                sp_end = batch[input_name].token_to_chars(i,end)
                # import copy
                text = batch['context'][sample_id][sp_start[0]:sp_end[1]]
                # print(outer.shape,start,end,sample_id,start_index.shape,end_index.shape)
                # print(outer[i,start,end],outer[i,start,end].shape)
                answers[sample_id].append({'text': text, 'score': outer[i,start,end].item()})
            
        # follow the logic in longformer code, only use the highst score answer of each sample 
        # sample to calculate the metric
        for sample_id in range(len(answers)):
            answers[sample_id] = sorted(answers[sample_id], key=lambda x: x['score'], reverse=True)[0]
        #answers = (n_smaples,)
        #aliase_list = (n_samples,)
        if aliases is not None:
            f1_scores = [evaluation_utils.metric_max_over_ground_truths(evaluation_utils.f1_score, answer['text'],
                                                                        aliase_list)
                        for answer, aliase_list in zip(answers, aliases)]
            # TODO: if slow, skip em_scores, and use (f1_score == 1.0) instead
            em_scores = [evaluation_utils.metric_max_over_ground_truths(evaluation_utils.exact_match_score, answer['text'],
                                                                        aliase_list)
                        for answer, aliase_list in zip(answers, aliases)]
            answer_scores = [answer['score'] for answer in answers]
            return (f1_scores,em_scores,answer_scores,qids)
        else:
            return (answers,qids)

    def sync_list_across_gpus(self, list_to_sync, device, dtype):
        l_tensor = torch.tensor(list_to_sync, device=device, dtype=dtype)
        gather_l_tensor = [torch.ones_like(l_tensor) for _ in range(self.trainer.world_size)]
        torch.distributed.all_gather(gather_l_tensor, l_tensor)
        return torch.cat(gather_l_tensor).tolist()
    
    # def on_validation_epoch_end(self, epoch_outputs):
    #     pass


class QCPairHead(nn.Module):
    def __init__(self, input_dim, dropout):
        super(QCPairHead, self).__init__()
        ##### for hotpotqa, triviaqa
        self.qc_att = BiAttention(input_dim, dropout)
        self.linear_1 = nn.Sequential(
                nn.Linear(input_dim*4, input_dim),
                nn.ReLU()
            )
        self.self_att = BiAttention(input_dim, dropout)
        self.linear_2 = nn.Sequential(
                nn.Linear(input_dim*4, input_dim),
                nn.ReLU()
            )
        self.linear_start = nn.Linear(input_dim, 1)
        self.linear_end = nn.Linear(input_dim, 1)

    def forward(self,input_q,input_c,output_q,output_c):
        '''
        The huggingface tokenized sequence is [CLS] + sequence + [SEP](if need pad :+ [PAD])
        But attention mask doesn't mask special tokens, so we need calculate the real mask
        '''
        input_q['real_mask'] = input_q['attention_mask'] - input_q['special_tokens_mask']
        # in attention mask, pad will be 0, but in special_tokens_mask, pad will be 1
        # thus we need to set real_mask[real_mask<0] = 0 But real sequence will also be 1 and 0 respectively
        input_q['real_mask'][input_q['real_mask']<0] = 0
        output = self.qc_att(output_c, output_q, input_q['real_mask'])
        output = self.linear_1(output)

        output_t = output
        input_c['real_mask'] = input_c['attention_mask'] - input_c['special_tokens_mask']
        input_c['real_mask'][input_c['real_mask']<0] = 0
        output_t = self.self_att(output_t, output_t, input_c['real_mask'])
        output_t = self.linear_2(output_t)

        output = output + output_t
        # print(output.shape,self.linear_start)
        start_logits = self.linear_start(output).squeeze(2)
        # print(start_logits.shape,input_c['real_mask'].shape)
        start_logits[(1-input_c['real_mask']).bool()] = float('-inf')
        end_logits = self.linear_end(output).squeeze(2)
        end_logits[(1-input_c['real_mask']).bool()] = float('-inf')
        # the first token is [CLS]
        return (start_logits, end_logits)

class QCSelfAttnHead(nn.Module):
    def __init__(self, input_dim):
        super(QCSelfAttnHead, self).__init__()
        self.linear_start = nn.Linear(input_dim, 1)
        self.linear_end = nn.Linear(input_dim, 1)

    def forward(self,output,input):
        # to limit the answer only occur in the context
        input['real_mask'] = (input['attention_mask']==input['token_type_ids']).long() - input['special_tokens_mask']
        # print(output.shape,self.linear_start)
        start_logits = self.linear_start(output).squeeze(2)
        # print(start_logits.shape,input_c['real_mask'].shape)
        start_logits[(1-input['real_mask']).bool()] = float('-inf')
        end_logits = self.linear_end(output).squeeze(2)
        end_logits[(1-input['real_mask']).bool()] = float('-inf')