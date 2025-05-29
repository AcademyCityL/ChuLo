import torch
import torch.nn as nn
from sklearn.metrics import f1_score,classification_report
from tools.distance import get_distance_func
import numpy as np
import re
import string
from collections import Counter

def compare_first(elem):
    return elem[0]

def probability_score(**kwargs):
    def func(output):
        preds = nn.functional.softmax(output,dim=1)
        return preds[:,1].tolist()
    return func

def cosine_similarity_score(**kwargs):
    cosine = nn.CosineSimilarity(**kwargs)
    def func(output1,output2):
        sim = cosine(output1,output2)
        return sim.tolist()
    return func


class HotpotQA_metrics(object):
    # F1 and EM
    def __init__(self,name,data, sp_threshold=0.5):
        self.name = name
        self.data = {
            'answer':{
                'f1':0,
                'best_f1':0,
                'em':0,
                'best_em':0,
                'prec':0,
                'recall':0
            },
            'sp':{
                'f1':0,
                'best_f1':0,
                'em':0,
                'best_em':0,
                'prec':0,
                'recall':0
            },
            'joint':{
                'f1':0,
                'best_f1':0,
                'em':0,
                'best_em':0,
                'prec':0,
                'recall':0
            }
        }
        self.answer_dict = {}
        self.sp_dict = {}
        self.sp_threshold = sp_threshold
        self.eval_file = None
        self.reset()

    def reset(self):
        for k,v in self.data.items():
            v['f1'] = 0
            v['em'] = 0
            v['prec'] = 0
            v['recall'] = 0
        self.update_best = False
        self.answer_dict = {}
        self.sp_dict = {}

    def save(self):
        save = self.data
        return save

    def convert_tokens(self, eval_file, qa_id, pp1, pp2, p_type):
        answer_dict = {}
        for qid, p1, p2, type in zip(qa_id, pp1, pp2, p_type):
            if type == 0:
                context = eval_file[str(qid)]["context"]
                spans = eval_file[str(qid)]["spans"]
                start_idx = spans[p1][0]
                end_idx = spans[p2][1]
                answer_dict[str(qid)] = context[start_idx: end_idx]
            elif type == 1:
                answer_dict[str(qid)] = 'yes'
            elif type == 2:
                answer_dict[str(qid)] = 'no'
            elif type == 3:
                answer_dict[str(qid)] = 'noanswer'
            else:
                assert False
        return answer_dict

    def update(self, input, output, targets, **kwargs):
        y1,y2,ids,q_type,is_support,eval_file = targets['y1'],targets['y2'],targets['ids'],\
        targets['q_type'],targets['is_support'],targets['eval_file']
        if self.eval_file is None:
            self.eval_file = eval_file
        logit1, logit2, predict_type, predict_support, yp1, yp2 = output
        answer_dict_ = self.convert_tokens(eval_file, ids, yp1.data.cpu().numpy().tolist(), \
            yp2.data.cpu().numpy().tolist(), np.argmax(predict_type.data.cpu().numpy(), 1))
        self.answer_dict.update(answer_dict_)

        predict_support_np = torch.sigmoid(predict_support[:, :, 1]).data.cpu().numpy()
        for i in range(predict_support_np.shape[0]):
            cur_sp_pred = []
            cur_id = ids[i]
            for j in range(predict_support_np.shape[1]):
                if j >= len(eval_file[cur_id]['sent2title_ids']): break
                if predict_support_np[i, j] > self.sp_threshold:
                    cur_sp_pred.append(eval_file[cur_id]['sent2title_ids'][j])
            self.sp_dict.update({cur_id: cur_sp_pred})

    def update_answer(self, metrics, prediction, gold):
        # print(prediction, gold)
        em = self.exact_match_score(prediction, gold)
        f1, prec, recall = self.f1_score(prediction, gold)
        metrics['em'] += float(em)
        metrics['f1'] += f1
        metrics['prec'] += prec
        metrics['recall'] += recall
        if metrics['em'] > metrics['best_em'] or metrics['f1'] > metrics['best_f1']:
            metrics['best_em'] = metrics['em']
            metrics['best_f1'] = metrics['f1']
        return em, prec, recall

    def update_sp(self, metrics, prediction, gold):
        cur_sp_pred = set(map(tuple, prediction))
        gold_sp_pred = set(map(tuple, gold))
        tp, fp, fn = 0, 0, 0
        for e in cur_sp_pred:
            if e in gold_sp_pred:
                tp += 1
            else:
                fp += 1
        for e in gold_sp_pred:
            if e not in cur_sp_pred:
                fn += 1
        prec = 1.0 * tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = 1.0 * tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2 * prec * recall / (prec + recall) if prec + recall > 0 else 0.0
        em = 1.0 if fp + fn == 0 else 0.0
        print(metrics)
        metrics['em'] += em
        metrics['f1'] += f1
        metrics['prec'] += prec
        metrics['recall'] += recall
        if metrics['em'] > metrics['best_em'] or metrics['f1'] > metrics['best_f1']:
            metrics['best_em'] = metrics['em']
            metrics['best_f1'] = metrics['f1']
        return em, prec, recall

    def compute(self):
        for cur_id, dp in self.eval_file.items():
            can_eval_joint = True
            if cur_id not in self.answer_dict:
                print('missing answer {}'.format(cur_id))
                can_eval_joint = False
            else:
                em, prec, recall = self.update_answer(
                    self.data['answer'], self.answer_dict[cur_id], dp['answer'][0])
            if cur_id not in self.sp_dict:
                print('missing sp fact {}'.format(cur_id))
                can_eval_joint = False
            else:
                sp_em, sp_prec, sp_recall = self.update_sp(
                    self.data['sp'], self.sp_dict[cur_id], dp['supporting_facts'])

            if can_eval_joint:
                joint_prec = prec * sp_prec
                joint_recall = recall * sp_recall
                if joint_prec + joint_recall > 0:
                    joint_f1 = 2 * joint_prec * joint_recall / (joint_prec + joint_recall)
                else:
                    joint_f1 = 0.
                joint_em = em * sp_em

                self.data['joint']['em'] += joint_em
                self.data['joint']['f1'] += joint_f1
                self.data['joint']['prec'] += joint_prec
                self.data['joint']['recall'] += joint_recall

                if self.data['joint']['f1'] > self.data['joint']['best_f1'] or \
                    self.data['joint']['em'] > self.data['joint']['best_em']:
                    self.update_best = True
                    self.data['joint']['best_f1'] = self.data['joint']['f1']
                    self.data['joint']['best_em'] = self.data['joint']['em']

    def normalize_answer(self,s):

        def remove_articles(text):
            return re.sub(r'\b(a|an|the)\b', ' ', text)

        def white_space_fix(text):
            return ' '.join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return ''.join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(s))))


    def f1_score(self, prediction, ground_truth):
        normalized_prediction = self.normalize_answer(prediction)
        normalized_ground_truth = self.normalize_answer(ground_truth)

        ZERO_METRIC = (0, 0, 0)

        if normalized_prediction in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
            return ZERO_METRIC
        if normalized_ground_truth in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
            return ZERO_METRIC

        prediction_tokens = normalized_prediction.split()
        ground_truth_tokens = normalized_ground_truth.split()
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return ZERO_METRIC
        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1, precision, recall


    def exact_match_score(self, prediction, ground_truth):
        return (self.normalize_answer(prediction) == self.normalize_answer(ground_truth))


    def metric_max_over_ground_truths(self,metric_fn, prediction, ground_truths):
        scores_for_ground_truths = []
        for ground_truth in ground_truths:
            score = metric_fn(prediction, ground_truth)
            scores_for_ground_truths.append(score)
        return max(scores_for_ground_truths)

    def show(self,end_char = '\n'):
        h = ""
        for k,v  in self.data.items():
            h = h + k + " "
            for kk,vv in v.items():
                if 'best' not in kk:
                    h = h + "{}: {:.4f}, ".format(kk, vv)
        print(h,end = end_char)

    def show_best(self, end_char = '\n'):
        h = ""
        for k,v  in self.data.items():
            h = h + k + " "
            for kk,vv in v.items():
                if 'best' in kk:
                    h = h + "{}: {:.4f}, ".format(kk, vv)
        print(h,end = end_char)

class QAAccuracy(object):
    def __init__(self,name,data,update = 'single', score_name = 'probability',sep = '[SEP]',score_kwargs = {}):
        self.name = name
        self.update_type = update
        self.score_name = score_name
        self.score_kwargs = score_kwargs
        self.acc = 0
        self.best_acc = 0
        self.spe_id = data.get_token_id(sep)
        if self.score_name == 'cosine_similarity':
            self.score_func = cosine_similarity_score(**self.score_kwargs)
        elif self.score_name == 'probability':
            self.score_func = probability_score(**self.score_kwargs)
        self.reset()

    def reset(self):
        self.data = {}
        self.update_best = False

    def save(self):
        save = {
            'data':self.data,
            'acc':self.acc,
            'best_acc':self.best_acc,
        }
        return save

    def _update_triplet(self, input, output):
        anchor_ids = input[-1]
        anchor,pos,neg = output
        pos_score = self.score_func(anchor,pos)
        neg_score = self.score_func(anchor,neg)
        for str_q,p_score,n_score, in zip(anchor_ids,pos_score,neg_score):
            anchor = str_q
            # print("str_q", str_q)
            if anchor not in self.data:
                self.data[anchor] = []
            anchor_info = self.data[anchor]
            anchor_info.append([p_score,1]) # positive
            anchor_info.append([n_score,0]) # negative
    
    def _update_pair(self,input, output, targets):
        anchor_ids = input[-1]
        anchor_f,target_f = output
        all_score = self.score_func(anchor_f,target_f)
        for str_q, score, label in zip(anchor_ids, all_score, targets):
            anchor = str_q
            if anchor not in self.data:
                self.data[anchor] = []
            anchor_info = self.data[anchor]
            anchor_info.append([score,label])
    
    def _update_single(self, input, output, targets):
        anchor_ids = input[-1]
        all_score = self.score_func(output)
        for idx_sequence, score, label in zip(anchor_ids, all_score, targets):
            anchor = []
            for idx in idx_sequence:
                if idx == self.spe_id:
                    break
                anchor.append(str(idx))
            anchor = ' '.join(anchor)
            if anchor not in self.data:
                self.data[anchor] = []
            anchor_info = self.data[anchor]
            anchor_info.append([score,label])

    def calculate_QAACC(self):
        correct,count = 0,0
        for anchor, anchor_info in self.data.items():
            count += 1
            anchor_info.sort(key = compare_first,reverse = True) 
            # print(anchor_info)
            if anchor_info[0][1] == 1:
                correct += 1
        return correct/count

    def update(self,input, output, targets, **kwargs):
        if not isinstance(input, tuple):
            input = (0,input)
        if self.update_type == 'single':
            self._update_single(input, output, targets)
        elif self.update_type == 'pair':
            self._update_pair(input, output, targets)
        elif self.update_type == 'triplet':
            if len(output) == 2:
                self._update_pair(input, output, targets)
            elif len(output) == 3:
                self._update_triplet(input, output)

    def compute(self):
        self.acc = self.calculate_QAACC()
        if self.acc > self.best_acc:
            self.update_best = True
            self.best_acc = self.acc

    def show(self, end_char = '\n'):
        h = "ACC : {:.4f}".format(self.acc)
        print(h,end = end_char)

    def show_best(self, end_char = '\n'):
        print("Best ACC : {:.4f}".format(self.best_acc),end = end_char)

class MeanScoreRank(object):
    def __init__(self,name,data,update = 'single', score_name = 'cosine_similarity',sep = '[SEP]',score_kwargs = {}):
        self.name = name
        self.update_type = update
        self.score_name = score_name
        self.score_kwargs = score_kwargs
        self.map = 0
        self.mrr = 0
        self.best_map = 0
        self.best_mrr = 0
        self.spe_id = data.get_token_id(sep)
        if self.score_name == 'cosine_similarity':
            self.score_func = cosine_similarity_score(**self.score_kwargs)
        elif self.score_name == 'probability':
            self.score_func = probability_score(**self.score_kwargs)
        self.reset()

    def reset(self):
        self.data = {}
        self.update_best = False

    def save(self):
        save = {
            'data':self.data,
            'map':self.map,
            'mrr':self.mrr,
            'best_map':self.best_map,
            'best_mrr':self.best_mrr,
        }
        return save

    def _update_triplet(self, input, output):
        anchor_ids = input[-1]
        anchor,pos,neg = output
        pos_score = self.score_func(anchor,pos)
        neg_score = self.score_func(anchor,neg)
        for str_q,p_score,n_score, in zip(anchor_ids,pos_score,neg_score):
            anchor = str_q
            # print("str_q", str_q)
            if anchor not in self.data:
                self.data[anchor] = []
            anchor_info = self.data[anchor]
            anchor_info.append([p_score,1]) # positive
            anchor_info.append([n_score,0]) # negative
    
    def _update_pair(self,input, output, targets):
        anchor_ids = input[-1]
        anchor_f,target_f = output
        all_score = self.score_func(anchor_f,target_f)
        for str_q, score, label in zip(anchor_ids, all_score, targets):
            anchor = str_q
            if anchor not in self.data:
                self.data[anchor] = []
            anchor_info = self.data[anchor]
            anchor_info.append([score,label])
    
    def _update_single(self, input, output, targets):
        anchor_ids = input[-1]
        all_score = self.score_func(output)
        for idx_sequence, score, label in zip(anchor_ids, all_score, targets):
            anchor = []
            for idx in idx_sequence:
                if idx == self.spe_id:
                    break
                anchor.append(str(idx))
            anchor = ' '.join(anchor)
            if anchor not in self.data:
                self.data[anchor] = []
            anchor_info = self.data[anchor]
            anchor_info.append([score,label])

    def calculate_MAP_MRR(self):
        ap,rr,count = 0,0,0
        for anchor, anchor_info in self.data.items():
            count += 1
            anchor_info.sort(key = compare_first,reverse = True) 
            pos_num = 0
            q_ap = 0
            q_rr = 0
            for i in range(len(anchor_info)):
                candidate_info = anchor_info[i]
                # print(answer_info)
                if candidate_info[1] == 1:
                    pos_num += 1
                    q_ap += pos_num/(i+1)
                    if pos_num == 1:
                        q_rr = 1/(i+1)
            q_ap = q_ap/pos_num if pos_num > 0 else 0
            ap += q_ap
            rr += q_rr
        return ap/count, rr/count

    def update(self,input, output, targets, **kwargs):
        if not isinstance(input, tuple):
            input = (0,input)
        if self.update_type == 'single':
            self._update_single(input, output, targets)
        # elif self.update_type == 'pair':
        #     self._update_pair(input, output, targets)
        elif self.update_type == 'triplet':
            if len(output) == 2:
                self._update_pair(input, output, targets)
            elif len(output) == 3:
                self._update_triplet(input, output)

    def compute(self):
        self.map, self.mrr = self.calculate_MAP_MRR()
        if self.map > self.best_map:
            self.update_best = True
            self.best_map = self.map
        if self.mrr > self.best_mrr:
            self.best_mrr = self.mrr
            self.update_best = True

    def show(self, end_char = '\n'):
        h = "MAP : {:.4f}, MRR : {:.4f}".format(self.map,self.mrr)
        print(h,end = end_char)

    def show_best(self, end_char = '\n'):
        print("Best MAP : {:.4f}, Best MRR : {:.4f}".format(self.best_map,self.best_mrr),end = end_char)

# Calcuate the accuracy according to the prediction and the true label.
def topk_accuracy(output, targets, topk=(1,)):
    r"""Computes the precision@k for the specified values of k."""
    maxk = max(topk)
    batch_size = targets.shape(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1).expand_as(pred))

    res = {}
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(int(correct_k.item()), correct_k.item() * (100.0 / batch_size))
    return res

class Accuracy(object):
    def __init__(self,name,topk=1):
        self.name = name
        self.topk = topk
        self.acc = 0
        self.best_acc = 0
        self.reset()

    def reset(self):
        self.k_correct = 0
        self.count = 0
        self.update_best = False
    
    def save(self):
        save = {
            'k_correct':self.k_correct,
            'topk':self.topk,
            'count':self.count,
            'acc':self.acc,
            'best_acc':self.best_acc,
        }
        return save

    def update(self, input, output, targets, **kwargs):
        res = topk_accuracy(output, targets, (self.topk,))
        self.k_correct += res[self.topk][0]
        self.count += len(targets)

    def compute(self):
        self.acc = self.k_correct/self.count
        if self.acc > self.best_acc:
            self.best_acc = self.acc
            self.update_best = True
    
    def show(self,end_char = '\n'):
        h = "Top_{} Accuracy: {:.4f} ".format(self.topk,self.acc)

        print(h,end = end_char)

    def show_best(self, end_char = '\n'):
        print("Best Top_{} Accuracy : {:.4f}, ".format(self.topk,self.best_acc),end = end_char)

class Loss(object):
    def __init__(self,name):
        self.name = name
        self.loss = 0
        self.best_loss = float('inf')
        self.reset()

    def reset(self):
        self.val = 0
        self.sum = 0
        self.count = 0
        self.update_best = False
    
    def save(self):
        save ={
            'loss':self.loss,
            'best_loss':self.best_loss
        }
        return save

    def update(self, input, output, targets, **kwargs):
        num = len(output)
        val = kwargs['loss']
        self.val = val
        self.sum += val * num
        self.count += num

    def compute(self):
        self.loss = self.sum/self.count
        if self.loss < self.best_loss:
            self.best_loss = self.loss
            self.update_best = True

    def show(self,end_char = '\n'):
        print(" Loss: {:.4f} ".format(self.sum/self.count),end = end_char)

    def show_best(self, end_char = '\n'):
        print(" Best Loss: {:.4f}, ".format(self.best_loss),end = end_char)

class F1score(object):
    def __init__(self,name,average='binary'):
        self.name = name
        self.average = average
        self.f1score = 0
        self.best_f1score = 0
        self.reset()

    def reset(self):
        self.preds = []
        self.targets = []
        self.update_best = False

    def save(self):
        save = {
            'f1score':self.f1score,
            'best_f1score':self.best_f1score,
            'report':classification_report(self.targets,self.preds,digits=4)
        }
        return save

    def update(self, input, output, targets, **kwargs):
        self.preds  += output.argmax(1).tolist()
        self.targets += targets.tolist()

    def compute(self):
        self.f1score = f1_score(self.preds,self.targets,average = self.average)
        if self.f1score > self.best_f1score:
            self.best_f1score = self.f1score
            self.update_best = True

    def show(self,end_char = '\n'):
        print("F1: {:.4f}".format(f1_score(self.preds,self.targets,average = self.average)),end = end_char)

    def show_best(self, end_char = '\n'):
        print("Best F1 Score: {:.4f}, ".format(self.best_f1score),end = end_char)
    