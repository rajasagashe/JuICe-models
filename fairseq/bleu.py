# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import ctypes
import math

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction, corpus_bleu
import pandas as pd
import torch

try:
    from fairseq import libbleu
except ImportError as e:
    import sys
    sys.stderr.write('ERROR: missing libbleu.so. run `pip install --editable .`\n')
    raise e


C = ctypes.cdll.LoadLibrary(libbleu.__file__)


class BleuStat(ctypes.Structure):
    _fields_ = [
        ('reflen', ctypes.c_size_t),
        ('predlen', ctypes.c_size_t),
        ('match1', ctypes.c_size_t),
        ('count1', ctypes.c_size_t),
        ('match2', ctypes.c_size_t),
        ('count2', ctypes.c_size_t),
        ('match3', ctypes.c_size_t),
        ('count3', ctypes.c_size_t),
        ('match4', ctypes.c_size_t),
        ('count4', ctypes.c_size_t),
    ]


class SacrebleuScorer(object):
    def __init__(self):
        import sacrebleu
        self.sacrebleu = sacrebleu
        self.reset()

    def reset(self, one_init=False):
        if one_init:
            raise NotImplementedError
        self.ref = []
        self.sys = []

    def add_string(self, ref, pred):
        self.ref.append(ref)
        self.sys.append(pred)

    def score(self, order=4):
        return self.result_string(order).score

    def result_string(self, order=4):
        if order != 4:
            raise NotImplementedError
        return self.sacrebleu.corpus_bleu(self.sys, [self.ref])

class Metric():
    '''This class computes bleu and em on generated code.'''
    def __init__(self):
        self.target_key = 'targ'
        self.pred_key = 'pred'
        self.targs = []
        self.preds = []
        columns = ['bleu', 'em', 'corpus_bleu']
        columns.append('epoch')
        self.metrics_per_epoch = pd.DataFrame(columns=columns)

    def compute_corpus_bleu(self, targs, preds):
        references = [[t[self.target_key]] for t in targs]
        hypothesis = [p[self.pred_key] for p in preds]
        sm = SmoothingFunction()
        return corpus_bleu(references, hypothesis, smoothing_function=sm.method3) * 100

    def compute_bleu(self, test, preds):
        def get_bleu(targ, pred):
            # This is how Ling et al. compute bleu score.
            sm = SmoothingFunction()
            ngram_weights = [0.25] * min(4, len(targ))
            return sentence_bleu([targ], pred,
                                 weights=ngram_weights, smoothing_function=sm.method3)
        bleus = []
        for targ, pred in zip(test, preds):
            bleus.append(get_bleu(targ[self.target_key], pred[self.pred_key]))
        return self.avg(bleus)
        # return sum(bleus)/len(bleus)

    def compute_em(self, test, preds):
        ems = []
        for targ, pred in zip(test, preds):
            ems.append(targ[self.target_key] == pred[self.pred_key])
        return self.avg(ems)
        # return sum(ems)/len(ems)

    @staticmethod
    def avg(lst):
        return sum(lst)/len(lst) * 100

    def compute_precision_recall_f1(self, test, preds):
        precisions = []
        recalls = []
        f1s = []
        for targ, pred in zip(test, preds):
            targ_set = set(targ[self.target_key])
            pred_set = set(pred[self.pred_key])
            # print('===========')
            # print(targ_set)
            # print(pred_set)
            num_correct = len(targ_set & pred_set)
            num_pred = len(pred_set)
            num_target = len(targ_set)

            p = num_correct / num_pred if num_pred > 0 else 0
            r = num_correct / num_target if num_target > 0 else 0
            f1 = 2 * p * r / (p + r) if p + r > 0 else 0
            precisions.append(p)
            recalls.append(r)
            f1s.append(f1)
        return self.avg(precisions), self.avg(recalls), self.avg(f1s)

    def compute_metrics_helper(self, test, preds, epoch):
        if len(test) != preds:
            # if trunc param was passed preds will be shorter so we truncate.
            test = test[:len(preds)]

        metrics_dict = {'epoch': [epoch]}

        metrics_dict['bleu'] = self.compute_bleu(test, preds)
        metrics_dict['em'] = self.compute_em(test, preds)
        metrics_dict['corpus_bleu'] = self.compute_corpus_bleu(test, preds)

        p, r, f1 = self.compute_precision_recall_f1(test, preds)
        metrics_dict['precision'] = p
        metrics_dict['recall'] = r
        metrics_dict['f1'] = f1


        # we need to pass index since values are scalars and not lists
        t = pd.DataFrame(metrics_dict, index=[0])

        self.metrics_per_epoch = self.metrics_per_epoch.append(t, ignore_index=True, sort=False)
        self.metrics_per_epoch = self.metrics_per_epoch.apply(pd.to_numeric)


    def add_string(self, ref, pred):
        self.targs.append(ref)
        self.preds.append(pred)

    def compute_metrics(self, epoch):
        '''Computes bleu em on all targ/pred and stores the values.'''
        self.compute_metrics_helper([{self.target_key: t.split()} for t in self.targs],
                                    [{self.pred_key: p.split()} for p in self.preds], epoch)

    def get_metric(self, name):
        for col_name in self.metrics_per_epoch.columns.values:
            if col_name == name:
                return self.metrics_per_epoch[col_name]

    def result_string(self):
        return self.metrics_string()

    def print_best_string(self):
        '''This is more useful for multi epoch printing where max value for each
        needs to be computed.'''
        print('Best metrics...')
        # print(self.metrics_per_epoch.dtypes)
        lst = []
        for col_name in self.metrics_per_epoch.columns.values:
            if col_name == 'epoch':
                continue
            # print(col_name)
            # print(self.metrics_per_epoch)
            max_idx = self.metrics_per_epoch.idxmax(axis=0)
            # print('___________')
            # print(max_idx)
            row = self.metrics_per_epoch.iloc[max_idx[col_name]]
            lst.append((col_name, f'{row[col_name]:.4f}', row.epoch))
            # print(f'Best {col_name}: {row[col_name]:.2f}, Epoch {row.epoch}')
        print(pd.DataFrame(lst, columns=['metric', 'max', 'epoch']).T)

    def metrics_string(self):
        # print(self.metrics_per_epoch)
        return str(pd.DataFrame(self.metrics_per_epoch.iloc[-1]).T)

    def save(self, filename):
        self.metrics_per_epoch.to_json(filename, orient='records')



class EmScorer(object):
    def __init__(self):
        self.reset()

    def reset(self, one_init=False):
        if one_init:
            raise NotImplementedError
        self.ref = []
        self.sys = []

    def add_string(self, ref, pred):
        self.ref.append(ref)
        self.sys.append(pred)

    def score(self, order=4):
        raise NotImplementedError
        # return self.result_string(order).score

    def result_string(self):
        correct = 0
        for ref, pred in zip(self.ref, self.sys):
            if ref == pred:
                # print('correct')
                # print(ref)
                # print(pred)
                correct += 1
        return (correct / len(self.ref)) * 100
        # if order != 4:
        #     raise NotImplementedError
        # return self.sacrebleu.corpus_bleu(self.sys, [self.ref])

class Scorer(object):
    def __init__(self, pad, eos, unk):
        self.stat = BleuStat()
        self.pad = pad
        self.eos = eos
        self.unk = unk
        self.reset()

    def reset(self, one_init=False):
        if one_init:
            C.bleu_one_init(ctypes.byref(self.stat))
        else:
            C.bleu_zero_init(ctypes.byref(self.stat))

    def add(self, ref, pred):
        if not isinstance(ref, torch.IntTensor):
            raise TypeError('ref must be a torch.IntTensor (got {})'
                            .format(type(ref)))
        if not isinstance(pred, torch.IntTensor):
            raise TypeError('pred must be a torch.IntTensor(got {})'
                            .format(type(pred)))

        # don't match unknown words
        rref = ref.clone()
        assert not rref.lt(0).any()
        rref[rref.eq(self.unk)] = -999

        rref = rref.contiguous().view(-1)
        pred = pred.contiguous().view(-1)

        C.bleu_add(
            ctypes.byref(self.stat),
            ctypes.c_size_t(rref.size(0)),
            ctypes.c_void_p(rref.data_ptr()),
            ctypes.c_size_t(pred.size(0)),
            ctypes.c_void_p(pred.data_ptr()),
            ctypes.c_int(self.pad),
            ctypes.c_int(self.eos))

    def score(self, order=4):
        psum = sum(math.log(p) if p > 0 else float('-Inf')
                   for p in self.precision()[:order])
        return self.brevity() * math.exp(psum / order) * 100

    def precision(self):
        def ratio(a, b):
            return a / b if b > 0 else 0

        return [
            ratio(self.stat.match1, self.stat.count1),
            ratio(self.stat.match2, self.stat.count2),
            ratio(self.stat.match3, self.stat.count3),
            ratio(self.stat.match4, self.stat.count4),
        ]

    def brevity(self):
        r = self.stat.reflen / self.stat.predlen
        return min(1, math.exp(1 - r))

    def result_string(self, order=4):
        assert order <= 4, "BLEU scores for order > 4 aren't supported"
        fmt = 'BLEU{} = {:2.2f}, {:2.1f}'
        for _ in range(1, order):
            fmt += '/{:2.1f}'
        fmt += ' (BP={:.3f}, ratio={:.3f}, syslen={}, reflen={})'
        bleup = [p * 100 for p in self.precision()[:order]]
        return fmt.format(order, self.score(order=order), *bleup,
                          self.brevity(), self.stat.predlen/self.stat.reflen,
                          self.stat.predlen, self.stat.reflen)
