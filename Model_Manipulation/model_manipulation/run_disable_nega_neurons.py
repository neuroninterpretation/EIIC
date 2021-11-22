# -*- encoding: utf-8 -*-
import os
import json
import re
from tqdm import tqdm

import settings
from loader.model_loader import loadmodel
from dissection.neuron import hook_feature, NeuronOperator
from dissection import contrib
from util.clean import clean
from util.misc import safe_layername
from tqdm import tqdm
import pickle
import os
import pandas as pd
from loader.data_loader import ade20k
import numpy as np


def noop(*args, **kwargs):
    return None


layernames = list(map(safe_layername, settings.FEATURE_NAMES))


with open("neuron_important_data/true_false_prediction_index.json", 'r', encoding='utf-8') as _f:
    true_false_index = json.load(_f)

with open("neuron_important_data/true_scene_neuron_tar.json", 'r', encoding='utf-8') as _f:
    true_neuron = json.load(_f)

with open("neuron_important_data/false_scene_neuron_tar.json",'r', encoding='utf-8') as _f:
    false_neuron = json.load(_f)



hook_modules = []

def probe(def_idx, top_k, pred_type, act_neuron, tf_idx):
    model = loadmodel(hook_feature, hook_modules=hook_modules)
    model.cuda()
    fo = NeuronOperator()
    
    features, maxfeature, preds, logits = fo.feature_extraction(model=model, def_idx=def_idx, top_k=top_k, pred_type=pred_type, act_neuron =act_neuron, tf_idx=tf_idx)
    
    thresholds = [
        fo.quantile_threshold(lf, savepath=f"quantile_{ln}.npy")
        for lf, ln in zip(features, layernames)
    ]
    
    wholeacts = features[-1] > thresholds[-1][np.newaxis, :, np.newaxis, np.newaxis]
    wholeacts = wholeacts.any((2, 3))
    
    pred_records = []
    for i, ((p, t), acts) in enumerate(zip(preds, wholeacts)):
        acts = acts * 1 
        pred_name = ade20k.I2S[p]
        target_name = f"{fo.data.scene(i)}-s"
        if target_name in ade20k.S2I:
            pred_records.append((pred_name, target_name, *acts))
    
    pred_df = pd.DataFrame.from_records(
        pred_records, columns=["pred", "target", *map(str, range(wholeacts.shape[1]))]
    )
    print('pred_type: ', pred_type)
    print('def_idx: ',def_idx)
    print('top_k: ', top_k)
    print(f"Accuracy: {(pred_df.pred == pred_df.target).mean() * 100:.2f}%")
    
    return (pred_df.pred == pred_df.target).mean() * 100

def run_overall():
    # config
    # def_key = ['def_1','def_2','def_3','def_4','def_5']
    def_key = ['def_5']
    write_path = 'neuron_effect_to_model_performance/negative_neuron_on_false_prediction_def_5.json'
    
    
    # top 20
    top_k = [i for i in range(1,1+20)]    
    results = {'def_1':{},'def_2':{},'def_3':{},'def_4':{},'def_5':{}}
    for def_idx in def_key:
        for top_i in top_k:
            accuracy = probe(def_idx, top_i, 'false_prediction', false_neuron, true_false_index)
            if top_i not in results[def_idx]:
                results[def_idx][top_i] = {'accuracy':0}
            results[def_idx][top_i]['accuracy'] = accuracy
    
    
    with open(write_path, 'w', encoding='utf-8') as _w:
        json.dump(results, _w)
    









if __name__=='__main__':
    run_overall()
