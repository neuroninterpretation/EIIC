# -*- encoding: utf-8 -*-
'''
@Desc    :   1) This is the code for true prediction.
             2) The string 'unk' in data/pred_idx.csv means the data with that index in ADE20k has no scene. So we remove these data in our experiments.

'''
import os
import json
import re
from Cython.Compiler.PyrexTypes import specialize_entry
from _pytest.monkeypatch import K
from numpy import pi
import pandas as pd
from pycparser.ply.yacc import restart
from pylab import *
from tqdm import tqdm

def statistic_true_prediction(neuron_type, data_path, definition_path, pred_path, write_path):
    perd_csv = pd.read_csv(pred_path, sep=',')
    pred = perd_csv['pred'].values.tolist()
    target = perd_csv['target'].values.tolist()
    assert len(pred) == len(target)
    
    data = []
    with open(data_path,'r', encoding='utf-8') as _f:
        for line in _f:
            data.append(json.loads(line))
    
    with open(definition_path, 'r', encoding='utf-8') as _f:
        five_definitions = json.load(_f)
    
    true_explanation_count = {'CM_TP':{'SCC':0,'ICC':0},'SM_TP':{'SCC':0,'ICC':0}, 'CM_F_TP':{'SCC':0,'ICC':0},'SM_F_TP':{'SCC':0,'ICC':0}}  
    true_explanation_count_ratio = {'CM_TP':{'SCC':0,'ICC':0},'SM_TP':{'SCC':0,'ICC':0}, 'CM_F_TP':{'SCC':0,'ICC':0},'SM_F_TP':{'SCC':0,'ICC':0}}
    false_count = 0
    true_count = 0
    for idx, val in tqdm(enumerate(pred)):

        if target[idx] != 'unk' and target[idx] in five_definitions and pred[idx] != 'unk' and pred[idx] in five_definitions:           
            assert target[idx] in five_definitions and target[idx] in five_definitions        
            
            concept_iou, concept_threshold, concept_layer = extract_three_type_neuron_concept(idx, target[idx], data)

            neu_concepts = []
            assert neuron_type in ['highest_iou', 'threshold', 'whole_layer']
            if neuron_type == 'highest_iou':
                neu_concepts = concept_iou
            elif neuron_type == 'threshold':
                neu_concepts = concept_threshold
            else:
                neu_concepts = concept_layer
            if len(neu_concepts) > 0:
                if target[idx] != pred[idx]:
                    false_count += 1

                    target_def_3 = five_definitions[target[idx]]['definition_3']
                    target_def_4 = five_definitions[target[idx]]['definition_4']                                          

                    # true prediction                    
                    ratio_inter_tar_3, jaccard_tar_3 = per_def_statistic_on_six_metrics_true_prediction(neu_concepts, target_def_3)
                    true_explanation_count['CM_F_TP']['SCC'] += ratio_inter_tar_3
                    true_explanation_count['SM_F_TP']['SCC'] += jaccard_tar_3

                    ratio_inter_tar_4, jaccard_tar_4 = per_def_statistic_on_six_metrics_true_prediction(neu_concepts, target_def_4)
                    true_explanation_count['CM_F_TP']['ICC'] += ratio_inter_tar_4
                    true_explanation_count['SM_F_TP']['ICC'] += jaccard_tar_4  

                else:
                    true_count += 1

                    target_def_3 = five_definitions[target[idx]]['definition_3']
                    target_def_4 = five_definitions[target[idx]]['definition_4']
                    
                    ratio_inter_tar_3, jaccard_tar_3 = per_def_statistic_on_six_metrics_true_prediction(neu_concepts, target_def_3)
                    true_explanation_count['CM_TP']['SCC'] += ratio_inter_tar_3
                    true_explanation_count['SM_TP']['SCC'] += jaccard_tar_3

                    ratio_inter_tar_4, jaccard_tar_4 = per_def_statistic_on_six_metrics_true_prediction(neu_concepts, target_def_4)
                    true_explanation_count['CM_TP']['ICC'] += ratio_inter_tar_4
                    true_explanation_count['SM_TP']['ICC'] += jaccard_tar_4

 
    for key_1 in true_explanation_count.keys():
        oper_3 = true_explanation_count[key_1]
        assert key_1.endswith('F_TP') or key_1.endswith('TP')
        for key_2 in oper_3.keys():
            if key_1.endswith('F_TP'):
                true_explanation_count_ratio[key_1][key_2] = true_explanation_count[key_1][key_2] / false_count
            else:
                true_explanation_count_ratio[key_1][key_2] = true_explanation_count[key_1][key_2] / true_count

    with open(write_path, 'w', encoding='utf-8') as _w:
        json.dump(true_explanation_count_ratio, _w)

    print('true prediction explanation: ', len(true_explanation_count_ratio))
    print(true_explanation_count_ratio)

def per_def_statistic_on_four_metrics(concepts, pred_def, tar_def):
    inter_pred = list(set(concepts) & set(pred_def))
    con_def_pred = list(set(concepts) - set(pred_def))
    def_con_pred = list(set(pred_def) - set(concepts))

    ratio_inter_pred = len(inter_pred) / len(pred_def)
    ratio_con_def_pred = len(con_def_pred) / len(pred_def)
    ratio_def_con_pred = len(def_con_pred) / len(concepts)

    inter_tar = list(set(concepts) & set(tar_def))
    con_def_tar = list(set(concepts) - set(tar_def))
    def_con_tar = list(set(tar_def) - set(concepts))

    ratio_inter_tar = len(inter_tar) / len(tar_def)
    ratio_con_def_tar = len(con_def_tar) / len(tar_def)
    ratio_def_con_tar = len(def_con_tar) / len(concepts)

    jaccard_pred = len(inter_pred) / (len(inter_pred) + len(con_def_pred) + len(def_con_pred))
    jaccard_tar = len(inter_tar) / (len(inter_tar) + len(con_def_tar) + len(def_con_tar))

    flag_intersection = 0
    flag_differenceset = 0
    flag_jaccard = 0


    if ratio_inter_pred > ratio_inter_tar:
        flag_intersection = 1
    
    if ratio_con_def_pred > ratio_con_def_tar:
        flag_differenceset = 1

    if jaccard_pred > jaccard_tar:
        flag_jaccard = 1


    return flag_intersection, flag_differenceset, flag_jaccard

def per_def_statistic_on_six_metrics_true_prediction(concepts, tar_def):

    inter_tar = list(set(concepts) & set(tar_def))
    con_def_tar = list(set(concepts) - set(tar_def))
    def_con_tar = list(set(tar_def) - set(concepts))

    ratio_inter_tar = len(inter_tar) / len(tar_def)

    jaccard_tar = len(inter_tar) / (len(inter_tar) + len(con_def_tar) + len(def_con_tar))



    return ratio_inter_tar, jaccard_tar


def extract_three_type_neuron_concept(data_idx,scene, data):
    for per_d in data:
        if per_d['data_idx'] == data_idx:
            units = per_d['per_units']    
    
    concept_layer = []
    results = {}
    first_con = {}
    threshold = 0.0
    val_list = []
    for unit in units.keys():
        val_list.append(units[unit]['activated_concepts_val'][0])
    val_list = sorted(val_list)
    threshold = val_list[0]
    
    for unit in units.keys():
        idx = units[unit]['activated_concepts_idx']
        val = units[unit]['activated_concepts_val']
        if scene in idx:
            idx.remove(scene)
            val.remove(val[-1])
        for idx_v, val_v in enumerate(val):
            if val_v >= threshold and not idx[idx_v].endswith('-c'):
                if idx[idx_v] not in results:
                    results[idx[idx_v]] = 1
                else:
                    results[idx[idx_v]] = results[idx[idx_v]] + 1
    
        if idx[0] not in first_con:
            first_con[idx[0]] = 1
        else:
            first_con[idx[0]] = first_con[idx[0]] +1
        
        for per_con in idx:
            if per_con not in concept_layer and '-c' not in idx:
                concept_layer.append(per_con)

    
    first_con=dict(sorted(first_con.items(),key=lambda x:x[1],reverse=True))
    results=dict(sorted(results.items(),key=lambda x:x[1],reverse=True))
    
    concept_iou = []
    for line in first_con:
        if not line.endswith('-c'):
            concept_iou.append(line)
    concept_threshold = []
    for line in results:
        if not line.endswith('-c'):
            concept_threshold.append(line)

    return concept_iou, concept_threshold, concept_layer



if __name__=='__main__':
    # false prediction
    # neuron concepts type: whole_layer, highest_iou, threshold
    neuron_type = 'whole_layer'
    data_path = 'data/per_data_results.json'
    definition_path = 'data/five_definitions.json'
    pred_path = 'data/pred_idx.csv' 
    write_path = 'results/results.json'   
    statistic_true_prediction(neuron_type, data_path, definition_path, pred_path, write_path)



    
    
