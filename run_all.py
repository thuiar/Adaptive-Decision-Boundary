import itertools
import subprocess
import os
##############fixed######################
max_seq_lengths = {'oos':'30', 'stackoverflow':'35','banking':'55'}
seeds = [str(i) for i in range(1)]
# #############group parameters#################
datasets = ['oos','banking','stackoverflow']
known_cls_ratios = ['0.75']
labeled_ratios = ['1.0']
methods = ['ADB']
#################single parameter###############3
gpu_id = '1'
num_train_epochs = '100'
#save and freeze => add --save / --freeze_bert_parameters
pro_id = {'seed':0, 'task_name':1, 'known_cls_ratio':2, 'labeled_ratio':3, 'method':4}
c_parameters = [seeds, datasets, known_cls_ratios, labeled_ratios, methods]

for param in itertools.product(*c_parameters):     
    command = [
        'python','run.py',
        '--seed', param[pro_id['seed']],
        '--task_name', param[pro_id['task_name']],
        '--known_cls_ratio', param[pro_id['known_cls_ratio']],
        '--labeled_ratio', param[pro_id['labeled_ratio']],
        '--method', param[pro_id['method']], 
        '--gpu_id', gpu_id, 
        '--max_seq_length', max_seq_lengths[param[pro_id['task_name']]],
        '--num_train_epochs', num_train_epochs,
        '--freeze_bert_parameters',
        '--save_results'
    ]
    p = subprocess.Popen(command)
    p.communicate()
    print('seed:{} finished, process on {}, return code={}'.format(param[pro_id['seed']], p.pid, p.returncode))
