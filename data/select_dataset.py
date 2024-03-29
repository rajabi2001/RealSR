

'''
# --------------------------------------------
# select dataset
# --------------------------------------------
'''

def define_Dataset(dataset_opt):
    dataset_type = dataset_opt['dataset_type'].lower()
    
    # -----------------------------------------
    # Real Image Super-Resolution
    # -----------------------------------------
    if dataset_type in ['stage1', 'swinir1']:
        from data.dataset_stage1 import DatasetStage1 as D
    
    else:
        raise NotImplementedError('Dataset [{:s}] is not found.'.format(dataset_type))

    dataset = D(dataset_opt)
    print('Dataset [{:s} - {:s}] is created.'.format(dataset.__class__.__name__, dataset_opt['name']))
    return dataset
