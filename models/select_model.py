
"""
# --------------------------------------------
# define training model
# --------------------------------------------
"""


def define_Model(opt):
    model = opt['model']      # one input: L
    
    if model == 'plain':
        from models.model_plain import ModelPlain as M

    elif model == 'plaindiscrossformer':
        from models.model_plain_stage1 import ModelPlainStage1 as M

    else:
        raise NotImplementedError('Model [{:s}] is not defined.'.format(model))

    m = M(opt)

    print('Training model [{:s}] is created.'.format(m.__class__.__name__))
    return m
