from models import *

def get_model(model, metric, model_settings, opt):
    print('Using model bank')
    if model == 'Resnet18' and metric == 'Linear_softmax_ce_head':
        backbone = Resnet18(model_settings['in_feat'], model_settings['emb_size'])
        return Linear_softmax_ce_head(backbone, model_settings)

    elif model == 'Resnet18' and metric == 'AM_normfree_softmax_anneal_ce_head':
        backbone = Resnet18(model_settings['in_feat'], model_settings['emb_size'])
        return AM_normfree_softmax_anneal_ce_head(backbone, model_settings)

    elif model == 'Xvector_SAP' and metric == 'Linear_softmax_ce_head':
        backbone = Xvector.Xvector_SAP(model_settings['in_feat'], model_settings['emb_size'])
        return Xvector.Linear_softmax_ce_head(backbone, model_settings)
    
    elif model == 'Xvector_SAP' and metric == 'AM_normfree_softmax_anneal_ce_head':
        backbone = Xvector.Xvector_SAP(model_settings['in_feat'], model_settings['emb_size'])
        return Xvector.AM_normfree_softmax_anneal_ce_head(backbone, model_settings)

    elif model == 'Xvector_SAP_1L' and metric == 'Linear_softmax_ce_head':
        backbone = Xvector.Xvector_SAP_1L(model_settings['in_feat'], model_settings['emb_size'])
        return Xvector.Linear_softmax_ce_head(backbone, model_settings)

    elif model == 'Xvector_SAP_1L' and metric == 'AM_normfree_softmax_anneal_ce_head':
        backbone = Xvector.Xvector_SAP_1L(model_settings['in_feat'], model_settings['emb_size'])
        return Xvector.AM_normfree_softmax_anneal_ce_head(backbone, model_settings)

    elif model == 'Xvector_SAP_1L' and metric == 'AM_normfree_softmax_anneal_ce_SycnBN':
        backbone = Xvector.Xvector_SAP_1L(model_settings['in_feat'], model_settings['emb_size'])
        return Xvector.AM_normfree_softmax_anneal_ce_SycnBN(backbone, model_settings)

    elif model == 'Standard_ETDNN' and metric == 'AM_normfree_softmax_anneal_ce_head':
        backbone = E_tdnn.Standard_ETDNN(model_settings['in_feat'], model_settings['emb_size'])
        return Xvector.AM_normfree_softmax_anneal_ce_head(backbone, model_settings)

    elif model == 'Standard_ETDNN_plus' and metric == 'AM_normfree_softmax_anneal_ce_head':
        backbone = E_tdnn_plus.Res_Big_ETDNN(model_settings['in_feat'], model_settings['emb_size'])
        return Xvector.AM_normfree_softmax_anneal_ce_head(backbone, model_settings)

    else:
        raise NotImplementedError
