from models import *

__all__ = ["get_model"]

def get_model(model, metric, model_settings, opt):
    print('Using model bank')
    if model == 'Resnet18' and metric == 'Linear_softmax_ce_head':
        backbone = Resnet18(model_settings['in_feat'], model_settings['emb_size'])
        return Linear_softmax_ce_head(backbone, model_settings)

    elif model == 'Resnet50' and metric == 'AM_normfree_softmax_anneal_ce_head':
        backbone = ResNets_std.Resnet50(model_settings['in_feat'], model_settings['emb_size'])
        return ResNets_std.AM_normfree_softmax_anneal_ce_head(backbone, model_settings)

    elif model == 'ResNet50_SAP_T' and metric == 'AM_normfree_softmax_anneal_ce_head':
        backbone = Resnet50.ResNet50_SAP_T(model_settings['in_feat'], model_settings['emb_size'])
        return Resnet50.AM_normfree_softmax_anneal_ce_head(backbone, model_settings)

    elif model == 'Resnet34_SAP' and metric == 'AM_normfree_softmax_anneal_ce_head':
        backbone = ResNets_std.Resnet34_SAP(model_settings['in_feat'], model_settings['emb_size'])
        return ResNets_std.AM_normfree_softmax_anneal_ce_head(backbone, model_settings)

    elif model == 'ResNet_ASV' and metric == 'Resnet_ASV_large_margin_annealing':
        backbone = ResNet_ASV()
        return Resnet_ASV_large_margin_annealing(backbone, model_settings)

    elif model == 'Resnet34_SAP_XMU' and metric == 'AM_normfree_softmax_anneal_ce_head':
        backbone = Resnet_XMU.ResNet_ASV()
        return Resnet_XMU.AM_normfree_softmax_anneal_ce_head(backbone, model_settings)

    elif model == 'Resnet34' and metric == 'AM_normfree_softmax_anneal_ce_head':
        backbone = ResNets_std.Resnet34(model_settings['in_feat'], model_settings['emb_size'])
        return ResNets_std.AM_normfree_softmax_anneal_ce_head(backbone, model_settings)

    elif model == 'Resnet18' and metric == 'AM_normfree_softmax_anneal_ce_head':
        backbone = ResNets_std.Resnet18(model_settings['in_feat'], model_settings['emb_size'])
        return ResNets_std.AM_normfree_softmax_anneal_ce_head(backbone, model_settings)

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
