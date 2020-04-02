from models import metrics
from models import ResNets, Xvector
from models import AM_NoiseT
# from models.Efficient_net.model import EfficientNet
from models.E_tdnn import Standard_ETDNN
from models.E_tdnn_plus import Res_Big_ETDNN
import sys
import torch

def get_model(model, metric, num_classes, model_settings, opt):
    print('Using new training model bank')
    if model == 'Resnet18' and metric == 'Linear_softmax_ce_head':
        backbone = ResNets.Resnet18(model_settings['in_feat'], model_settings['emb_size'])
        return ResNets.Linear_softmax_ce_head(backbone, model_settings)
    elif model == 'SeResnet18' and metric == 'Linear_softmax_ce_head':
        backbone = ResNets.SeResnet18(model_settings['in_feat'], model_settings['emb_size'])
        return ResNets.Linear_softmax_ce_head(backbone, model_settings)
    elif model == 'IBNResnet18' and metric == 'Linear_softmax_ce_head':
        backbone = ResNets.IBNResnet18(model_settings['in_feat'], model_settings['emb_size'])
        return ResNets.Linear_softmax_ce_head(backbone, model_settings)
    elif model == 'Resnet18' and metric == 'A_softmax_ce_head':
        backbone = ResNets.Resnet18(model_settings['in_feat'], model_settings['emb_size'])
        return ResNets.A_softmax_ce_head(backbone, model_settings)
    elif model == 'Resnet18' and metric == 'AAM_normfree_softmax_anneal_ce_head':
        backbone = ResNets.Resnet18(model_settings['in_feat'], model_settings['emb_size'])
        return ResNets.AAM_normfree_softmax_anneal_ce_head(backbone, model_settings)

    elif model == 'Resnet18' and metric == 'AAM_normfree_softmax_anneal_inter_ce_head':
        backbone = ResNets.Resnet18(model_settings['in_feat'], model_settings['emb_size'])
        return ResNets.AAM_normfree_softmax_anneal_inter_ce_head(backbone, model_settings)

    elif model == 'Resnet18' and metric == 'AAM_normfree_softmax_anneal_ce_head_finetune_inter':
        backbone = ResNets.Resnet18(model_settings['in_feat'], model_settings['emb_size'])
        model = ResNets.AAM_normfree_softmax_anneal_inter_ce_head(backbone, model_settings)
        model_load_path = '/Lun2/rzz/kaldi-master/egs/zhiyong/torch/train_process/train_exp_new/resnet18_aam/ckpt/9.model'
        checkpoint = torch.load(model_load_path, map_location='cpu')
        model.load_state_dict(checkpoint['model'], strict=True)
        return model

    elif model == 'Resnet18' and metric == 'AAM_softmax_anneal_ce_head':
        backbone = ResNets.Resnet18(model_settings['in_feat'], model_settings['emb_size'])
        return ResNets.AAM_softmax_anneal_ce_head(backbone, model_settings)

    elif model == 'Resnet18' and metric == 'AM_normfree_softmax_anneal_ce_head':
        backbone = ResNets.Resnet18(model_settings['in_feat'], model_settings['emb_size'])
        return ResNets.AM_normfree_softmax_anneal_ce_head(backbone, model_settings)

    elif model == 'Resnet18' and metric == 'AM_normfree_NT':
        backbone = ResNets.Resnet18(model_settings['in_feat'], model_settings['emb_size'])
        return AM_NoiseT.AM_normfree_NT(backbone, model_settings)
    
    elif model == 'Resnet18' and metric == 'AM_normfree_softmax_anneal_ce_head_finetune_inter':
        backbone = ResNets.Resnet18(model_settings['in_feat'], model_settings['emb_size'])
        model = ResNets.AM_normfree_softmax_anneal_inter_ce_head(backbone, model_settings)
        model_load_path = '/Lun2/rzz/kaldi-master/egs/zhiyong/torch/train_process/train_exp_new/resnet18_am/ckpt/10.model'
        checkpoint = torch.load(model_load_path, map_location='cpu')
        model.load_state_dict(checkpoint['model'], strict=True)
        return model

    elif model == 'Resnet18' and metric == 'A_softmax_ce_head_finetune_inter':
        backbone = ResNets.Resnet18(model_settings['in_feat'], model_settings['emb_size'])
        model = ResNets.A_softmax_inter_ce_head(backbone, model_settings)
        # model_load_path = '/Lun2/rzz/kaldi-master/egs/zhiyong/torch/train_process/train_exp_new/resnet18_a_2_compete/ckpt/min_eer.model'
        model_load_path = '/Lun2/rzz/kaldi-master/egs/zhiyong/torch/train_process/train_exp_new/resnet18_a_4_again/ckpt/10.model'

        checkpoint = torch.load(model_load_path, map_location='cpu')
        model.load_state_dict(checkpoint['model'], strict=True)
        return model

    elif model == 'Resnet18' and metric == 'AM_normfree_softmax_anneal_inter_ce_head':
        backbone = ResNets.Resnet18(model_settings['in_feat'], model_settings['emb_size'])
        return ResNets.AM_normfree_softmax_anneal_inter_ce_head(backbone, model_settings)

    elif model == 'Resnet18' and metric == 'AM_normfree_softmax_anneal_MHE_ce_head':
        backbone = ResNets.Resnet18(model_settings['in_feat'], model_settings['emb_size'])
        return ResNets.AM_normfree_softmax_anneal_MHE_ce_head(backbone, model_settings, opt)

    elif model == 'Resnet18' and metric == 'AM_softmax_anneal_ce_head':
        backbone = ResNets.Resnet18(model_settings['in_feat'], model_settings['emb_size'])
        return ResNets.AM_softmax_anneal_ce_head(backbone, model_settings)

    elif model == 'Resnet18' and metric == 'A_softmax_mixup_ce_head':
        backbone = ResNets.Resnet18(model_settings['in_feat'], model_settings['emb_size'])
        return ResNets.A_softmax_mixup_ce_head(backbone, model_settings)
    
    elif model == 'Resnet18' and metric == 'A_softmax_mixup_inter_ce_head':
        backbone = ResNets.Resnet18(model_settings['in_feat'], model_settings['emb_size'])
        return ResNets.A_softmax_mixup_inter_ce_head(backbone, model_settings)

    elif model == 'Resnet18' and metric == 'A_softmax_inter_ce_head':
        backbone = ResNets.Resnet18(model_settings['in_feat'], model_settings['emb_size'])
        return ResNets.A_softmax_inter_ce_head(backbone, model_settings)

    elif model == 'Resnet18' and metric == 'A_softmax_inter_a_ce_head':
        backbone = ResNets.Resnet18(model_settings['in_feat'], model_settings['emb_size'])
        return ResNets.A_softmax_inter_a_ce_head(backbone, model_settings)

    elif model == 'Resnet18' and metric == 'A_softmax_MHE_ce_head':
        backbone = ResNets.Resnet18(model_settings['in_feat'], model_settings['emb_size'])
        return ResNets.A_softmax_MHE_ce_head(backbone, model_settings)

    elif model == 'Resnet18_Maxpool' and metric == 'A_softmax_ce_head':
        backbone = ResNets.Resnet18_Maxpool(model_settings['in_feat'], model_settings['emb_size'])
        return ResNets.A_softmax_ce_head(backbone, model_settings)
    elif model == 'Xvector_SAP' and metric == 'Linear_softmax_ce_head':
        backbone = Xvector.Xvector_SAP(model_settings['in_feat'], model_settings['emb_size'])
        return Xvector.Linear_softmax_ce_head(backbone, model_settings)
    elif model == 'Xvector_SAP' and metric == 'A_softmax_ce_head':
        backbone = Xvector.Xvector_SAP(model_settings['in_feat'], model_settings['emb_size'])
        return Xvector.A_softmax_ce_head(backbone, model_settings)
    
    elif model == 'Xvector_SAP' and metric == 'AM_normfree_softmax_anneal_ce_head':
        backbone = Xvector.Xvector_SAP(model_settings['in_feat'], model_settings['emb_size'])
        return Xvector.AM_normfree_softmax_anneal_ce_head(backbone, model_settings)

    elif model == 'Xvector_SAP_1L' and metric == 'AM_normfree_softmax_anneal_ce_head':
        backbone = Xvector.Xvector_SAP_1L(model_settings['in_feat'], model_settings['emb_size'])
        return Xvector.AM_normfree_softmax_anneal_ce_head(backbone, model_settings)
    
    elif model == 'Xvector_SAP_nodilate_1L' and metric == 'AM_normfree_softmax_anneal_ce_head':
        backbone = Xvector.Xvector_SAP_nodilate_1L(model_settings['in_feat'], model_settings['emb_size'])
        return Xvector.AM_normfree_softmax_anneal_ce_head(backbone, model_settings)

    elif model == 'Xvector_nodilate_1L_selfatt' and metric == 'AM_normfree_softmax_anneal_ce_head':
        backbone = Xvector.Xvector_nodilate_1L_selfatt(model_settings['in_feat'], model_settings['emb_size'])
        return Xvector.AM_normfree_softmax_anneal_ce_head(backbone, model_settings)

    elif model == 'Xvector_1L_selfatt' and metric == 'AM_normfree_softmax_anneal_ce_head':
        backbone = Xvector.Xvector_1L_selfatt(model_settings['in_feat'], model_settings['emb_size'])
        return Xvector.AM_normfree_softmax_anneal_ce_head(backbone, model_settings)

    elif model == 'Xvector_nodilate_selfatt' and metric == 'AM_normfree_softmax_anneal_ce_head':
        backbone = Xvector.Xvector_nodilate_selfatt(model_settings['in_feat'], model_settings['emb_size'])
        return Xvector.AM_normfree_softmax_anneal_ce_head(backbone, model_settings)

    elif model == 'Xvector_SAP_nodilate_1L' and metric == 'AM_softmax_anneal_ce_head':
        backbone = Xvector.Xvector_SAP_nodilate_1L(model_settings['in_feat'], model_settings['emb_size'])
        return Xvector.AM_softmax_anneal_ce_head(backbone, model_settings)

    elif model == 'Xvector_SAP_nodilate' and metric == 'AM_normfree_softmax_anneal_ce_head':
        backbone = Xvector.Xvector_SAP_nodilate(model_settings['in_feat'], model_settings['emb_size'])
        return Xvector.AM_normfree_softmax_anneal_ce_head(backbone, model_settings)

    elif model == 'Standard_ETDNN' and metric == 'AM_normfree_softmax_anneal_ce_head':
        backbone = Standard_ETDNN(model_settings['in_feat'], model_settings['emb_size'])
        return Xvector.AM_normfree_softmax_anneal_ce_head(backbone, model_settings)

    elif model == 'Standard_ETDNN_plus' and metric == 'AM_normfree_softmax_anneal_ce_head':
        backbone = Res_Big_ETDNN(model_settings['in_feat'], model_settings['emb_size'])
        return Xvector.AM_normfree_softmax_anneal_ce_head(backbone, model_settings)

    elif model == 'Xvector_SAP' and metric == 'AM_softmax_anneal_ce_head':
        backbone = Xvector.Xvector_SAP(model_settings['in_feat'], model_settings['emb_size'])
        return Xvector.AM_softmax_anneal_ce_head(backbone, model_settings)

    elif model == 'Xvector_SAP' and metric == 'AM_normfree_softmax_anneal_inter_ce_head':
        backbone = Xvector.Xvector_SAP(model_settings['in_feat'], model_settings['emb_size'])
        return Xvector.AM_normfree_softmax_anneal_inter_ce_head(backbone, model_settings)        

    # elif model == 'EfficientNet-b0' and metric == 'Linear_softmax_ce_head':
    #     backbone = EfficientNet.from_name('efficientnet-b0')
    #     return ResNets.Linear_softmax_ce_head(backbone, model_settings)
    else:
        print('Invalid model or/and metric')
        sys.exit(1)
