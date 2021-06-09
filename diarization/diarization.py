import librosa
import scipy
import glob
import soundfile as sf
from scipy.io import wavfile
import librosa
import m_mfcc_vad
import sys
import os
import importlib
import torch
import numpy
import torch.nn.functional as F
from sklearn.cluster import KMeans
from scipy import signal

def baseline_ASD(inputfile, sadfile, GPU='3', code_path='/workspace/GREAT_ASV_system/train_dist', \
    model_path='/workspace/LOGS_OUTPUT/server9_nvme1/ASV_LOGS_202102/train_logs_201120/sdsv21_full_pretrain/model/model000000024.model'):
    # read wav and segment with oracle vad
    print('read wav and segment with oracle vad')
    with open(sadfile, 'r') as f:
        sad = f.readlines()
    sample_rate, raw_wav = wavfile.read(inputfile)
    
    # Resample data if not 16k
    if (sample_rate != 16000):
        number_of_samples = round(len(raw_wav) * float(16000) / sample_rate)
        raw_wav = signal.resample(raw_wav, number_of_samples)

    perspk_dict_tmpinsample = {}
    sample_len = len(raw_wav)
    start, end = None, None

    for count, j in enumerate(sad):
        start = int(float(j.split('\t')[0]) * 16000)
        if end is not None:
            perspk_dict_tmpinsample[str(end)+'.'+str(start)] = 'silence'
        end = int(float(j.split('\t')[1]) * 16000)
        perspk_dict_tmpinsample[str(start)+'.'+str(min(end, sample_len))] = 'unknown'
        if end >= sample_len:
            break

    if end < sample_len:
        perspk_dict_tmpinsample[str(end)+'.'+str(sample_len)] = 'silence'

    # segment to 1.5s
    print('segment to 1.5s')
    seg_len = int(1.5 * 16000)
    seg_hop = int(0.75 * 16000)

    perspk_dict_tmpinsample_seg = {}
    for j in perspk_dict_tmpinsample:
        if perspk_dict_tmpinsample[j] == 'unknown':
            start, end = j.split('.')
            start, end = int(start), int(end)
            if (end - start) <= seg_len:
                perspk_dict_tmpinsample_seg[str(start)+'.'+str(end)] = 'unknown'
            else:
                start_tmp = start
                end_tmp = start + seg_len
                assert end_tmp < end
                while(True):
                    if (end_tmp+seg_hop) > end:
                        end_tmp = end                    
                        perspk_dict_tmpinsample_seg[str(start_tmp)+'.'+str(end_tmp)] = 'unknown'
                        break
                    else:
                        perspk_dict_tmpinsample_seg[str(start_tmp)+'.'+str(end_tmp)] = 'unknown'
                        start_tmp = start_tmp + seg_hop
                        end_tmp = end_tmp + seg_hop
        elif perspk_dict_tmpinsample[j] == 'silence':
            start, end = j.split('.')
            start, end = int(start), int(end)
            perspk_dict_tmpinsample_seg[str(start)+'.'+str(end)] = 'silence'
        else:
            raise Exception 
    
    # extract embs
    print('extract embs')
    sys.path.append(code_path)
    os.environ['CUDA_VISIBLE_DEVICES'] = GPU
    SpeakerNetModel = importlib.import_module('models.'+'EPACA-TDNN').__getattribute__('MainModel')

    S = SpeakerNetModel(n_mels=40, nOut=192, spec_aug=False)

    S.cuda()
    loaded_state = torch.load(model_path, map_location="cuda:0")

    self_state = S.state_dict()

    for name, param in loaded_state['model'].items():
        origname = name
        if name not in self_state:
            name = name.replace("__S__.", "")

            if name not in self_state:
                print("#%s is not in the model."%origname)
                continue

        if self_state[name].size() != loaded_state['model'][origname].size():
            print("#Wrong parameter length: %s, model: %s, loaded: %s"%(origname, self_state[name].size(), loaded_state[origname].size()))
            continue

        self_state[name].copy_(param)
    
    S.eval()

    def audio2emb(audio):
        audiosize = audio.shape[0]
        max_audio = seg_len
        if audiosize <= max_audio:
            shortage    = max_audio - audiosize + 1 
            audio       = numpy.pad(audio, (0, shortage), 'wrap')
        audio = audio[None, :].astype(numpy.float)

        inp1 = torch.FloatTensor(audio).cuda()

        ref_feat = S.forward(inp1).detach().cpu()
        return ref_feat
    
    perspk_matind2time = {}
    count = 0
    for j in perspk_dict_tmpinsample_seg:
        if perspk_dict_tmpinsample_seg[j] == 'unknown':
            perspk_matind2time[count] = j
            count += 1

    matind2time = perspk_matind2time
    whole_wav = raw_wav
    embs = None
    for j in range(len(matind2time)):
        start, end = matind2time[j].split('.')
        start, end = int(start), int(end)
        if embs is not None:
            embs = torch.cat([embs, audio2emb(whole_wav[start:end])], dim=0)
        else:
            embs = audio2emb(whole_wav[start:end])

    embs_n = F.normalize(embs, p=2, dim=1)
    similarity = embs_n.mm(embs_n.T)
    similarity = similarity.numpy()

    # SC
    print('SC')
    def SC(inp_matrix):
        # 1 . Construct S and set diagonal elements to 0.
        for i in range(inp_matrix.shape[0]):
            inp_matrix[i][i] = 0.0
        # 2 . Compute Laplacian matrix L and perform normalization
        S = inp_matrix
        D = numpy.diag(numpy.sum(S, axis=1))
        L = D - S
        invsqrtD = numpy.diag(1.0 / (numpy.sum(S, axis=1) ** (0.5)))
        L_norm = numpy.dot(numpy.dot(invsqrtD, L), invsqrtD)
        # 3 . Compute eigenvalues and eigenvectors of L norm
        lam, H = numpy.linalg.eig(L_norm)
        
        # ADDED!
        try:
            lam = lam.real
            H = H.real
        except:
            pass
        
        # 4 . Take the k smallest eigenvalues
        TH = 0.99
        
        sort_index = numpy.argsort(lam)
        eig_vecs = []
        count = 0
        for i in sort_index:
            if (lam[i] > TH) and (count >= 2):
                break
            else:
                eig_vecs.append(H[:, i])
                count += 1

        for count, i in enumerate(eig_vecs):
            if count == 0:
                P = i[:, None]
            else:
                P = numpy.append(P, i[:, None], axis=1)
        # 5. Cluster row vectors y 1 , y 2 , ...y n of P to k classes by the K-means algorithm.            
        sp_kmeans = KMeans(n_clusters=P.shape[1]).fit(P)
        return sp_kmeans.labels_

    inp_matrix = (similarity + 1.0) / 2.0
    spk_clustering_result = SC(inp_matrix)

    # combine detection result
    print('combine detection result')
    tmpinsample2 = perspk_dict_tmpinsample_seg
    matind2time = perspk_matind2time
    clustering_result = spk_clustering_result

    assert len(matind2time) == len(clustering_result)
    for j in matind2time:
        assert matind2time[j] in tmpinsample2.keys()
        tmpinsample2[matind2time[j]] = 'speaker'+str(clustering_result[j]+1)
    for k in tmpinsample2:
        assert tmpinsample2[k] != 'unknown'

    whole_start = list(tmpinsample2.keys())[0].split('.')[0]
    if whole_start != '0':
        tmpinsample2['0.'+whole_start] = 'silence'
    
    output = []
    for i in tmpinsample2:
        line = str(float(i.split('.')[0])/16000)+' '+str(float(i.split('.')[1])/16000)+' '+tmpinsample2[i]+'/n'
        output.append(line)
    
    return output

if __name__ == '__main__':
    print(baseline_ASD('test.wav', 'sad.lab'))
