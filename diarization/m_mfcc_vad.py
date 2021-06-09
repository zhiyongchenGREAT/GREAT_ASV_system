from m_mfcc_function import *


def m_mfcc(path, alpha=0.02, k=1, ou=10):
    """
    @param path: 音频路径, 当前默认采样率 16k
    @param alpha: m_mfcc调节系数
    @param k: vad切割阈值调节系数
    @param ou: vad窗
    """
    y, sr = read_wav(path)
    time = [i / sr for i in range(y.shape[0])]

    # 计算 mfcc_n
    mfcc_n = python_speech_features.mfcc(y, sr)

    # 计算前20帧的均值向量,基本要求前 0.2s不能有语音（先验条件）
    average_vector = mfcc_n[0].copy()
    for i in range(1, 20):
        average_vector += mfcc_n[i]
    average_vector /= 20

    # 计算M_n
    frame = frames(y, sr)
    frame_abs = np.abs(frame)
    M_n = np.mean(frame_abs, axis=1)

    # 计算MFCC（语音帧与噪声帧的）余弦相似度的距离矩阵
    # d(m_noise, m_i)
    distance = []
    a = 0.97  # 动态噪声更新的调节因子
    for i in range(len(mfcc_n)):
        # 更新噪声
        m_noise_i = a * average_vector + (1 - a) * mfcc_n[i]
        # 计算余弦距离
        distance_i = np.abs((np.linalg.norm(m_noise_i) * (np.linalg.norm(mfcc_n[i])) / np.dot(m_noise_i, mfcc_n[i])))
        distance.append(distance_i)

    # 卷积平滑distance
    distance_conv = signal.convolve(distance, np.array(
        [1, 1, 1, 1, 1, 1, 1]
    ))[3:-3]

    # M-MFCC均衡
    # 均衡因子p
    p = alpha / np.average(M_n) if np.average(M_n) > alpha else 1

    M_MFCC = (1 - p) * (distance_conv / max(distance_conv)) + p * (M_n / max(M_n))

    # 切割起止点
    # 阈值
    # 用户可调节阈值调节因子
    Threshold = np.average(M_MFCC) * k
    out = np.zeros_like(M_MFCC)
    for i in range(ou, len(M_MFCC) - ou):
        out[i] = 1 if np.average(M_MFCC[i - ou:i + ou]) > Threshold else 0

    res = []
    start_flag = 0
    end_flag = 0
    start = 0.
    end = 0.
    for i in range(1, len(out)):
        if start_flag == 0 and end_flag == 0:
            if out[i] == 1 and out[i - 1] == 0:
                # 起始点
                start = (i-1) / 100
                start_flag = 1
        elif start_flag == 1 and end_flag == 0:
            if out[i] == 0 and out[i - 1] == 1:
                # 终止点
                end = i / 100
                end_flag = 1
        elif start_flag == 1 and end_flag == 1:
            start_flag = 0
            end_flag = 0
            res.append({
                "start": start,
                "end": end
            })
    return res
