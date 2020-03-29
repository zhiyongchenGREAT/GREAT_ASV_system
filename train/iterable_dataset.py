import librosa
import numpy as np
import torch
import pickle
import random

fft = librosa.get_fftlib()
class VoxIterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, data_dir_dict, data_len_dict, config):        
        with open(data_dir_dict['spk2utt_train_dict'], 'rb') as handle:
            self.spk2utt_train_dict = pickle.load(handle)
        with open(data_dir_dict['music_dict'], 'rb') as handle:
            self.music_dict = pickle.load(handle)
        with open(data_dir_dict['noise_dict'], 'rb') as handle:
            self.noise_dict = pickle.load(handle)
        with open(data_dir_dict['babble_dict'], 'rb') as handle:
            self.babble_dict = pickle.load(handle)
        with open(data_dir_dict['rir_dict'], 'rb') as handle:
            self.rir_dict = pickle.load(handle)
            
        with open(data_len_dict['spk2utt_train_len'], 'rb') as handle:
            self.spk2utt_train_len = pickle.load(handle)
        with open(data_len_dict['music_len'], 'rb') as handle:
            self.music_len = pickle.load(handle)
        with open(data_len_dict['noise_len'], 'rb') as handle:
            self.noise_len = pickle.load(handle)
        with open(data_len_dict['babble_len'], 'rb') as handle:
            self.babble_len = pickle.load(handle)
        
        
        self.random_spkrs_batchlist = None
        self.ramdom_batch_len = None
        self.random_noise_type = None
        
        
        self.possible_babble_num = [3, 4, 5, 6, 7]
        self.possible_babble_snr = [13, 15, 17, 20]
        self.possible_noise_snr = [0, 5, 10, 15]
        self.possible_music_snr = [5, 8, 10, 15]
        
        self.sr = config['sr']
        self.repeats = config['repeats']
        self.batch_size = config['batch_size']
        self.extended_prefectch = config['extended_prefectch']
        
        self.mfcc_dim = 30
        
        # Auxiliary paras
        self.multi_read_count = 0
        self.preload_mem = False
        
        
        
    def __iter__(self):
        return self
    
    def __next__(self):
        assert len(self.ramdom_batch_len) == len(self.random_spkrs_batchlist)
        try:
            batch_frame_len = self.ramdom_batch_len.pop(0)
            batch_spkrs = self.random_spkrs_batchlist.pop(0)
            batch_noise_type = self.random_noise_type.pop(0)
            batched_feats = np.zeros([self.batch_size, batch_frame_len, self.mfcc_dim])
            batched_labels = np.zeros(self.batch_size)
            
            for batch_index, (spkr, noise_type) in enumerate(zip(batch_spkrs, batch_noise_type)):
                
                concat_wav, VAD_result = self._colleting_and_slicing(spkr, batch_frame_len,\
                hop_len=160, extended_prefectch=self.extended_prefectch)
            
                
                if noise_type == 0:
                    aug_wav = concat_wav
                
                elif noise_type == 1:
                    aug_wav = self._add_rebverb(concat_wav)
                   
                elif noise_type == 2:
                    aug_wav = self._add_noise(concat_wav)
                    
                elif noise_type == 3:
                    aug_wav = self._add_music(concat_wav)
                  
                elif noise_type == 4:
                    aug_wav = self._add_babble(concat_wav)
             
                else:
                    raise NotImplementedError
                    
            
                single_feats = librosa.feature.mfcc(y=aug_wav, sr=self.sr, n_mfcc=30, \
                dct_type=2, n_fft=512, hop_length=160, \
                win_length=None, window='hann', power=2.0, \
                center=True, pad_mode='reflect', n_mels=30, \
                fmin=20, fmax=7600)
                # Note single_feats needs transpose
                out_feats = self._CMVN(single_feats.T, cmn_window = 300, normalize_variance = False)
                # Apply VAD
                assert out_feats.shape[0] == VAD_result.shape[0]
                out_feats = out_feats[VAD_result.astype(np.bool)]
                batched_feats[batch_index] = out_feats[:batch_frame_len]
                batched_labels[batch_index] = spkr
                
            return batched_feats, batched_labels
        
        except IndexError:
            raise StopIteration

    def process_one_utt(self, utt_dir):
        try:
            concat_wav, _ = librosa.load(utt_dir, sr=self.sr)
            
            VAD_result = self._VAD_detection(concat_wav)
            
            aug_wav = concat_wav

            single_feats = librosa.feature.mfcc(y=aug_wav, sr=self.sr, n_mfcc=30, \
            dct_type=2, n_fft=512, hop_length=160, \
            win_length=None, window='hann', power=2.0, \
            center=True, pad_mode='reflect', n_mels=30, \
            fmin=20, fmax=7600)
            # Note single_feats needs transpose
            out_feats = self._CMVN(single_feats.T, cmn_window = 300, normalize_variance = False)
            # Apply VAD
            assert out_feats.shape[0] == VAD_result.shape[0]
            out_feats = out_feats[VAD_result.astype(np.bool)]
            
            batched_feats = out_feats[None, :, :]
                
            return batched_feats
        
        except Exception:
            traceback.print_exc()
    
    def noise_data_preload(self):
        print('preloading music_dict')
        for count, i in enumerate(self.music_dict):
            _, _ = librosa.load(self.music_dict[i], sr=self.sr)
            if (count+1)%100 == 0:
                print(count+1)
        print('preloading noise_dict')        
        for count, i in enumerate(self.noise_dict):
            _, _ = librosa.load(self.noise_dict[i], sr=self.sr)
            if (count+1)%100 == 0:
                print(count+1)
        print('preloading babble_dict')        
        for count, i in enumerate(self.babble_dict):
            _, _ = librosa.load(self.babble_dict[i], sr=self.sr)
            if (count+1)%100 == 0:
                print(count+1)
    
    def noise_data_preload2mem(self):
        print('preloading to memory')
        
        self.music_preload_dict = {}
        self.noise_preload_dict = {}
        self.babble_preload_dict = {}
        self.preload_mem = True
        print('preloading music_dict')
        for count, i in enumerate(self.music_dict):
            self.music_preload_dict[i], _ = librosa.load(self.music_dict[i], sr=self.sr)
            if (count+1)%100 == 0:
                print(count+1)
        print('preloading noise_dict')        
        for count, i in enumerate(self.noise_dict):
            self.noise_preload_dict[i], _ = librosa.load(self.noise_dict[i], sr=self.sr)
            if (count+1)%100 == 0:
                print(count+1)
        print('preloading babble_dict')        
        for count, i in enumerate(self.babble_dict):
            self.babble_preload_dict[i], _ = librosa.load(self.babble_dict[i], sr=self.sr)
            if (count+1)%100 == 0:
                print(count+1)       
        
        
    def get_random_list(self):
        spkrs_list = self.repeats * list(self.spk2utt_train_dict.keys())
        random.shuffle(spkrs_list)
        len_spkrs_list = len(spkrs_list)
        self.random_spkrs_batchlist = [spkrs_list[i*self.batch_size:i*self.batch_size+self.batch_size]\
        for i in range(len_spkrs_list // self.batch_size)]
        
        self.ramdom_batch_len = [random.randint(200, 400) for i in range(len_spkrs_list // self.batch_size)]
        
        noise_type_list = [i%5 for i in range(len_spkrs_list)]

        random.shuffle(noise_type_list)
        self.random_noise_type = [noise_type_list[i*self.batch_size:i*self.batch_size+self.batch_size]\
        for i in range(len_spkrs_list // self.batch_size)]
        
        assert len(self.random_spkrs_batchlist) == len(self.ramdom_batch_len)\
        == len(self.random_noise_type)
        
    def _colleting_and_slicing(self, spkr, batch_frame_len, hop_len=160, extended_prefectch=2.0):
        
        least_wav_len = (batch_frame_len - 1) * hop_len
        concat_utt = np.zeros(0)
        valid_frames_len = 0
        
        # Use to count multi_read_count
        get_count = 0

        while valid_frames_len < batch_frame_len:
            utt_dir = self._get_random_spk_utt(spkr, self.spk2utt_train_dict)
            utt_len = self.spk2utt_train_len[utt_dir]
#             off = self._get_random_offset(least_wav_len, utt_len) / self.sr
            off = self._get_random_offset(least_wav_len+extended_prefectch*self.sr, utt_len) / self.sr
            dur = least_wav_len / self.sr + extended_prefectch
            
            utt_part, _ = librosa.load(utt_dir, sr=self.sr, offset=off, duration=dur)
            
            concat_utt = np.append(concat_utt, utt_part)
            detected_frames = self._VAD_detection(concat_utt)
            valid_frames_len = np.sum(detected_frames)

            get_count += 1

        if get_count > 1:
            self.multi_read_count += 1

        VAD_result = detected_frames
        return concat_utt, VAD_result
    
    def _add_rebverb(self, in_wav):
        power_before_reverb = in_wav.dot(in_wav) / len(in_wav)
        shift_index = 0
        signal = in_wav
        filter_dir = self._get_random_noise(self.rir_dict)
        filter, _ = librosa.load(filter_dir, sr=self.sr)
        
        signal_length = len(signal)
        filter_length = len(filter)
        output_length = signal_length + filter_length - 1
        output = np.zeros(output_length)

        fft_length = 2**np.ceil(np.log2(4 * filter_length)).astype(np.int)
        block_length = fft_length - filter_length + 1


        filter_padded = np.zeros(fft_length)
        filter_padded[0:filter_length] = filter
        filter_padded = fft.rfft(filter_padded)



        for i in range(signal_length//block_length + 1):
            process_length = min(block_length, signal_length - i * block_length);
            signal_block_padded = np.zeros(fft_length)
            signal_block_padded[0:process_length] = signal[i * block_length : i * block_length + process_length]
            signal_block_padded = fft.rfft(signal_block_padded)

            signal_block_padded = filter_padded * signal_block_padded

            signal_block_padded = fft.irfft(signal_block_padded, n=fft_length)

            if (i*block_length + fft_length) <= output_length:
                output[i*block_length : i*block_length + fft_length] += signal_block_padded
            else:
                output[i*block_length : output_length] += signal_block_padded[:output_length-i*block_length]
        
        # shift with max index of filter
        shift_index = np.argmax(filter)
        
        final_out = output[shift_index:shift_index+signal_length]
        power_after_reverb = final_out.dot(final_out) / len(final_out)
        final_out = np.sqrt(power_before_reverb/power_after_reverb) * final_out
        out_wav = final_out
        
        return out_wav
    
    def _add_noise(self, in_wav):
        power_before_reverb = in_wav.dot(in_wav) / len(in_wav)
        shift_index = 0
        signal = in_wav
        
        signal_len = len(signal)
        total_noise_len = 0
        signal_off = 0
        while total_noise_len < signal_len:
            
            noise_dir, noise_index = self._get_random_noise(self.noise_dict, return_index=True)
            noise_len = self.noise_len[noise_index]
            if noise_len > signal_len:
                noise_off = self._get_random_offset(signal_len, noise_len)
                total_noise_len += signal_len
                if self.preload_mem:
                    noise = self.noise_preload_dict[noise_index][noise_off:noise_off+signal_len]
                else:
                    noise, _ = librosa.load(noise_dir, sr=self.sr, offset=noise_off/self.sr,\
                    duration=signal_len/self.sr)
                
            else:
                total_noise_len += noise_len
                if self.preload_mem:
                    noise = self.noise_preload_dict[noise_index]
                else:
                    noise, _ = librosa.load(noise_dir, sr=self.sr)
                
            snr_db = self.possible_noise_snr[random.randint(0, len(self.possible_noise_snr)-1)]
        
            signal = self._add_db(signal, noise, signal_off, snr_db, power_before_reverb)
            
            signal_off += len(noise)
        
        output = signal
        final_out = output[shift_index:shift_index+signal_len]
        power_after_reverb = final_out.dot(final_out) / len(final_out)
        final_out = np.sqrt(power_before_reverb/power_after_reverb) * final_out
        out_wav = final_out
        
        return out_wav
    
    def _add_music(self, in_wav):
        power_before_reverb = in_wav.dot(in_wav) / len(in_wav)
        shift_index = 0
        signal = in_wav
        
        signal_len = len(signal)
        total_noise_len = 0
        signal_off = 0
        while total_noise_len < signal_len:
            
            noise_dir, noise_index = self._get_random_noise(self.music_dict, return_index=True)
            noise_len = self.music_len[noise_index]
            if noise_len > signal_len:
                noise_off = self._get_random_offset(signal_len, noise_len)
                total_noise_len += signal_len
                if self.preload_mem:
                    noise = self.music_preload_dict[noise_index][noise_off:noise_off+signal_len]
                else:
                    noise, _ = librosa.load(noise_dir, sr=self.sr, offset=noise_off/self.sr,\
                    duration=signal_len/self.sr)
            else:
                total_noise_len += noise_len
                if self.preload_mem:
                    noise = self.music_preload_dict[noise_index]
                else:
                    noise, _ = librosa.load(noise_dir, sr=self.sr)
                
            snr_db = self.possible_music_snr[random.randint(0, len(self.possible_music_snr)-1)]
        
            signal = self._add_db(signal, noise, signal_off, snr_db, power_before_reverb)
            
            signal_off += len(noise)
        
        output = signal
        final_out = output[shift_index:shift_index+signal_len]
        power_after_reverb = final_out.dot(final_out) / len(final_out)
        final_out = np.sqrt(power_before_reverb/power_after_reverb) * final_out
        out_wav = final_out
        
        return out_wav
    
    def _add_babble(self, in_wav):
        power_before_reverb = in_wav.dot(in_wav) / len(in_wav)
        shift_index = 0
        signal = in_wav
        
        signal_len = len(signal)
        signal_off = 0
        bg_spks_num = self.possible_babble_num[random.randint(0, len(self.possible_babble_num)-1)]    
        for _ in range(bg_spks_num):            
            noise_dir, noise_index = self._get_random_noise(self.babble_dict, return_index=True)
            noise_len = self.babble_len[noise_index]
            if noise_len > signal_len:
                noise_off = self._get_random_offset(signal_len, noise_len)
                if self.preload_mem:
                    noise = self.babble_preload_dict[noise_index][noise_off:noise_off+signal_len]
                else:
                    noise, _ = librosa.load(noise_dir, sr=self.sr, offset=noise_off/self.sr,\
                    duration=signal_len/self.sr)
            else:
                if self.preload_mem:
                    noise = self.babble_preload_dict[noise_index]
                else:
                    noise, _ = librosa.load(noise_dir, sr=self.sr)
                
            snr_db = self.possible_babble_snr[random.randint(0, len(self.possible_babble_snr)-1)]
        
            signal = self._add_db(signal, noise, signal_off, snr_db, power_before_reverb)
            
        output = signal
        final_out = output[shift_index:shift_index+signal_len]
        power_after_reverb = final_out.dot(final_out) / len(final_out)
        final_out = np.sqrt(power_before_reverb/power_after_reverb) * final_out
        out_wav = final_out
        
        return out_wav
    
    def _add_db(self, in_wav, noise, signal_off, snr_db, power_before_reverb):
        signal = in_wav

        noise_power = noise.dot(noise) / len(noise)
        scale_factor = np.sqrt(10**(-snr_db / 10) * power_before_reverb / noise_power)
        noise = scale_factor * noise

        add_length = min(len(noise), len(signal)-signal_off)
        signal[signal_off:signal_off+add_length] += noise[:add_length]
        out_wav = signal      
        
        return out_wav
    
    def _CMVN(self, in_feat, cmn_window = 300, normalize_variance = False):             
        num_frames = in_feat.shape[0]
        dim = in_feat.shape[1]
        last_window_start = -1
        last_window_end = -1
        cur_sum = np.zeros(dim)
        cur_sumsq = np.zeros(dim)

        out_feat = np.zeros([num_frames, dim])

        for t in range(num_frames):
            window_start = 0
            window_end = 0

            window_start = t - int(cmn_window / 2)
            window_end = window_start + cmn_window

            if (window_start < 0):
                window_end -= window_start
                window_start = 0

            if (window_end > num_frames):
                window_start -= (window_end - num_frames)
                window_end = num_frames
                if (window_start < 0):
                    window_start = 0

            if (last_window_start == -1):
                input_part = in_feat[window_start:window_end]
                cur_sum = np.sum(input_part, axis=0, keepdims=False)
                if normalize_variance:
                    cur_sumsq = np.sum(input_part**2, axis=0, keepdims=False)
            else:
                if (window_start > last_window_start):
                    frame_to_remove = in_feat[last_window_start]
                    cur_sum -= frame_to_remove
                    if normalize_variance:
                        cur_sumsq -= frame_to_remove**2

                if (window_end > last_window_end):
                    frame_to_add = in_feat[last_window_end]
                    cur_sum += frame_to_add
                    if normalize_variance:
                        cur_sumsq += frame_to_add**2

            window_frames = window_end - window_start
            last_window_start = window_start
            last_window_end = window_end

            out_feat[t] = in_feat[t] - (1.0 / window_frames) * cur_sum


            if normalize_variance:
                if (window_frames == 1):
                    out_feat[t] = 0.0
                else:
                    variance = (1.0 / window_frames) * cur_sumsq - (1.0 / window_frames**2) * cur_sum**2
                    variance = np.maximum(1.0e-10, variance)
                    out_feat[t] /= variance**(0.5)
                    
        return out_feat

    def _get_random_noise(self, noise_dict, return_index=False):
        dict_len = len(noise_dict)
        i = random.randint(0, dict_len-1)
        noise_dir = noise_dict[i]
        
        if return_index:
            return noise_dir, i
        else:
            return noise_dir
    
    def _get_random_spk_utt(self, spkr, spk2utt):
        this_utts = spk2utt[spkr]
        this_num_utts = len(this_utts)
        i = random.randint(0, this_num_utts-1)
        utt_dir = this_utts[i]
        return utt_dir

    def _get_random_offset(self, expected_length, utt_len):
        if expected_length > utt_len:
            return 0
        
        free_length = utt_len - expected_length
        offset = random.randint(0, free_length)
        return offset
        
    @property
    def _VAD_config(self):
        vad_energy_threshold = -3.0
        vad_energy_mean_scale = 1.0
        vad_frames_context = 0
        vad_proportion_threshold = 0.12
        
        return vad_energy_threshold, vad_energy_mean_scale,\
        vad_frames_context, vad_proportion_threshold
        
        
    def _VAD_detection(self, wav):
        vad_energy_threshold, vad_energy_mean_scale,\
        vad_frames_context, vad_proportion_threshold = self._VAD_config
        
        y_tmp = np.pad(wav, int(512 // 2), mode='reflect')
        y_tmp = librosa.util.frame(y_tmp, frame_length=512, hop_length=160)
        y_log_energy = np.log(np.maximum(np.sum(y_tmp**2, axis=0), 1e-15))

        T = len(y_log_energy)
        output_voiced = np.zeros(T)
        if (T == 0):
            raise Exception("zero wave length")

        energy_threshold = vad_energy_threshold
        if (vad_energy_mean_scale != 0.0):
            assert(vad_energy_mean_scale > 0.0)
            energy_threshold += vad_energy_mean_scale * np.sum(y_log_energy) / T


        assert(vad_frames_context >= 0)
        assert(vad_proportion_threshold > 0.0 and vad_proportion_threshold < 1.0);

        for t in range(T):
            num_count = 0
            den_count = 0
            context = vad_frames_context
            for t2 in range(t - context, t + context+1):
                if (t2 >= 0 and t2 < T):
                    den_count+=1
                    if (y_log_energy[t2] > energy_threshold):
                        num_count+=1

            if (num_count >= den_count * vad_proportion_threshold):
                output_voiced[t] = 1.0
            else:
                output_voiced[t] = 0.0
        
        return output_voiced