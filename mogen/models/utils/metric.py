import librosa
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import math
from scipy.signal import argrelextrema
from scipy import linalg


class L1div(object):
    def __init__(self):
        self.counter = 0
        self.sum = 0
    def run(self, results):
        self.counter += results.shape[0]
        mean = np.mean(results, 0)
        for i in range(results.shape[0]):
            results[i, :] = abs(results[i, :] - mean)
        sum_l1 = np.sum(results)
        self.sum += sum_l1
    def avg(self):
        return self.sum/self.counter
    def reset(self):
        self.counter = 0
        self.sum = 0
        

class SRGR(object):
    def __init__(self, threshold=0.3, joints=55):
        self.threshold = threshold
        self.pose_dimes = joints
        self.counter = 0
        self.sum = 0
        
    def run(self, results, targets, semantic):
        results = results.reshape(-1, self.pose_dimes, 3)
        targets = targets.reshape(-1, self.pose_dimes, 3)
        semantic = semantic.reshape(-1)
        diff = np.sum(abs(results-targets),2)
        success = np.where(diff<self.threshold, 1.0, 0.0)
        for i in range(success.shape[0]):
            # srgr == 0.165 when all success, scale range to [0, 1]
            success[i, :] *= semantic[i] * (1/0.165) 
        rate = np.sum(success)/(success.shape[0]*success.shape[1])
        self.counter += success.shape[0]
        self.sum += (rate*success.shape[0])
        return rate
    
    def avg(self):
        return self.sum/self.counter

class alignment(object):
    def __init__(self, sigma, order, mmae=None, upper_body=[3,6,9,12,13,14,15,16,17,18,19,20,21]):
        self.sigma = sigma
        self.order = order
        self.upper_body= upper_body
        # self.times = self.oenv = self.S = self.rms = None
        self.pose_data = []
        self.mmae = mmae
        self.threshold = 0.3
    
    def load_audio(self, wave, t_start=None, t_end=None, without_file=False, sr_audio=16000):
        hop_length = 512
        if without_file:
            y = wave
            sr = sr_audio
        else: y, sr = librosa.load(wave)
        if t_start is None:
            short_y = y
        else:
            short_y = y[t_start:t_end]
        # print(short_y.shape)
        onset_t = librosa.onset.onset_detect(y=short_y, sr=sr_audio, hop_length=hop_length, units='time')
        return onset_t

    def load_pose(self, pose, t_start, t_end, pose_fps, without_file=False):
        data_each_file = []
        if without_file:
            for line_data_np in pose: #,args.pre_frames, args.pose_length
                data_each_file.append(line_data_np)
                    #data_each_file.append(np.concatenate([line_data_np[9:18], line_data_np[75:84], ],0))
        else: 
            with open(pose, "r") as f:
                for i, line_data in enumerate(f.readlines()):
                    if i < 432: continue
                    line_data_np = np.fromstring(line_data, sep=" ",)
                    if pose_fps == 15:
                        if i % 2 == 0:
                            continue
                    data_each_file.append(np.concatenate([line_data_np[30:39], line_data_np[112:121], ],0))
                    
        data_each_file = np.array(data_each_file)
        #print(data_each_file.shape)
        
        joints = data_each_file.transpose(1, 0)
        dt = 1/pose_fps
        # first steps is forward diff (t+1 - t) / dt
        init_vel = (joints[:, 1:2] - joints[:, :1]) / dt
        # middle steps are second order (t+1 - t-1) / 2dt
        middle_vel = (joints[:, 2:] - joints[:, 0:-2]) / (2 * dt)
        # last step is backward diff (t - t-1) / dt
        final_vel = (joints[:, -1:] - joints[:, -2:-1]) / dt
        #print(joints.shape, init_vel.shape, middle_vel.shape, final_vel.shape)
        vel = np.concatenate([init_vel, middle_vel, final_vel], 1).transpose(1, 0).reshape(data_each_file.shape[0], -1, 3)
        #print(vel.shape)
        #vel = data_each_file.reshape(data_each_file.shape[0], -1, 3)[1:] - data_each_file.reshape(data_each_file.shape[0], -1, 3)[:-1]
        vel = np.linalg.norm(vel, axis=2) / self.mmae
        
        beat_vel_all = []
        for i in range(vel.shape[1]):
            vel_mask = np.where(vel[:, i]>self.threshold)
            #print(vel.shape)
            #t_end = 80
            #vel[::2, :] -= 0.000001
            #print(vel[t_start:t_end, i], vel[t_start:t_end, i].shape)
            beat_vel = argrelextrema(vel[t_start:t_end, i], np.less, order=self.order) # n*47
            #print(beat_vel, t_start, t_end)
            beat_vel_list = []
            for j in beat_vel[0]:
                if j in vel_mask[0]:
                    beat_vel_list.append(j)
            beat_vel = np.array(beat_vel_list)
            beat_vel_all.append(beat_vel)
        #print(beat_vel_all)
        return beat_vel_all #beat_right_arm, beat_right_shoulder, beat_right_wrist, beat_left_arm, beat_left_shoulder, beat_left_wrist
    
    
    def load_data(self, wave, pose, t_start, t_end, pose_fps):
        onset_raw, onset_bt, onset_bt_rms = self.load_audio(wave, t_start, t_end)
        beat_right_arm, beat_right_shoulder, beat_right_wrist, beat_left_arm, beat_left_shoulder, beat_left_wrist = self.load_pose(pose, t_start, t_end, pose_fps)
        return onset_raw, onset_bt, onset_bt_rms, beat_right_arm, beat_right_shoulder, beat_right_wrist, beat_left_arm, beat_left_shoulder, beat_left_wrist 

    def eval_random_pose(self, wave, pose, t_start, t_end, pose_fps, num_random=60):
        onset_raw, onset_bt, onset_bt_rms = self.load_audio(wave, t_start, t_end)
        dur = t_end - t_start
        for i in range(num_random):
            beat_right_arm, beat_right_shoulder, beat_right_wrist, beat_left_arm, beat_left_shoulder, beat_left_wrist = self.load_pose(pose, i, i+dur, pose_fps)
            dis_all_b2a= self.calculate_align(onset_raw, onset_bt, onset_bt_rms, beat_right_arm, beat_right_shoulder, beat_right_wrist, beat_left_arm, beat_left_shoulder, beat_left_wrist)
            print(f"{i}s: ",dis_all_b2a)


    @staticmethod
    def plot_onsets(audio, sr, onset_times_1, onset_times_2):
        import librosa
        import librosa.display
        import matplotlib.pyplot as plt
        # Plot audio waveform
        fig, axarr = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
        
        # Plot audio waveform in both subplots
        librosa.display.waveshow(audio, sr=sr, alpha=0.7, ax=axarr[0])
        librosa.display.waveshow(audio, sr=sr, alpha=0.7, ax=axarr[1])
        
        # Plot onsets from first method on the first subplot
        for onset in onset_times_1:
            axarr[0].axvline(onset, color='r', linestyle='--', alpha=0.9, label='Onset Method 1')
        axarr[0].legend()
        axarr[0].set(title='Onset Method 1', xlabel='', ylabel='Amplitude')
        
        # Plot onsets from second method on the second subplot
        for onset in onset_times_2:
            axarr[1].axvline(onset, color='b', linestyle='-', alpha=0.7, label='Onset Method 2')
        axarr[1].legend()
        axarr[1].set(title='Onset Method 2', xlabel='Time (s)', ylabel='Amplitude')
    
        
        # Add legend (eliminate duplicate labels)
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())
        
        # Show plot
        plt.title("Audio waveform with Onsets")
        plt.savefig("./onset.png", dpi=500)
    
    def audio_beat_vis(self, onset_raw, onset_bt, onset_bt_rms):
        import librosa.display
        figure(figsize=(24, 6), dpi=80)
        fig, ax = plt.subplots(nrows=4, sharex=True)
        librosa.display.specshow(librosa.amplitude_to_db(self.S, ref=np.max),
                                y_axis='log', x_axis='time', ax=ax[0])
        ax[0].label_outer()
        ax[1].plot(self.times, self.oenv, label='Onset strength')
        ax[1].vlines(librosa.frames_to_time(onset_raw), 0, self.oenv.max(), label='Raw onsets', color='r')
        ax[1].legend()
        ax[1].label_outer()

        ax[2].plot(self.times, self.oenv, label='Onset strength')
        ax[2].vlines(librosa.frames_to_time(onset_bt), 0, self.oenv.max(), label='Backtracked', color='r')
        ax[2].legend()
        ax[2].label_outer()

        ax[3].plot(self.times, self.rms[0], label='RMS')
        ax[3].vlines(librosa.frames_to_time(onset_bt_rms), 0, self.oenv.max(), label='Backtracked (RMS)', color='r')
        ax[3].legend()
        fig.savefig("./onset.png", dpi=500)
    
    @staticmethod
    def motion_frames2time(vel, offset, pose_fps):
        time_vel = vel/pose_fps + offset 
        return time_vel    
    
    @staticmethod
    def GAHR(a, b, sigma):
        dis_all_a2b = 0
        dis_all_b2a = 0
        for b_each in b:
            l2_min = np.inf
            for a_each in a:
                l2_dis = abs(a_each - b_each)
                if l2_dis < l2_min:
                    l2_min = l2_dis
            dis_all_b2a += math.exp(-(l2_min**2)/(2*sigma**2))
        dis_all_b2a /= len(b)
        return dis_all_b2a 
    
    @staticmethod
    def fix_directed_GAHR(a, b, sigma):
        a = alignment.motion_frames2time(a, 0, 30)
        b = alignment.motion_frames2time(b, 0, 30)
        t = len(a)/30
        a = [0] + a + [t]
        b = [0] + b + [t]
        dis_a2b = alignment.GAHR(a, b, sigma)
        return dis_a2b

    def calculate_align(self, onset_bt_rms, beat_vel, pose_fps=30):
        audio_bt = onset_bt_rms
        avg_dis_all_b2a_list = []
        for its, beat_vel_each in enumerate(beat_vel):
            if its not in self.upper_body:
                continue
            #print(beat_vel_each)
            #print(audio_bt.shape, beat_vel_each.shape)
            pose_bt = self.motion_frames2time(beat_vel_each, 0, pose_fps)
            #print(pose_bt)
            avg_dis_all_b2a_list.append(self.GAHR(pose_bt, audio_bt, self.sigma))
        # avg_dis_all_b2a = max(avg_dis_all_b2a_list)
        avg_dis_all_b2a = sum(avg_dis_all_b2a_list)/len(avg_dis_all_b2a_list) #max(avg_dis_all_b2a_list)
        #print(avg_dis_all_b2a, sum(avg_dis_all_b2a_list)/47)
        return avg_dis_all_b2a  
    

class FIDCalculator(object):
    """
    Wrapper class for calculating Frechet Inception Distance (FID) between two sets of samples.
    """
    def __init__(self):
        pass

    @staticmethod
    def frechet_distance(samples_A, samples_B):
        A_mu = np.mean(samples_A, axis=0)
        A_sigma = np.cov(samples_A, rowvar=False)
        B_mu = np.mean(samples_B, axis=0)
        B_sigma = np.cov(samples_B, rowvar=False)
        try:
            frechet_dist = FIDCalculator.calculate_frechet_distance(A_mu, A_sigma, B_mu, B_sigma)
        except ValueError:
            frechet_dist = 1e+10
        return frechet_dist


    @staticmethod
    def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
        """ from https://github.com/mseitzer/pytorch-fid/blob/master/fid_score.py 
        Numpy implementation of the Frechet Distance.
        The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
        and X_2 ~ N(mu_2, C_2) is
                d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
        Stable version by Dougal J. Sutherland.
        Params:
        -- mu1   : Numpy array containing the activations of a layer of the
                    inception net (like returned by the function 'get_predictions')
                    for generated samples.
        -- mu2   : The sample mean over activations, precalculated on an
                    representative data set.
        -- sigma1: The covariance matrix over activations for generated samples.
        -- sigma2: The covariance matrix over activations, precalculated on an
                    representative data set.
        Returns:
        --   : The Frechet Distance.
        """

        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)
        #print(mu1[0], mu2[0])
        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)
        #print(sigma1[0], sigma2[0])
        assert mu1.shape == mu2.shape, \
            'Training and test mean vectors have different lengths'
        assert sigma1.shape == sigma2.shape, \
            'Training and test covariances have different dimensions'

        diff = mu1 - mu2

        # Product might be almost singular
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        #print(diff, covmean[0])
        if not np.isfinite(covmean).all():
            msg = ('fid calculation produces singular product; '
                    'adding %s to diagonal of cov estimates') % eps
            print(msg)
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        # Numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError('Imaginary component {}'.format(m))
            covmean = covmean.real

        tr_covmean = np.trace(covmean)

        return (diff.dot(diff) + np.trace(sigma1) +
                np.trace(sigma2) - 2 * tr_covmean)
    


def calc_diversity(feats):
    feat_array = np.array(feats)
    n, c = feat_array.shape
    diff = np.array([feat_array] * n) - feat_array.reshape(n, 1, c)
    return np.sqrt(np.sum(diff**2, axis=2)).sum() / n / (n-1)

def calculate_avg_distance(feature_list, mean=None, std=None):
    feature_list = np.stack(feature_list)
    n = feature_list.shape[0]
    # normalize the scale
    if (mean is not None) and (std is not None):
        feature_list = (feature_list - mean) / std

    # breakpoint()
    dist = 0
    for i in range(n):
        for j in range(i + 1, n):
            # breakpoint()
            dist += np.linalg.norm(feature_list[i] - feature_list[j]) / feature_list[j].shape[0]
    dist /= (n * n - n) / 2
    return dist


class MPJPE:
    def __init__(self):
        self.total_error = 0.0
        self.total_joints = 0

    def compute_error(self, predicted, ground_truth, mask=None):
        """
        Compute MPJPE for a single pose or a batch of poses.

        Parameters:
        - predicted: numpy array of shape (N, J, 3) where N is the number of poses,
                     J is the number of joints, and 3 corresponds to (x, y, z) coordinates.
        - ground_truth: numpy array of shape (N, J, 3), the ground-truth joint positions.
        - mask: numpy array of shape (N, J, 1) with 0s for joints that are not visible and 1s for visible joints

        Returns:
        - mpjpe: the mean per joint position error for the given batch
        """
        # Ensure input arrays are numpy arrays
        predicted = np.asarray(predicted)
        ground_truth = np.asarray(ground_truth)

        # Compute the Euclidean distance for each joint
        error = np.linalg.norm(predicted - ground_truth, axis=-1)  # shape: (N, J)

        # Mask out joints that are not visible
        if mask is not None:
            error *= mask

        mpjpe = np.mean(error)  # Mean over all joints and all poses

        # Accumulate the total error and joint counts
        self.total_error += np.sum(error)
        self.total_joints += error.size

        return mpjpe

    def get_average_error(self):
        """
        Get the average MPJPE over all poses processed so far.

        Returns:
        - average_mpjpe: the average MPJPE across all accumulated data
        """
        if self.total_joints == 0:
            return 0.0
        return self.total_error / self.total_joints

    def reset(self):
        """
        Reset the accumulated error and joint counts.
        """
        self.total_error = 0.0
        self.total_joints = 0
