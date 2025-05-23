import numpy as np
import torch
import math
import smplx
from attrdict import AttrDict
from collections import defaultdict
import json

# import torch.distributed as dist
import lmdb
import pyarrow
import shutil
from termcolor import colored

# from rich.progress import track
from torch.utils import data
from tqdm import tqdm
import glob
from scipy.interpolate import interp1d
import librosa
import pandas as pd
import os
import re
import copy
import random
from .utils.disco_utils import parse_discourse_tokens, parse_discourse_relations
from .utils.beatx_utils import joints_list
# from .utils.quaternion import qbetween_np, qrot_np
from .builder import DATASETS
from mogen.utils import get_root_logger


import time
import soundfile as sf

# ignore warnings
import warnings
# warnings.simplefilter(action="ignore", category=FutureWarning)

@DATASETS.register_module()
class BEATXDataset(data.Dataset):
    def __init__(
        self,
        split,
        build_cache=True,
        tiny=False,
        debug=False,
        **kwargs,
    ):
        super(BEATXDataset, self).__init__()

        self.tiny = tiny
        self.debug = debug
        loader_type = split

        self.loader_type = loader_type
        self.args = AttrDict(kwargs)
        self.logger = get_root_logger()

        # self.rank = dist.get_rank()
        self.ori_stride = self.args.stride
        self.ori_length = self.args.pose_length

        self.ori_joint_list = joints_list["beat_smplx_joints"]
        # self.tar_joint_list = "beat_smplx_joints"
        self.face_joint_list = joints_list["beat_smplx_face"]
        self.upper_joint_list = joints_list["beat_smplx_upper"]
        self.lower_joint_list = joints_list["beat_smplx_lower"]
        self.hands_joint_list = joints_list["beat_smplx_hands"]
        self.all_tar_joint_list = list(self.face_joint_list.keys()) + list(self.upper_joint_list.keys()) + list(self.lower_joint_list.keys()) + list(self.hands_joint_list.keys())
        # breakpoint()
        assert "smplx" in self.args.pose_rep, "only support smplx"
        self.face_mask = np.zeros(len(list(self.ori_joint_list.keys())) * 3)
        self.upper_mask = np.zeros_like(self.face_mask)
        self.lower_mask = np.zeros_like(self.face_mask)
        self.hands_mask = np.zeros_like(self.face_mask)

        self.face_joints = len(list(self.face_joint_list.keys()))
        self.upper_joints = len(list(self.upper_joint_list.keys()))
        self.lower_joints = len(list(self.lower_joint_list.keys()))
        self.hands_joints = len(list(self.hands_joint_list.keys()))
        self.joints = self.face_joints + self.upper_joints + self.lower_joints + self.hands_joints

        for joint_name in self.all_tar_joint_list:
            mask_start = (
                self.ori_joint_list[joint_name][1] - self.ori_joint_list[joint_name][0]
            )
            mask_end = self.ori_joint_list[joint_name][1]

            if joint_name in self.face_joint_list:
                self.face_mask[mask_start:mask_end] = 1

            if joint_name in self.upper_joint_list:
                self.upper_mask[mask_start:mask_end] = 1

            if joint_name in self.lower_joint_list:
                self.lower_mask[mask_start:mask_end] = 1

            if joint_name in self.hands_joint_list:
                self.hands_mask[mask_start:mask_end] = 1

        # select trainable joints
        self.smplx = smplx.create(
            self.args.deps_path+"smplx_models/", 
            model_type='smplx',
            gender='NEUTRAL_2020', 
            use_face_contour=False,
            num_betas=300,
            num_expression_coeffs=100, 
            ext='npz',
            use_pca=False,
        ).cuda().eval()
        # breakpoint()
        # if loader_type == "test":
        #     self.args.training_speakers = [2]
        
        # breakpoint()
        split_rule = pd.read_csv(self.args.data_path+"train_test_split.csv")
        self.selected_file = split_rule.loc[
            (split_rule['type'] == loader_type) 
            & (split_rule['id'].str.split("_").str[0].astype(int).isin(self.args.training_speakers))
            ]
        if self.args.additional_data and loader_type == 'train':
            split_b = split_rule.loc[
                (split_rule['type'] == 'additional') 
                & (split_rule['id'].str.split("_").str[0].astype(int).isin(self.args.training_speakers))
                ]
            #self.selected_file = split_rule.loc[
            #   (split_rule['type'] == 'additional') 
            #   & (split_rule['id'].str.split("_").str[0].astype(int).isin(self.args.training_speakers))
            #   ]
            self.selected_file = pd.concat([split_b, self.selected_file])
        if self.selected_file.empty:
            self.logger.warning(f"{loader_type} is empty for speaker {self.args.training_speakers}, use train set 0-8 instead")
            self.selected_file = split_rule.loc[
                (split_rule['type'] == 'train') 
                & (split_rule['id'].str.split("_").str[0].astype(int).isin(self.args.training_speakers))
                ]
            self.selected_file = self.selected_file.iloc[0:8]
        self.data_dir = self.args.data_path 
        # breakpoint() # check self.data_dir


        self.max_length = int(self.args.pose_length)
        self.max_audio_pre_len = math.floor(self.args.pose_length / self.args.pose_fps * self.args.audio_sr)
        if self.max_audio_pre_len > self.args.pose_length*self.args.audio_sr: 
            self.max_audio_pre_len = self.args.pose_length*self.args.audio_sr


        if self.debug: 
            self.selected_file = self.selected_file.iloc[0:10]
            self.args.cache_path = self.args.cache_path.replace("/beatx_cache/", "/beatx_debug_cache/")
            # self.args.new_cache = True

        if self.tiny:
            self.selected_file = self.selected_file.iloc[0:1]
            self.args.cache_path = self.args.cache_path.replace("/beatx_cache/", "/beatx_tiny_cache/")
            # self.args.new_cache = True
        
        # breakpoint()
        preloaded_dir = self.args.cache_path + loader_type + f"/{self.args.pose_rep}_cache" 

        # if self.args.beat_align: # TODO: Figure out what to do with thuis. 
        #     breakpoint()
        #     if not os.path.exists(self.args.data_path+f"weights/mean_vel_{self.args.pose_rep}.npy"):
        #         self.calculate_mean_velocity(self.args.data_path+f"weights/mean_vel_{self.args.pose_rep}.npy")
        #     self.avg_vel = np.load(self.args.data_path+f"weights/mean_vel_{self.args.pose_rep}.npy")
        self.names_json_path = os.path.dirname(preloaded_dir) + "/names_to_idx.json"

        self.name_to_idx = {}
        if build_cache:
            self.build_cache(preloaded_dir)
        self.lmdb_env = lmdb.open(preloaded_dir, readonly=True, lock=False)
        with self.lmdb_env.begin() as txn:
            self.n_samples = txn.stat()["entries"] 

        # self.lmdb_names = lmdb.open(preloaded_dir+"_names", readonly=True, lock=False)
        # breakpoint()
        with open(self.names_json_path, "r", encoding="utf-8") as f:
            self.name_to_idx = json.load(f)

    
    def idmapping(self, id):
        # map 1,2,3,4,5, 6,7,9,10,11,  12,13,15,16,17,  18,20,21,22,23,  24,25,27,28,30 to 0-24
        if id == 30: id = 8
        if id == 28: id = 14
        if id == 27: id = 19
        return id - 1
    
    def __len__(self):
        return self.n_samples
    

    def calculate_mean_velocity(self, save_path):
        """
        Calculate the mean velocity of the dataset and save it to the path
        """
        self.smplx = smplx.create(
            self.args.deps_path+"smplx_models/", 
            model_type='smplx',
            gender='NEUTRAL_2020', 
            use_face_contour=False,
            num_betas=300,
            num_expression_coeffs=100, 
            ext='npz',
            use_pca=False,
        ).cuda().eval()
        dir_p = self.data_dir + self.args.pose_rep + "/"
        all_list = []
        from tqdm import tqdm
        for tar in tqdm(os.listdir(dir_p)):
            if tar.endswith(".npz"):
                m_data = np.load(dir_p+tar, allow_pickle=True)
                betas, poses, trans, exps = m_data["betas"], m_data["poses"], m_data["trans"], m_data["expressions"]
                n, c = poses.shape[0], poses.shape[1]
                betas = betas.reshape(1, 300)
                betas = np.tile(betas, (n, 1))
                betas = torch.from_numpy(betas).cuda().float()
                poses = torch.from_numpy(poses.reshape(n, c)).cuda().float()
                exps = torch.from_numpy(exps.reshape(n, 100)).cuda().float()
                trans = torch.from_numpy(trans.reshape(n, 3)).cuda().float()
                max_length = 128
                s, r = n//max_length, n%max_length
                #print(n, s, r)
                all_tensor = []
                for i in range(s):
                    with torch.no_grad():
                        joints = self.smplx(
                            betas=betas[i*max_length:(i+1)*max_length], 
                            transl=trans[i*max_length:(i+1)*max_length], 
                            expression=exps[i*max_length:(i+1)*max_length], 
                            jaw_pose=poses[i*max_length:(i+1)*max_length, 66:69], 
                            global_orient=poses[i*max_length:(i+1)*max_length,:3], 
                            body_pose=poses[i*max_length:(i+1)*max_length,3:21*3+3], 
                            left_hand_pose=poses[i*max_length:(i+1)*max_length,25*3:40*3], 
                            right_hand_pose=poses[i*max_length:(i+1)*max_length,40*3:55*3], 
                            return_verts=True,
                            return_joints=True,
                            leye_pose=poses[i*max_length:(i+1)*max_length, 69:72], 
                            reye_pose=poses[i*max_length:(i+1)*max_length, 72:75],
                        )['joints'][:, :55, :].reshape(max_length, 55*3)
                    all_tensor.append(joints)
                if r != 0:
                    with torch.no_grad():
                        joints = self.smplx(
                            betas=betas[s*max_length:s*max_length+r], 
                            transl=trans[s*max_length:s*max_length+r], 
                            expression=exps[s*max_length:s*max_length+r], 
                            jaw_pose=poses[s*max_length:s*max_length+r, 66:69], 
                            global_orient=poses[s*max_length:s*max_length+r,:3], 
                            body_pose=poses[s*max_length:s*max_length+r,3:21*3+3], 
                            left_hand_pose=poses[s*max_length:s*max_length+r,25*3:40*3], 
                            right_hand_pose=poses[s*max_length:s*max_length+r,40*3:55*3], 
                            return_verts=True,
                            return_joints=True,
                            leye_pose=poses[s*max_length:s*max_length+r, 69:72], 
                            reye_pose=poses[s*max_length:s*max_length+r, 72:75],
                        )['joints'][:, :55, :].reshape(r, 55*3)
                    all_tensor.append(joints)
                joints = torch.cat(all_tensor, axis=0)
                joints = joints.permute(1, 0)
                dt = 1/30
            # first steps is forward diff (t+1 - t) / dt
                init_vel = (joints[:, 1:2] - joints[:, :1]) / dt
                # middle steps are second order (t+1 - t-1) / 2dt
                middle_vel = (joints[:, 2:] - joints[:, 0:-2]) / (2 * dt)
                # last step is backward diff (t - t-1) / dt
                final_vel = (joints[:, -1:] - joints[:, -2:-1]) / dt
                #print(joints.shape, init_vel.shape, middle_vel.shape, final_vel.shape)
                vel_seq = torch.cat([init_vel, middle_vel, final_vel], dim=1).permute(1, 0).reshape(n, 55, 3)
                #print(vel_seq.shape)
                #.permute(1, 0).reshape(n, 55, 3)
                vel_seq_np = vel_seq.cpu().numpy()
                vel_joints_np = np.linalg.norm(vel_seq_np, axis=2) # n * 55
                all_list.append(vel_joints_np)
        avg_vel = np.mean(np.concatenate(all_list, axis=0),axis=0) # 55
        np.save(save_path, avg_vel)

    def build_cache(self, preloaded_dir):
        """
        Wrapper function to build the cache
        """

        self.logger.info(f"Audio bit rate: {self.args.audio_fps}")
        self.logger.info(f"Reading data '{self.data_dir}'...")
        self.logger.info("Creating the dataset cache...")
        if self.args.new_cache:
            # os.remove(self.names_json_path)
            breakpoint()
            if os.path.exists(preloaded_dir):
                shutil.rmtree(preloaded_dir)
                # shutil.rmtree(preloaded_dir+"_names")
        # breakpoint()
        if os.path.exists(preloaded_dir) and os.path.exists(self.names_json_path):
            self.logger.info(f"Found the cache {preloaded_dir}")
        elif self.loader_type == "test":
            self.cache_generation(
                preloaded_dir, True, 
                0, 0,
                is_test=True)
        else:
            self.cache_generation(
                preloaded_dir, True, 
                self.args.clean_first_seconds, self.args.clean_final_seconds,
                is_test=False)

    def cache_generation(self, out_lmdb_dir, disable_filtering, clean_first_seconds,  clean_final_seconds, is_test=False):
        self.n_out_samples = 0
        # create db for samples
        if not os.path.exists(out_lmdb_dir): os.makedirs(out_lmdb_dir)
        # breakpoint()
        if len(self.args.training_speakers) == 1:
            dst_lmdb_env = lmdb.open(out_lmdb_dir, map_size= int(1024 ** 3 * 500))# 50G
        else:
            dst_lmdb_env = lmdb.open(out_lmdb_dir, map_size= int(1024 ** 3 * 1500))# 1000G

        # create db for names
        # if not os.path.exists(out_lmdb_dir+"_names"): os.makedirs(out_lmdb_dir+"_names")
        # dst_lmdb_names = lmdb.open(out_lmdb_dir+"_names", map_size= int(1024 ** 3 * 1))# 1G

        n_filtered_out = defaultdict(int)
    
        for index, file_name in self.selected_file.iterrows():
            f_name = file_name["id"]
            ext = ".npz" if "smplx" in self.args.pose_rep else ".bvh"
            pose_file = self.data_dir + self.args.pose_rep + "/" + f_name + ext
            pose_each_file = []
            trans_each_file = []
            shape_each_file = []
            audio_each_file = []
            facial_each_file = []
            word_each_file = []
            textsegs_each_file = []
            sem_each_file = []
            sem_score_each_file = []
            vid_each_file = []
            prominence_each_file = None
            discourse_each_file = None
            id_pose = f_name #1_wayne_0_1_1
            
            self.logger.info(colored(f"# ---- Building cache for Pose   {id_pose} ---- #", "blue"))
            if "smplx" in self.args.pose_rep:
                pose_data = np.load(pose_file, allow_pickle=True)
                assert 30%self.args.pose_fps == 0, 'pose_fps should be an aliquot part of 30'
                stride = int(30/self.args.pose_fps)
                pose_each_file = pose_data["poses"][::stride] 
                trans_each_file = pose_data["trans"][::stride]
                exps = pose_data["expressions"][::stride]
                poses = pose_each_file
                trans = trans_each_file
                shape_each_file = np.repeat(pose_data["betas"].reshape(1, 300), pose_each_file.shape[0], axis=0)
                betas = pose_data["betas"]

                # assert self.args.pose_fps == 30, "should 30"
                # breakpoint()
                # m_data = np.load(pose_file, allow_pickle=True)
                # betas, poses, trans, exps = m_data["betas"], m_data["poses"], m_data["trans"], m_data["expressions"]
                n, c = poses.shape[0], poses.shape[1]
                betas = betas.reshape(1, 300)
                betas = np.tile(betas, (n, 1))
                betas = torch.from_numpy(betas).cuda().float()
                poses = torch.from_numpy(poses.reshape(n, c)).cuda().float()
                exps = torch.from_numpy(exps.reshape(n, 100)).cuda().float()
                trans = torch.from_numpy(trans.reshape(n, 3)).cuda().float()
                max_length = 128
                s, r = n//max_length, n%max_length
                #print(n, s, r)
                all_tensor = []
                for i in range(s):
                    with torch.no_grad():
                        joints = self.smplx(
                            betas=betas[i*max_length:(i+1)*max_length], 
                            transl=trans[i*max_length:(i+1)*max_length], 
                            expression=exps[i*max_length:(i+1)*max_length], 
                            jaw_pose=poses[i*max_length:(i+1)*max_length, 66:69], 
                            global_orient=poses[i*max_length:(i+1)*max_length,:3], 
                            body_pose=poses[i*max_length:(i+1)*max_length,3:21*3+3], 
                            left_hand_pose=poses[i*max_length:(i+1)*max_length,25*3:40*3], 
                            right_hand_pose=poses[i*max_length:(i+1)*max_length,40*3:55*3], 
                            return_verts=True,
                            return_joints=True,
                            leye_pose=poses[i*max_length:(i+1)*max_length, 69:72], 
                            reye_pose=poses[i*max_length:(i+1)*max_length, 72:75],
                        )['joints'][:, (7,8,10,11), :].reshape(max_length, 4, 3).cpu()
                    all_tensor.append(joints)
                if r != 0:
                    with torch.no_grad():
                        joints = self.smplx(
                            betas=betas[s*max_length:s*max_length+r], 
                            transl=trans[s*max_length:s*max_length+r], 
                            expression=exps[s*max_length:s*max_length+r], 
                            jaw_pose=poses[s*max_length:s*max_length+r, 66:69], 
                            global_orient=poses[s*max_length:s*max_length+r,:3], 
                            body_pose=poses[s*max_length:s*max_length+r,3:21*3+3], 
                            left_hand_pose=poses[s*max_length:s*max_length+r,25*3:40*3], 
                            right_hand_pose=poses[s*max_length:s*max_length+r,40*3:55*3], 
                            return_verts=True,
                            return_joints=True,
                            leye_pose=poses[s*max_length:s*max_length+r, 69:72], 
                            reye_pose=poses[s*max_length:s*max_length+r, 72:75],
                        )['joints'][:, (7,8,10,11), :].reshape(r, 4, 3).cpu()
                    all_tensor.append(joints)
                joints = torch.cat(all_tensor, axis=0) # all, 4, 3
                # print(joints.shape)
                feetv = torch.zeros(joints.shape[1], joints.shape[0])
                joints = joints.permute(1, 0, 2)
                # print(joints.shape, feetv.shape)
                feetv[:, :-1] = (joints[:, 1:] - joints[:, :-1]).norm(dim=-1)
                #print(feetv.shape)
                contacts = (feetv < 0.01).numpy().astype(float)
                # print(contacts.shape)
                contacts = contacts.transpose(1, 0)
                # print(contacts.shape)
                pose_upper_each_file = pose_each_file * self.upper_mask
                pose_upper_each_file = pose_upper_each_file[:, self.upper_mask.astype(bool)]
                # pose_upper_each_file = np.concatenate([pose_upper_each_file, contacts], axis=1)

                pose_face_each_file = pose_each_file * self.face_mask
                pose_face_each_file = pose_face_each_file[:, self.face_mask.astype(bool)]
                # pose_face_each_file = np.concatenate([pose_face_each_file, contacts], axis=1)

                pose_lower_each_file = pose_each_file * self.lower_mask
                pose_lower_each_file = pose_lower_each_file[:, self.lower_mask.astype(bool)]
                # pose_lower_each_file = np.concatenate([pose_lower_each_file, contacts], axis=1)

                pose_hands_each_file = pose_each_file * self.hands_mask
                pose_hands_each_file = pose_hands_each_file[:, self.hands_mask.astype(bool)]
                # pose_hands_each_file = np.concatenate([pose_hands_each_file, contacts], axis=1)

                pose_each_file = np.concatenate([pose_each_file, contacts], axis=1)
                
                pose_per_bodypart = {
                    "upper": pose_upper_each_file,
                    "face": pose_face_each_file,
                    "lower": pose_lower_each_file,
                    "hands": pose_hands_each_file,
                    "all": pose_each_file,
                }
                # print(pose_each_file.shape)
                # breakpoint()
                
                if self.args.facial_rep is not None:
                    self.logger.info(f"# ---- Building cache for Facial {id_pose} and Pose {id_pose} ---- #")
                    facial_each_file = pose_data["expressions"][::stride]
                    # if self.args.facial_norm: 
                    #     facial_each_file = (facial_each_file - self.mean_facial) / self.std_facial
                    
            else:
                raise NotImplementedError
                        
            if self.args.id_rep is not None:
                int_value = self.idmapping(int(f_name.split("_")[0]))
                vid_each_file = np.repeat(np.array(int_value).reshape(1, 1), pose_each_file.shape[0], axis=0)
    
            if self.args.audio_rep is not None:
                self.logger.info(f"# ---- Building cache for Audio  {id_pose} and Pose {id_pose} ---- #")
                audio_file = pose_file.replace(self.args.pose_rep, 'wave16k').replace(ext, ".wav")
                if not os.path.exists(audio_file):
                    self.logger.warning(f"# ---- file not found for Audio  {id_pose}, skip all files with the same id ---- #")
                    self.selected_file = self.selected_file.drop(self.selected_file[self.selected_file['id'] == id_pose].index)
                    continue
                audio_each_file, sr = librosa.load(audio_file)
                audio_each_file = librosa.resample(audio_each_file, orig_sr=sr, target_sr=self.args.audio_sr)
                if self.args.audio_rep == "onset+amplitude":
                    from numpy.lib import stride_tricks
                    frame_length = 1024
                    # hop_length = 512
                    shape = (audio_each_file.shape[-1] - frame_length + 1, frame_length)
                    strides = (audio_each_file.strides[-1], audio_each_file.strides[-1])
                    rolling_view = stride_tricks.as_strided(audio_each_file, shape=shape, strides=strides)
                    amplitude_envelope = np.max(np.abs(rolling_view), axis=1)
                    # pad the last frame_length-1 samples
                    amplitude_envelope = np.pad(amplitude_envelope, (0, frame_length-1), mode='constant', constant_values=amplitude_envelope[-1])
                    audio_onset_f = librosa.onset.onset_detect(y=audio_each_file, sr=self.args.audio_sr, units='frames')
                    onset_array = np.zeros(len(audio_each_file), dtype=float)
                    onset_array[audio_onset_f] = 1.0
                    # print(amplitude_envelope.shape, audio_each_file.shape, onset_array.shape)
                    audio_each_file = np.concatenate([amplitude_envelope.reshape(-1, 1), onset_array.reshape(-1, 1)], axis=1)
                elif self.args.audio_rep == "melspec":
                    # breakpoint()
                    audio_each_file = librosa.feature.melspectrogram(y=audio_each_file, sr=self.args.audio_sr, n_mels=self.args.num_mels, hop_length=self.args.hop_length)
                    audio_each_file = audio_each_file.transpose(1, 0)
                    self.args.audio_fps = self.args.audio_sr / self.args.hop_length
                    # print(audio_each_file.shape, pose_each_file.shape)

                elif self.args.audio_rep == "wav2vec":
                    from transformers import AutoProcessor, Wav2Vec2Model

                    self.wav2vec2_processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base-960h")
                    self.wav2vec2_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
                    self.wav2vec2_model.feature_extractor._freeze_parameters()
                    self.wav2vec2_model = self.wav2vec2_model.cuda()
                    self.wav2vec2_model.eval()
                    # shifted the rest to sample function
                    
                # if self.args.audio_norm and self.args.audio_rep == "wave16k": 
                #     audio_each_file = (audio_each_file - self.mean_audio) / self.std_audio
                    
            time_offset = 0
            if self.args.word_rep is not None:
                self.logger.info(f"# ---- Building cache for Word   {id_pose} and Pose {id_pose} ---- #")
                disco_filepath = pose_file.replace(self.args.pose_rep, 'discourse_rels').replace(ext, "_whisper_relations.json")
                if not os.path.exists(disco_filepath):
                    self.logger.warning(f"# ---- file not found for Word   {id_pose}, skip all files with the same id ---- #")
                    self.selected_file = self.selected_file.drop(self.selected_file[self.selected_file['id'] == id_pose].index)
                    continue
                
                word_each_file = parse_discourse_tokens(disco_filepath)
                # breakpoint()

                with open(disco_filepath, "r", encoding="utf-8") as f:
                    discourse_each_file = json.load(f)

                textsegs_each_file = [
                    [[float(s), float(e)], t]
                    for s, e, t in zip(
                        word_each_file["start"].tolist(),
                        word_each_file["end"].tolist(),
                        word_each_file["text"].tolist(),
                    )
                ]

                # OR read textsegs from the textseg file
                # textseg_file = pose_file.replace(self.args.pose_rep, 'coref_textseg').replace(ext, ".json")
                # with open(textseg_file, "r", encoding="utf-8") as f:
                #     textsegs_each_file = json.load(f)
                # breakpoint()

                if self.args.word_rep == "bert" or self.args.word_rep == "bert_framealigned":
                    from transformers import AutoTokenizer, AutoModel
                    self.bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-cased",
                                                                        max_length=512,
                                                                        max_position_embeddings=1024,
                                                                        use_fast=True
                                                                        )
                    self.bert_model = AutoModel.from_pretrained("bert-base-cased", output_hidden_states=True)
                    self.bert_model.training = False
                    for p in self.bert_model.parameters():
                        p.requires_grad = False
                    self.bert_model = self.bert_model.cuda()
                    self.bert_model.eval()

                    # move the rest to sample function

                
                
            if self.args.emo_rep is not None:
                self.logger.info(f"# ---- Building cache for Emo    {id_pose} and Pose {id_pose} ---- #")
                rtype, start = int(id_pose.split('_')[3]), int(id_pose.split('_')[3])
                if rtype == 0 or rtype == 2 or rtype == 4 or rtype == 6:
                    if start >= 1 and start <= 64:
                        score = 0
                    elif start >= 65 and start <= 72:
                        score = 1
                    elif start >= 73 and start <= 80:
                        score = 2
                    elif start >= 81 and start <= 86:
                        score = 3
                    elif start >= 87 and start <= 94:
                        score = 4
                    elif start >= 95 and start <= 102:
                        score = 5
                    elif start >= 103 and start <= 110:
                        score = 6
                    elif start >= 111 and start <= 118:
                        score = 7
                    else: pass
                else:
                    # you may denote as unknown in the future
                    score = 0
                emo_each_file = np.repeat(np.array(score).reshape(1, 1), pose_each_file.shape[0], axis=0)    
                #print(emo_each_file)
                
            if self.args.sem_rep is not None:
                self.logger.info(f"# ---- Building cache for Sem    {id_pose} and Pose {id_pose} ---- #")
                sem_file = f"{self.data_dir}sem/{id_pose}.txt" 
                sem_all = pd.read_csv(sem_file, 
                    sep='\t', 
                    names=["name", "start_time", "end_time", "duration", "score", "keywords"])
                # we adopt motion-level semantic score here. 
                # if self.args.sem_rep == "score":
                for i in range(pose_each_file.shape[0]):
                    found_flag = False
                    for j, (start, end, score) in enumerate(zip(sem_all['start_time'],sem_all['end_time'], sem_all['score'])):
                        current_time = i/self.args.pose_fps + time_offset
                        if start<=current_time and current_time<=end: 
                            sem_score_each_file.append(score)
                            found_flag=True
                            break
                        else: continue 
                    if not found_flag: sem_score_each_file.append(0.)
                sem_score_each_file = np.array(sem_score_each_file)

                if self.args.sem_rep == "score":
                    sem_each_file = sem_score_each_file

                if self.args.sem_rep == "info":
                    # sem_each_file = []
                    length = pose_each_file.shape[0]
                    for v, (name, start, end, word) in enumerate(
                        zip(
                            sem_all["name"],
                            sem_all["start_time"],
                            sem_all["end_time"],
                            sem_all["keywords"],
                        )
                    ):
                        # breakpoint() # check current time calculation k/self.pose_fps
                        for k in range(length):
                            current_time = k / self.args.pose_fps + 0
                            if start <= current_time and current_time <= end:
                                if "beat" in name:
                                    class_name = "beat"
                                    # break
                                elif "deictic" in name:
                                    class_name = "deictic"
                                elif "iconic" in name:
                                    class_name = "iconic"
                                elif "metaphoric" in name:
                                    class_name = "metaphoric"
                                else:
                                    break
                                # breakpoint()
                                if end > (length) / self.args.pose_fps:
                                    chunk_end = length / self.args.pose_fps
                                else:
                                    chunk_end = end 

                                if start < 0:
                                    chunk_start = 0
                                else:
                                    chunk_start = start 

                                if isinstance(word, float) and math.isnan(word):
                                    word = ""

                                sem_each_file.append(
                                    {
                                        "name": class_name,
                                        "start": chunk_start,
                                        "end": chunk_end,
                                        "word": word.strip(),
                                    }
                                )
                                break
                #print(sem_each_file)

            if self.args.prom_rep is not None:
                prominence_path = f"{self.data_dir}prom/{id_pose}.prom"
                prom_data = pd.read_csv(
                    prominence_path,
                    sep="\t",
                    names=["Basename", "start", "end", "word", "prominence", "boundary"],
                )
                prom_data["word"] = prom_data["word"].fillna("")
                # breakpoint()
                prominence_each_file = prom_data
            
            filtered_result = self._sample_from_clip(
                dst_lmdb_env,
                audio_each_file, pose_per_bodypart, trans_each_file, shape_each_file, facial_each_file, word_each_file, textsegs_each_file, discourse_each_file,
                vid_each_file, emo_each_file, sem_each_file, sem_score_each_file, prominence_each_file, f_name,
                disable_filtering, clean_first_seconds, clean_final_seconds, is_test,
                ) 
            for type in filtered_result.keys():
                n_filtered_out[type] += filtered_result[type]
                                
        with dst_lmdb_env.begin() as txn:
            self.logger.info(colored(f"no. of samples: {txn.stat()['entries']}", "cyan"))
            n_total_filtered = 0
            for type, n_filtered in n_filtered_out.items():
                self.logger.info("{}: {}".format(type, n_filtered))
                n_total_filtered += n_filtered
            self.logger.info(colored("no. of excluded samples: {} ({:.1f}%)".format(
                n_total_filtered, 100 * n_total_filtered / (txn.stat()["entries"] + n_total_filtered)), "cyan"))
        dst_lmdb_env.sync()
        dst_lmdb_env.close()

        with open(self.names_json_path, "w", encoding="utf-8") as f:
            json.dump(self.name_to_idx, f)
        
        

    def _sample_from_clip(
        self, dst_lmdb_env, audio_each_file, pose_per_bodypart, trans_each_file, shape_each_file, facial_each_file, word_each_file, textsegs_each_file, discourse_each_file,
        vid_each_file, emo_each_file, sem_each_file, sem_score_each_file, prominence_each_file, f_name,
        disable_filtering, clean_first_seconds, clean_final_seconds, is_test,
        ):
        """
        for data cleaning, we ignore the data for first and final n s
        for test, we return all data 
        """
        # audio_start = int(self.alignment[0] * self.args.audio_fps)
        # pose_start = int(self.alignment[1] * self.args.pose_fps)
        #logger.info(f"before: {audio_each_file.shape} {pose_each_file.shape}")
        # audio_each_file = audio_each_file[audio_start:]
        # pose_each_file = pose_each_file[pose_start:]
        # trans_each_file = 
        #logger.info(f"after alignment: {audio_each_file.shape} {pose_each_file.shape}")
        #print(pose_each_file.shape)

        pose_each_file = pose_per_bodypart["all"]
        pose_upper_each_file = pose_per_bodypart["upper"]
        pose_face_each_file = pose_per_bodypart["face"]
        pose_lower_each_file = pose_per_bodypart["lower"]
        pose_hands_each_file = pose_per_bodypart["hands"]

        round_seconds_skeleton = pose_each_file.shape[0] // self.args.pose_fps  # assume 1500 frames / 15 fps = 100 s
        #print(round_seconds_skeleton)
        if len(audio_each_file) != 0:
            if self.args.audio_rep == "melspec":
                round_seconds_audio = int(audio_each_file.shape[0] / self.args.audio_fps)
            else:
                round_seconds_audio = audio_each_file.shape[0] // self.args.audio_sr
            if len(facial_each_file) != 0:
                round_seconds_facial = facial_each_file.shape[0] // self.args.pose_fps
                self.logger.info(f"audio: {round_seconds_audio}s, pose: {round_seconds_skeleton}s, facial: {round_seconds_facial}s")
                round_seconds_skeleton = min(round_seconds_audio, round_seconds_skeleton, round_seconds_facial)
                max_round = max(round_seconds_audio, round_seconds_skeleton, round_seconds_facial)
                if round_seconds_skeleton != max_round: 
                    self.logger.warning(f"reduce to {round_seconds_skeleton}s, ignore {max_round-round_seconds_skeleton}s")  
            else:
                self.logger.info(f"pose: {round_seconds_skeleton}s, audio: {round_seconds_audio}s")
                round_seconds_skeleton = min(round_seconds_audio, round_seconds_skeleton)
                max_round = max(round_seconds_audio, round_seconds_skeleton)
                if round_seconds_skeleton != max_round: 
                    self.logger.warning(f"reduce to {round_seconds_skeleton}s, ignore {max_round-round_seconds_skeleton}s")
        
        # breakpoint() # check the audio_fps (also for line 722) and pose_fps -> audio_fps = audio_sr
        clip_s_t, clip_e_t = clean_first_seconds, round_seconds_skeleton - clean_final_seconds # assume [10, 90]s
        clip_s_f_audio, clip_e_f_audio = self.args.audio_fps * clip_s_t, clip_e_t * self.args.audio_fps # [160,000,90*160,000]
        clip_s_f_pose, clip_e_f_pose = clip_s_t * self.args.pose_fps, clip_e_t * self.args.pose_fps # [150,90*15]
        if self.args.audio_rep == "melspec":
            # print(clip_s_f_audio, clip_e_f_audio, self.args.audio_fps)
            clip_s_f_audio = math.floor(clip_s_f_audio)
            clip_e_f_audio = math.ceil(clip_e_f_audio)
        
        
        ratio = 1.0 # multi length training is not used
        if is_test:# stride = motion length for test
            cut_length = clip_e_f_pose - clip_s_f_pose
            self.args.stride = cut_length
            self.max_length = cut_length
            # cut_length = int(self.ori_length*ratio)
            # self.args.stride = int(self.ori_length*ratio)
            # self.max_length = int(self.ori_length*ratio)
        else:
            self.args.stride = int(ratio*self.ori_stride)
            cut_length = int(self.ori_length*ratio)
            
        num_subdivision = math.floor((clip_e_f_pose - clip_s_f_pose - cut_length) / self.args.stride) + 1
        self.logger.info(f"pose from frame {clip_s_f_pose} to {clip_e_f_pose}, length {cut_length}")
        self.logger.info(f"{num_subdivision} clips is expected with stride {self.args.stride}")
        
        if len(audio_each_file) != 0:
            audio_short_length = math.floor(cut_length / self.args.pose_fps * self.args.audio_fps)
            """
            for audio sr = 16000, fps = 15, pose_length = 34, 
            audio short length = 36266.7 -> 36266
            this error is fine.
            """
            self.logger.info(f"audio from frame {clip_s_f_audio} to {clip_e_f_audio}, length {audio_short_length}")
        
        n_filtered_out = defaultdict(int)
        sample_pose_list = []
        sample_pose_upper_list = []
        sample_pose_face_list = []
        sample_pose_lower_list = []
        sample_pose_hands_list = []
        sample_audio_list = []
        sample_audenc_list = []
        sample_facial_list = []
        sample_shape_list = []
        sample_word_list = []
        sample_wordenc_list = []
        sample_textfeature_list = []
        sample_textsegs_list = []
        sample_discourse_list = []
        sample_emo_list = []
        sample_sem_list = []
        sample_semscore_list = []
        sample_vid_list = []
        sample_trans_list = []
        sample_prominence_list = []

        for i in range(num_subdivision): # cut into around 2s chip, (self npose)
            start_idx = clip_s_f_pose + i * self.args.stride
            fin_idx = start_idx + cut_length 
            sample_pose = pose_each_file[start_idx:fin_idx]
            sample_pose_upper = pose_upper_each_file[start_idx:fin_idx]
            sample_pose_face = pose_face_each_file[start_idx:fin_idx]
            sample_pose_lower = pose_lower_each_file[start_idx:fin_idx]
            sample_pose_hands = pose_hands_each_file[start_idx:fin_idx]

            sample_trans = trans_each_file[start_idx:fin_idx]
            sample_shape = shape_each_file[start_idx:fin_idx]
            # print(sample_pose.shape)
            if self.args.audio_rep is not None:
                audio_start = clip_s_f_audio + math.floor(i * self.args.stride * self.args.audio_fps / self.args.pose_fps)
                audio_end = audio_start + audio_short_length
                sample_audio = audio_each_file[audio_start:audio_end]

                if self.args.audio_rep == "wav2vec":
                    wav_inputs = self.wav2vec2_processor(
                        sample_audio, 
                        sampling_rate=self.args.audio_sr, 
                        return_tensors="pt"
                        ).to("cuda")
                    with torch.no_grad():
                        wav_outputs = self.wav2vec2_model(**wav_inputs)
                        # breakpoint() # check output shape -> frames, 768
                        sample_audenc = wav_outputs.last_hidden_state.squeeze(0).cpu().numpy()
                else:
                    sample_audenc = np.array([-1])

            else:
                sample_audio = np.array([-1])
                sample_audenc = np.array([-1])

            if self.args.word_rep is not None:
                sample_word, sample_textsegs = self.beat_extract_discourse_tokens(word_each_file, textsegs_each_file, start_idx, cut_length)
                if sample_word == '':
                    return n_filtered_out
                sample_disco = self.beat_extract_discourse_relations(discourse_each_file, start_idx, cut_length)
                # breakpoint() # check the shape of sample_word
                if "bert" in self.args.word_rep:
                    word_list = [w[1] for w in sample_textsegs] # check consistency b/w sample_word and sample_textsegs
                    sample_word_vecs, sample_textfeature = self.bert_extract_word_embeddings(sample_word)

                    if self.args.word_rep == "bert_framealigned":
                        # breakpoint() # check the shape of sample_word_vecs
                        merged_textsegs = self.merge_disco_textsegs(sample_textsegs)
                        if not len(sample_word_vecs) == len(merged_textsegs): 
                            breakpoint() # check the length of sample_word_vecs
                        if not sample_word.split() == [w[1] for w in merged_textsegs]: 
                            breakpoint() # check the consistency between sample_word and merged_textsegs
                        sample_wordenc = np.zeros((sample_pose.shape[0], sample_word_vecs[0].shape[0]))
                        for w_idx, word_vec in enumerate(sample_word_vecs):
                            start_frame = int(merged_textsegs[w_idx][0][0] * self.args.pose_fps)
                            end_frame = int(merged_textsegs[w_idx][0][1] * self.args.pose_fps)
                            sample_wordenc[start_frame:end_frame] = word_vec
                    else:
                        breakpoint() # check the shape of sample_word_vecs
                        sample_wordenc = sample_textfeature # np.stack(sample_word_vecs)

                elif self.args.word_rep == "glove_framealigned":
                    raise NotImplementedError
                else:
                    sample_wordenc = sample_word
                    sample_textfeature = sample_word      


            else:
                sample_word = np.array([-1])
                sample_disco = np.array([-1])
                sample_textsegs = np.array([-1])
                sample_wordenc = np.array([-1])
                sample_textfeature = np.array([-1])

            if self.args.prom_rep is not None:
                sample_prominence = self.beat_extract_prominence(prominence_each_file, start_idx, cut_length)
            else:
                sample_prominence = [-1]
            
            if self.args.sem_rep is not None:
                if self.args.sem_rep == "info":
                    sample_sem = []
                    utt_start_sec = start_idx / self.args.pose_fps
                    utt_end_sec = fin_idx / self.args.pose_fps
                    for sem in sem_each_file:
                        if sem["start"] >= utt_start_sec and sem["end"] <= utt_end_sec:
                            gest_type = sem["name"]
                            gest_word = sem["word"]
                            start_sec = sem["start"] - utt_start_sec
                            end_sec = sem["end"] - utt_start_sec
                            sample_sem.append(
                                {
                                    "name": gest_type,
                                    "start": start_sec,
                                    "end": end_sec,
                                    "word": gest_word,
                                }
                            )
                    
                elif self.args.sem_rep == "score":
                    sample_sem = sem_each_file[start_idx:fin_idx]
                else:
                    raise NotImplementedError
            else:
                sample_sem = np.array([-1])

            sample_facial = facial_each_file[start_idx:fin_idx] if self.args.facial_rep is not None else np.array([-1])
            # sample_word = word_each_file[start_idx:fin_idx] if self.args.word_rep is not None else np.array([-1])
            sample_emo = emo_each_file[start_idx:fin_idx] if self.args.emo_rep is not None else np.array([-1])
            # sample_sem = sem_each_file[start_idx:fin_idx] if self.args.sem_rep is not None else np.array([-1])
            sample_semscore = sem_score_each_file[start_idx:fin_idx] if self.args.sem_rep is not None else np.array([-1])
            sample_vid = vid_each_file[start_idx:fin_idx] if self.args.id_rep is not None else np.array([-1])
            
            if len(sample_pose) != 0:
                # filtering motion skeleton data # changed from MotionProcessor in EMAGE 
                sample_pose_list.append(sample_pose)
                sample_pose_upper_list.append(sample_pose_upper)
                sample_pose_face_list.append(sample_pose_face)
                sample_pose_lower_list.append(sample_pose_lower)
                sample_pose_hands_list.append(sample_pose_hands)

                sample_audio_list.append(sample_audio)
                sample_audenc_list.append(sample_audenc)
                sample_facial_list.append(sample_facial)
                sample_shape_list.append(sample_shape)
                sample_word_list.append(sample_word)
                sample_wordenc_list.append(sample_wordenc)
                sample_textfeature_list.append(sample_textfeature)
                sample_discourse_list.append(sample_disco)
                sample_textsegs_list.append(sample_textsegs)
                sample_vid_list.append(sample_vid)
                sample_emo_list.append(sample_emo)
                sample_sem_list.append(sample_sem)
                sample_semscore_list.append(sample_semscore)
                sample_trans_list.append(sample_trans)
                sample_prominence_list.append(sample_prominence)

        if len(sample_pose_list) > 0:
            with dst_lmdb_env.begin(write=True) as txn:
                for smp_idx, (pose, upper, face, lower, hands, audio, audenc, facial, shape, word, word_enc, text_f, disco, textsegs, vid, emo, sem, semscore, trans, prom) in enumerate(zip(
                    sample_pose_list,
                    sample_pose_upper_list,
                    sample_pose_face_list,
                    sample_pose_lower_list,
                    sample_pose_hands_list,
                    sample_audio_list,
                    sample_audenc_list,
                    sample_facial_list,
                    sample_shape_list,
                    sample_word_list,
                    sample_wordenc_list,
                    sample_textfeature_list,
                    sample_discourse_list,
                    sample_textsegs_list,
                    sample_vid_list,
                    sample_emo_list,
                    sample_sem_list,
                    sample_semscore_list,
                    sample_trans_list,
                    sample_prominence_list,)):
                    k = "{:005}".format(self.n_out_samples).encode("ascii")
                    # breakpoint() # check the shape of each data
                    
                    v = [pose, upper, face, lower, hands, audio, audenc, facial, shape, word, word_enc, text_f, disco, textsegs, emo, sem, semscore, vid, trans, prom, f_name + "/" + str(smp_idx)]
                    # v = [pose, upper, face, lower, hands, audio, audenc, facial, shape, word, word_enc, text_f, "disco", "textsegs", emo, "sem", vid, trans, "prom", f_name]
                    # print(v[-1])
                    # breakpoint()
                    v = pyarrow.serialize(v).to_buffer()
                    txn.put(k, v)
                    
                    # self.name_list.append(f_name + "/" + str(i))

                    self.name_to_idx[f_name + "/" + str(smp_idx)] = "{:005}".format(self.n_out_samples)
                    self.n_out_samples += 1

        return n_filtered_out

    def beat_extract_prominence(self, prom_data, frame_idx, length):
        """
        read prominence values from prominence file

        Args:
        prom_data: pandas dataframe containing prominence data
        frame_idx: start frame index
        length: length of sequence
        """
        

        start_sec = frame_idx / self.args.pose_fps
        end_sec = (frame_idx + length) / self.args.pose_fps

        window_prom_data = prom_data[
            (prom_data["start"] >= start_sec) & (prom_data["end"] <= end_sec)
        ]
        window_prom_data.loc[:, "start"] -= start_sec
        window_prom_data.loc[:, "end"] -= start_sec

        # convert to a list of tuples (word, start, end, prom)
        prominence = [
            (w, s, e, p)
            for w, s, e, p in zip(
                window_prom_data["word"],
                window_prom_data["start"],
                window_prom_data["end"],
                window_prom_data["prominence"],
            )
        ]

        return prominence

    def beat_extract_discourse_tokens(self, text_data, seg_text, frame_idx, length):
        
        start_sec = frame_idx / self.args.pose_fps
        end_sec = (frame_idx + length) / self.args.pose_fps
        text_indices = np.where(
            (text_data["start"] >= start_sec) & (text_data["end"] <= end_sec)
        )[0]
        # reduced window by one sec

        seg_text = [
            seg for seg in seg_text if seg[0][0] >= start_sec and seg[0][1] <= end_sec
        ]
        seg_text = [
            [[seg[0][0] - start_sec, seg[0][1] - start_sec], seg[1]] for seg in seg_text
        ] # [start, end], text

        # text_data = text_data["text"][text_indices]
        # breakpoint()
        merged_seg_text = self.merge_disco_textsegs(seg_text)
        text_data = [seg[1] for seg in merged_seg_text]
        # breakpoint()
        assert len(text_data) == len(merged_seg_text)

        text_data = " ".join(text_data)
        # discourse related
        # text_data = (
        #     text_data.replace(" .", ".")
        #     .replace(" ,", ",")
        #     .replace(" ?", "?")
        #     .replace(" !", "!")
        #     .replace(" :", ":")
        #     .replace(" ;", ";")
        #     .replace(" -", "-")
        #     .replace(" '", "'")
        #     .replace("  ", " ")
        #     .replace("$ ", "$")
        #     .replace(" %", "%")
        #     # .replace(" pm", "pm")
        #     # .replace(" am", "am")
        # )
        # sub the time in 3 pm to 3pm
        # text_data = re.sub(r"(\d) ([ap]m)", r"\1\2", text_data)
        # breakpoint()

        return text_data, seg_text

    def beat_extract_discourse_relations(self, disco_dict, frame_idx, length):
        # extract connectives from frame_idx to frame_idx + length

        start_sec = frame_idx / self.args.pose_fps
        end_sec = (frame_idx + length) / self.args.pose_fps

        discourse_conns = parse_discourse_relations(disco_dict, start_sec, end_sec)
        # breakpoint()

        connective_texts = []
        for conn in discourse_conns:
            if conn["start"] >= start_sec and conn["end"] <= end_sec:
                connective_texts.append(
                    (
                        conn["connective"],
                        conn["sense"],
                        conn["Arg1"]["text"],
                        conn["Arg2"]["text"],
                        conn["start"] - start_sec,
                        conn["end"] - start_sec,
                        conn["conn_start"] - start_sec,
                        conn["conn_end"] - start_sec,
                    )
                )

        # breakpoint()
        return connective_texts
    
    @staticmethod
    def merge_disco_textsegs(textsegs):
        # merge text segments with the same start and end time into one
        textsegs = copy.deepcopy(textsegs)
        merged_textsegs = []
        for i, seg in enumerate(textsegs):
            if i == 0:
                merged_textsegs.append(seg)
            else:
                start_end = seg[0]
                word = seg[1]
                if start_end == textsegs[i - 1][0]:
                    merged_textsegs[-1][1] += word
                else:
                    merged_textsegs.append(seg)
        return merged_textsegs

    def bert_extract_word_embeddings(self, sentence, layers=None):
        # Get a word vector by first tokenizing the input sentence, getting all token idxs 
        # that make up the word of interest, and then `get_hidden_states`."""
        
        # get all token idxs that belong to the word of interest
        # Push input IDs through model. Stack and sum `layers` (last four by default).
        # Select only those subword token outputs that belong to our word of interest
        # and average them.

        layers = [-4, -3, -2, -1] if layers is None else layers
        
        def get_word_vectors(encoded, sent_words, output, tokenizer):
            word_vecs = []
            token_idx = 0
            for idx, word in enumerate(sent_words):
                # breakpoint()
                # print(f"word: {word}")
                
                # print(f"token_idx: {token_idx}")

                tokens = tokenizer.tokenize(word)
                # print(f"tokens: {tokens}")
                all_word_tokenids = []
                for word_tidx, token in enumerate(tokens):
                    # print(f"token: {token}", f"word_tidx: {word_tidx}")

                    if "##" in token: # skip subword tokens (why not punctuation? because they have seperate token ids in the encoded.word_ids)
                        # print(f"sub word token: {token}")
                        # breakpoint()
                        continue
                    
                    # print(f"token_idx: {token_idx}")
                    # get the subword idxs that belong to the word of interest
                    token_ids_word = np.where(np.array(encoded.word_ids()) == token_idx)[0].tolist()
                    # if len(token_ids_word) > 1 then we have a subword token, therefore we need only increase the token_idx by 1 (skip subword tokens)
                    all_word_tokenids.extend(token_ids_word)
                    # print(f"all_word_tokenids: {all_word_tokenids}")
                    token_idx += 1 # len(token_ids_word)
                
                # select only the subword tokens that belong to our word of interest
                word_tokens_output = output[all_word_tokenids]
                word_vec =  word_tokens_output.mean(dim=0) # average subword token outputs
                word_vecs.append(word_vec.cpu().numpy())
                # token_idx += 1

            assert len(word_vecs) == len(sent_words)
            return word_vecs
        
        # breakpoint() # go through this 
        encoded = self.bert_tokenizer.encode_plus(sentence, return_tensors="pt").to("cuda")
        with torch.no_grad():
            output = self.bert_model(**encoded)
        # Get all hidden states
        states = output.hidden_states
        # Stack and sum all requested layers as reprsentation (last four by default)
        output = torch.stack([states[i] for i in layers]).sum(0).squeeze()

        sent_words = sentence.split(" ")
        # breakpoint()
        word_vecs = get_word_vectors(encoded, sent_words, output, self.bert_tokenizer)

        return word_vecs, output.cpu().numpy()


    def __getitem__(self, idx):
        
        if isinstance(idx, str):
            # print(self.name_to_idx)
            key = self.name_to_idx[idx] # here idx is the name of the sample
            key = key.encode("ascii")
        else:
            key = "{:005}".format(idx).encode("ascii")


        with self.lmdb_env.begin(write=False) as txn:
            sample = txn.get(key)
            sample = pyarrow.deserialize(sample)
            # tar_pose, in_audio, in_facial, in_shape, in_word, emo, sem, vid, trans = sample
            if len(sample) == 20:
                tar_pose, tar_upper, tar_face, tar_lower, tar_hands, in_audio, in_audenc, in_facial, in_shape, in_word, in_word_enc, in_textf, disco, textsegs, emo, sem, vid, trans, prom, f_name = sample
                semscore = np.array([-1])
            else:
                tar_pose, tar_upper, tar_face, tar_lower, tar_hands, in_audio, in_audenc, in_facial, in_shape, in_word, in_word_enc, in_textf, disco, textsegs, emo, sem, semscore, vid, trans, prom, f_name = sample

            #print(in_shape)
            #vid = torch.from_numpy(vid).int()
            emo = torch.from_numpy(emo).int()
            sem = torch.from_numpy(sem).float() if self.args.sem_rep == "score" else sem
            # in_audio = torch.from_numpy(in_audio).float() 
            in_audenc = torch.from_numpy(in_audenc).float()
            # in_word = torch.from_numpy(in_word).float() 
            in_word_enc = torch.from_numpy(in_word_enc).float() 
            in_textf = torch.from_numpy(in_textf).float()

            semscore = torch.from_numpy(semscore).float() 
            # if self.loader_type == "test":
            #     # print("test")
            #     # print(tar_pose.shape, trans.shape, in_facial.shape)
            #     tar_pose = torch.from_numpy(tar_pose).reshape((tar_pose.shape[0], -1)).float()
            #     tar_upper = torch.from_numpy(tar_upper).reshape((tar_upper.shape[0], -1)).float()
            #     tar_face = torch.from_numpy(tar_face).reshape((tar_face.shape[0], -1)).float()
            #     tar_lower = torch.from_numpy(tar_lower).reshape((tar_lower.shape[0], -1)).float()
            #     tar_hands = torch.from_numpy(tar_hands).reshape((tar_hands.shape[0], -1)).float()

            #     trans = torch.from_numpy(trans).reshape((trans.shape[0], -1)).float()
            #     in_facial = torch.from_numpy(in_facial).reshape((in_facial.shape[0], -1)).float()
            #     vid = torch.from_numpy(vid).reshape((vid.shape[0])).long()
            #     in_shape = torch.from_numpy(in_shape).float()
            # else:
            #     # print("train")
            #     # print(tar_pose.shape, trans.shape, in_facial.shape, vid.shape)
            in_shape = torch.from_numpy(in_shape).reshape((in_shape.shape[0], -1)).float()
            trans = torch.from_numpy(trans).reshape((trans.shape[0], -1)).float()
            vid = torch.from_numpy(vid).reshape((vid.shape[0])).long()
            in_facial = torch.from_numpy(in_facial).reshape((in_facial.shape[0], -1)).float()

            tar_pose = torch.from_numpy(tar_pose).reshape((tar_pose.shape[0], -1)).float()
            tar_upper = torch.from_numpy(tar_upper).reshape((tar_upper.shape[0], -1)).float()
            tar_face = torch.from_numpy(tar_face).reshape((tar_face.shape[0], -1)).float()
            tar_lower = torch.from_numpy(tar_lower).reshape((tar_lower.shape[0], -1)).float()
            tar_hands = torch.from_numpy(tar_hands).reshape((tar_hands.shape[0], -1)).float()
                
            # seperate out contacts from pose
            tar_contact = tar_pose[:, -4:]
            tar_pose = tar_pose[:, :-4]

            m_length = tar_pose.shape[0]
            motion_mask = torch.ones(m_length)

            # utf-8 encoding for text
            in_word = in_word.encode("utf-8", "ignore").decode("utf-8")

            if torch.any(torch.isnan(tar_pose)):
                raise ValueError("nan in motion")

            return {
                "motion":tar_pose, 
                "motion_upper":tar_upper,
                "motion_face":tar_face,
                "motion_lower":tar_lower,
                "motion_hands":tar_hands,

                "motion_length": m_length,
                "motion_mask": motion_mask,

                "contact":tar_contact, 
                "trans":trans,

                "facial":in_facial, 
                "beta": in_shape, 

                "raw_audio":in_audio, 
                "audio":in_audenc,

                "raw_word":in_word,
                "word":in_word_enc, # frame-aligned word embeddings
                "text_feature":in_textf, # text feature from bert
                "text_segments":textsegs, # text segments
                
                "speaker_id":vid, 
                "emo":emo, 
                "gesture_labels":sem,
                "sem_score": semscore,
                "discourse":disco,
                "prominence":prom,

                "sample_name":f_name,
                "sample_idx":idx,
                }
