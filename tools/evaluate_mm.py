"""
This script assumes that the results are saved in npz format and the corresponding 
audio files are saved in wav format. Both the npz and wav files are saved in the
same directory. 
The script will load the npz files and the corresponding wav files and 
get the evaluation metrics on the saved results.
"""

import argparse

import os
import json
import os.path as osp
import mmcv
from collections import OrderedDict
import numpy as np
import torch
import soundfile as sf
from mogen.models.utils import rotation_conversions as rc
from mogen.models.utils import metric
import smplx
from mogen.datasets import build_dataset
import glob
from tqdm import tqdm



def parse_args():
    parser = argparse.ArgumentParser(description="mogen evaluation")
    parser.add_argument("npz_folder_path", help="test config file path")
    # parser.add_argument("--eval_batchsize", help="batch size for testing", type=int, default=32)

    parser.add_argument(
        "--deps_path", 
        help="path to dependencies", 
        default="/CT/GestureSynth1/work/GestureGPT/GestureRep/deps/" # TODO: change this before release
        ) 
    
    parser.add_argument("--speaker_specific", type=str, default=None, help="speaker specific eval")
    
    parser.add_argument("--eval_n", help="number of evaluation frames", type=int, default=300) 

    
    
    args = parser.parse_args()

    # args.e_path = args.dataset_path + args.e_path
    # args.avg_vel_path = args.dataset_path + args.avg_vel_path

    # args.variational = False
    # args.vae_test_len = 32
    # args.vae_test_dim = 330
    # args.vae_test_stride = 20
    # args.vae_length = 240
    # args.vae_layer = 4
    # args.vae_grow = [1,1,2,1]

    # args.audio_sr = 16000
    args.pose_fps = 30
    
    args.device = torch.device("cuda" if torch.cuda.is_available else "cpu")

    return args

# HAND_JOINTS = list(range(25, 55))
# UPPER_BODY_JOINTS = [3,6,9,12,13,14,15,16,17,18,19,20,21]
# NOT_UPPERHAND_JOINTS = [i for i in range(55) if i not in UPPER_BODY_JOINTS and i not in HAND_JOINTS]

class MMEvaluator:
    def __init__(self, args):
        self.args = args
        # self.alignmenter
        self.smplx_model = smplx.create(
            self.args.deps_path+"smplx_models/", 
            model_type='smplx',
            gender='NEUTRAL_2020', 
            use_face_contour=False,
            num_betas=300,
            num_expression_coeffs=100, 
            ext='npz',
            use_pca=False,
        ).to(self.args.device).eval()
        
            
            

    def evaluate(self):
        """
        Central function to evaluate the stored results.
        """
        mm_all = 0
        
        exp_dir = self.args.npz_folder_path
        # pred_files = glob.glob(osp.join(exp_dir, "*/*/pred_motion.npz"))
        pred_files = glob.glob(osp.join(exp_dir, "*/*_rep0"))
        
        eval_n = self.args.eval_n
        
        list_files = list(range(0, 5))
        # list_files.remove(4)

        if self.args.speaker_specific is not None:
            print(f"Speaker specific: {self.args.speaker_specific}")

        for file in tqdm(pred_files):

            if self.args.speaker_specific is not None:
                if '_' + self.args.speaker_specific + '_' not in file:
                    continue

            pred_all = []
            

            for lf in list_files:
                # breakpoint()
                # sample_file = file.replace("mm1", f"mm{lf}")
                sample_file = file.replace("rep0", f"rep{lf}")
                sample_file = sample_file + "/pred_motion.npz"

                # if lf == 4:
                #     continue

                if not osp.exists(sample_file):
                    print(f"File not found: {sample_file}")
                    continue

                pred = np.load(sample_file)

                rec_pose = torch.from_numpy(pred["poses"]).float()
                rec_pose = rec_pose.unsqueeze(0).to(self.args.device)

                rec_trans = torch.from_numpy(pred["trans"]).float()
                rec_trans = rec_trans.unsqueeze(0).to(self.args.device)

                rec_exps = torch.from_numpy(pred["expressions"]).float()
                rec_exps = rec_exps.unsqueeze(0).to(self.args.device)

                tar_beta_np = np.zeros_like(pred["betas"])
                tar_beta = torch.from_numpy(tar_beta_np).float()\
                    .unsqueeze(0).to(self.args.device)
                
                # breakpoint()

                rec_pose = rec_pose[:, :eval_n]

                rec_trans = rec_trans[:, :eval_n]
                
                # tar_exps = tar_exps[:, :eval_n]
                
                tar_beta = tar_beta.repeat(1, eval_n, 1)
                
                bs, n, nj = rec_pose.shape
                nj = nj//3

                rec_pose = rec_pose.reshape(bs*n, nj*3)
                
                
                vertices_rec = self.smplx_model(
                        betas=tar_beta.reshape(bs*n, 300),
                        transl=rec_trans.reshape(bs*n, 3)-rec_trans.reshape(bs*n, 3), 
                        expression=rec_exps.reshape(bs*n, 100)-rec_exps.reshape(bs*n, 100),
                        jaw_pose=rec_pose[:, 66:69], 
                        global_orient=rec_pose[:,:3], 
                        body_pose=rec_pose[:,3:21*3+3], 
                        left_hand_pose=rec_pose[:,25*3:40*3], 
                        right_hand_pose=rec_pose[:,40*3:55*3], 
                        return_joints=True, 
                        leye_pose=rec_pose[:, 69:72], 
                        reye_pose=rec_pose[:, 72:75],
                    )
                
                joints_rec = vertices_rec["joints"].detach().cpu().numpy().reshape(1, n, 127*3)[0, :n, :55*3]

                joints_rec_rootnorm = joints_rec.reshape(n, 55, 3).copy()
                joints_rec_rootnorm = joints_rec_rootnorm - joints_rec_rootnorm[:1, :1]
                
                pred_all.append(joints_rec_rootnorm[np.newaxis, :])
                

            pred_all = np.concatenate(pred_all, axis=0)
            preddiv = metric.calculate_avg_distance(pred_all)
            # print(f"mm: {preddiv}")

            mm_all += preddiv

        mm_all /= len(pred_files)
        print(f"mm_all: {mm_all}")

        return mm_all
if __name__ == "__main__":
    arguments = parse_args()
    
    evaluator = MMEvaluator(arguments)
    evaluator.evaluate()
