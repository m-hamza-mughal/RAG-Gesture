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
from mogen.models.eval_models.model import VAESKConv
import librosa
import smplx
from mogen.datasets import build_dataset
import glob
from tqdm import tqdm


def load_checkpoints(model, save_path, load_name='model'):
    states = torch.load(save_path)
    new_weights = OrderedDict()
    flag=False
    for k, v in states['model_state'].items():
        #print(k)
        if "module" not in k:
            break
        else:
            new_weights[k[7:]]=v
            flag=True
    if flag: 
        try:
            model.load_state_dict(new_weights)
        except:
            #print(states['model_state'])
            model.load_state_dict(states['model_state'])
    else:
        model.load_state_dict(states['model_state'])
    print(f"load self-pretrained checkpoints for {load_name}")

def parse_args():
    parser = argparse.ArgumentParser(description="mogen evaluation")
    parser.add_argument("npz_folder_path", help="test config file path")
    # parser.add_argument("--eval_batchsize", help="batch size for testing", type=int, default=32)

    parser.add_argument(
        "--deps_path", 
        help="path to dependencies", 
        default="/CT/GestureSynth1/work/GestureGPT/GestureRep/deps/" # TODO: change this before release
        ) 
    parser.add_argument(
        "--dataset_path",
        help="path to the dataset which contains eval_model and vel_path",
        default="/CT/GestureSynth1/work/GestureGPT/PantoMatrix/BEAT2/beat_english_v2.0.0/" #TODO: change this before release
    )
    parser.add_argument(
        "--e_path", 
        help="relative path to the model weights", 
        default="weights/AESKConv_240_100.bin" # TODO: change this before release
    )
    parser.add_argument(
        "--avg_vel_path", 
        help="relative path to average velocity file", 
        default="weights/mean_vel_smplxflame_30.npy"
    )
    parser.add_argument(
        "--test_cfg",
        help="path to the test config file",
        default="/CT/GestureSynth1/work/GestureGPT/PantoMatrix/BEAT2/beat_english_v2.0.0/beat_test_cfg.py" # TODO: change this before release
    )
    parser.add_argument("--speaker_specific", type=str, default=None, help="speaker specific eval")
    
    parser.add_argument("--eval_n", help="number of evaluation frames", type=int, default=300) 

    parser.add_argument("--calculate_srgr", help="calculate srgr", action="store_true")
    
    args = parser.parse_args()

    args.e_path = args.dataset_path + args.e_path
    args.avg_vel_path = args.dataset_path + args.avg_vel_path

    args.variational = False
    args.vae_test_len = 32
    args.vae_test_dim = 330
    args.vae_test_stride = 20
    args.vae_length = 240
    args.vae_layer = 4
    args.vae_grow = [1,1,2,1]

    args.audio_sr = 16000
    args.pose_fps = 30
    
    args.device = torch.device("cuda" if torch.cuda.is_available else "cpu")

    return args

HAND_JOINTS = list(range(25, 55))
UPPER_BODY_JOINTS = [3,6,9,12,13,14,15,16,17,18,19,20,21]
NOT_UPPERHAND_JOINTS = [i for i in range(55) if i not in UPPER_BODY_JOINTS and i not in HAND_JOINTS]

class Evaluator:
    def __init__(self, args):
        self.args = args
        self.reclatent_loss = torch.nn.MSELoss().to(self.args.device)
        self.vel_loss = torch.nn.L1Loss(reduction='mean').to(self.args.device)
        self.align_mask = 0
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
        self.avg_vel = np.load(args.avg_vel_path)
        self.alignmenter = metric.alignment(
            0.3, 
            7, 
            self.avg_vel, 
            upper_body=[3,6,9,12,13,14,15,16,17,18,19,20,21] # specific to SMPLX
        ) 
        self.align_mask = 10 # 1/3 sec changed from 2 sec (60 frames)
        self.l1_calculator = metric.L1div()
        self.gt_l1_calculator = metric.L1div()

        self.mpjpe_calculator = metric.MPJPE()

        if args.calculate_srgr:
            self.test_cfg = mmcv.Config.fromfile(args.test_cfg)
            self.test_dataset = build_dataset(self.test_cfg.data.test)
            self.srgr_calculator = metric.SRGR(threshold=0.3, joints=55) 
            
            

    def evaluate(self):
        """
        Central function to evaluate the stored results.
        """
        align = 0 
        gt_align = 0
        latent_out = []
        latent_ori = []
        l2_all = 0 
        lvel = 0
        total_length = 0
        
        # eval_model = VAESKConv(self.args)
        # eval_copy = VAESKConv(self.args).to(self.args.device)
        
        # # breakpoint()
        # load_checkpoints(eval_copy, self.args.e_path, "VAESKConv")
        # load_checkpoints(eval_model, self.args.e_path, "VAESKConv")
        # eval_model.eval()
        # eval_copy.eval()

        exp_dir = self.args.npz_folder_path
        pred_files = glob.glob(osp.join(exp_dir, "*/*/pred_motion.npz"))
        
        eval_n = self.args.eval_n

        done_count = 0

        pred_all = []
        gt_all = []

        for file in tqdm(pred_files):

            data_idx = "/".join(os.path.dirname(file).split("/")[-2:])
            # print(f"Processing {data_idx}")

            # breakpoint()
            # if "carla" in file:
            #     continue
            if self.args.speaker_specific is not None:
                if self.args.speaker_specific not in file:
                    continue


            pred = np.load(file)
            gt = np.load(file.replace("pred", "gt"))

            # breakpoint()

            if os.path.exists(os.path.join(os.path.dirname(file), "retrieval_0.npz")):
                retrieval = np.load(os.path.join(os.path.dirname(file), "retrieval_0.npz"))
                retrieval = retrieval["poses"]
                retrieval = torch.from_numpy(retrieval).float().to(self.args.device)
                retrieval = retrieval.unsqueeze(0)
            else:
                retrieval = None

            
            audio, sr = librosa.load(file.replace("pred_motion", "gt_audio").replace("npz", "wav"))
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

            rec_pose = torch.from_numpy(pred["poses"]).float()
            rec_pose = rec_pose.unsqueeze(0).to(self.args.device)
            tar_pose = torch.from_numpy(gt["poses"]).float()
            tar_pose = tar_pose.unsqueeze(0).to(self.args.device)

            rec_trans = torch.from_numpy(pred["trans"]).float()
            rec_trans = rec_trans.unsqueeze(0).to(self.args.device)
            tar_trans = torch.from_numpy(gt["trans"]).float()
            tar_trans = tar_trans.unsqueeze(0).to(self.args.device)

            tar_exps = torch.from_numpy(gt["expressions"]).float()
            tar_exps = tar_exps.unsqueeze(0).to(self.args.device)
            rec_exps = torch.from_numpy(pred["expressions"]).float()
            rec_exps = rec_exps.unsqueeze(0).to(self.args.device)

            tar_beta = torch.from_numpy(gt["betas"]).float()
            tar_beta = tar_beta.unsqueeze(0).to(self.args.device)
            
            # breakpoint()

            rec_pose = rec_pose[:, :eval_n]
            tar_pose = tar_pose[:, :eval_n]

            rec_trans = rec_trans[:, :eval_n]
            tar_trans = tar_trans[:, :eval_n]

            tar_exps = tar_exps[:, :eval_n]
            rec_exps = rec_exps[:, :eval_n]

            tar_beta = tar_beta.repeat(1, eval_n, 1)

            if retrieval is not None:
                retrieval = retrieval[:, :eval_n]
                # make a mask with 0 where the retrieval is 0 and 1 where it is not
                retrieval_reshaped = retrieval.squeeze(0).reshape(eval_n, 55, 3)

                mpjpe_mask = torch.ones_like(retrieval_reshaped)

                mpjpe_mask[retrieval_reshaped == 0] = 0
                # mask out joints that are not upper body and hand joints
                mpjpe_mask[:, NOT_UPPERHAND_JOINTS] = 0

                mpjpe_mask = mpjpe_mask.sum(dim=-1)
                mpjpe_mask = (mpjpe_mask > 0).float() # eval_n, 55
                


            bs, n, nj = rec_pose.shape
            nj = nj//3

            # rec_pose = rc.axis_angle_to_matrix(rec_pose.reshape(bs*n, nj, 3))
            # rec_pose = rc.matrix_to_rotation_6d(rec_pose).reshape(bs, n, nj*6)
            # tar_pose = rc.axis_angle_to_matrix(tar_pose.reshape(bs*n, nj, 3))
            # tar_pose = rc.matrix_to_rotation_6d(tar_pose).reshape(bs, n, nj*6)

            # # # breakpoint()
            # # remain = n%self.args.vae_test_len
            # # latent_out.append(
            # #     eval_copy.map2latent(rec_pose[:, :n-remain])
            # #     .reshape(-1, self.args.vae_length).detach().cpu().numpy()
            # # ) # bs * n/something * 240
            # # latent_ori.append(
            # #     eval_copy.map2latent(tar_pose[:, :n-remain])
            # #     .reshape(-1, self.args.vae_length).detach().cpu().numpy()
            # # )

            # rec_pose = rc.rotation_6d_to_matrix(rec_pose.reshape(bs*n, nj, 6))
            # rec_pose = rc.matrix_to_axis_angle(rec_pose).reshape(bs*n, nj*3)
            # tar_pose = rc.rotation_6d_to_matrix(tar_pose.reshape(bs*n, nj, 6))
            # tar_pose = rc.matrix_to_axis_angle(tar_pose).reshape(bs*n, nj*3)

            rec_pose = rec_pose.reshape(bs*n, nj*3)
            tar_pose = tar_pose.reshape(bs*n, nj*3)

            if retrieval is not None:
                # breakpoint()
                retrieval = retrieval.reshape(bs*n, -1)
            
            vertices_rec = self.smplx_model(
                    betas=tar_beta.reshape(bs*n, 300),
                    transl=rec_trans.reshape(bs*n, 3)-rec_trans.reshape(bs*n, 3), 
                    expression=tar_exps.reshape(bs*n, 100)-tar_exps.reshape(bs*n, 100),
                    jaw_pose=rec_pose[:, 66:69], 
                    global_orient=rec_pose[:,:3], 
                    body_pose=rec_pose[:,3:21*3+3], 
                    left_hand_pose=rec_pose[:,25*3:40*3], 
                    right_hand_pose=rec_pose[:,40*3:55*3], 
                    return_joints=True, 
                    leye_pose=rec_pose[:, 69:72], 
                    reye_pose=rec_pose[:, 72:75],
                )
            vertices_tar = self.smplx_model(
                    betas=tar_beta.reshape(bs*n, 300), 
                    transl=rec_trans.reshape(bs*n, 3)-rec_trans.reshape(bs*n, 3), 
                    expression=tar_exps.reshape(bs*n, 100)-tar_exps.reshape(bs*n, 100),
                    jaw_pose=tar_pose[:, 66:69], 
                    global_orient=tar_pose[:,:3], 
                    body_pose=tar_pose[:,3:21*3+3], 
                    left_hand_pose=tar_pose[:,25*3:40*3], 
                    right_hand_pose=tar_pose[:,40*3:55*3], 
                    return_joints=True, 
                    leye_pose=tar_pose[:, 69:72], 
                    reye_pose=tar_pose[:, 72:75],
                )
            
            if retrieval is not None:
                vertices_retrieval = self.smplx_model(
                    betas=tar_beta.reshape(bs*n, 300),
                    transl=torch.zeros_like(rec_trans.reshape(bs*n, 3)),
                    expression=torch.zeros_like(tar_exps.reshape(bs*n, 100)),
                    jaw_pose=retrieval[:, 66:69],
                    global_orient=retrieval[:,:3],
                    body_pose=retrieval[:,3:21*3+3],
                    left_hand_pose=retrieval[:,25*3:40*3],
                    right_hand_pose=retrieval[:,40*3:55*3],
                    return_joints=True,
                    leye_pose=retrieval[:, 69:72],
                    reye_pose=retrieval[:, 72:75],
                )
            
            vertices_rec_face = self.smplx_model(
                    betas=tar_beta.reshape(bs*n, 300), 
                    transl=rec_trans.reshape(bs*n, 3)-rec_trans.reshape(bs*n, 3), 
                    expression=rec_exps.reshape(bs*n, 100), 
                    jaw_pose=rec_pose[:, 66:69], 
                    global_orient=rec_pose[:,:3]-rec_pose[:,:3], 
                    body_pose=rec_pose[:,3:21*3+3]-rec_pose[:,3:21*3+3],
                    left_hand_pose=rec_pose[:,25*3:40*3]-rec_pose[:,25*3:40*3],
                    right_hand_pose=rec_pose[:,40*3:55*3]-rec_pose[:,40*3:55*3],
                    return_verts=True, 
                    return_joints=True,
                    leye_pose=rec_pose[:, 69:72]-rec_pose[:, 69:72],
                    reye_pose=rec_pose[:, 72:75]-rec_pose[:, 72:75],
                )
            vertices_tar_face = self.smplx_model(
                betas=tar_beta.reshape(bs*n, 300), 
                transl=tar_trans.reshape(bs*n, 3)-tar_trans.reshape(bs*n, 3), 
                expression=tar_exps.reshape(bs*n, 100), 
                jaw_pose=tar_pose[:, 66:69], 
                global_orient=tar_pose[:,:3]-tar_pose[:,:3],
                body_pose=tar_pose[:,3:21*3+3]-tar_pose[:,3:21*3+3], 
                left_hand_pose=tar_pose[:,25*3:40*3]-tar_pose[:,25*3:40*3],
                right_hand_pose=tar_pose[:,40*3:55*3]-tar_pose[:,40*3:55*3],
                return_verts=True, 
                return_joints=True,
                leye_pose=tar_pose[:, 69:72]-tar_pose[:, 69:72],
                reye_pose=tar_pose[:, 72:75]-tar_pose[:, 72:75],
            )  
            joints_rec = vertices_rec["joints"].detach().cpu().numpy().reshape(1, n, 127*3)[0, :n, :55*3]
            joints_tar = vertices_tar["joints"].detach().cpu().numpy().reshape(1, n, 127*3)[0, :n, :55*3]

            if retrieval is not None:
                joints_retrieval = vertices_retrieval["joints"].detach().cpu().numpy().reshape(1, n, 127*3)[0, :n, :55*3]
            
            # facial_rec = vertices_rec_face['vertices'].reshape(1, n, -1)[0, :n]
            # facial_tar = vertices_tar_face['vertices'].reshape(1, n, -1)[0, :n]
            # face_vel_loss = self.vel_loss(facial_rec[1:, :] - facial_tar[:-1, :], facial_tar[1:, :] - facial_tar[:-1, :])
            # l2 = self.reclatent_loss(facial_rec, facial_tar)
            # l2_all += l2.item() * n
            # lvel += face_vel_loss.item() * n

            # breakpoint()
            # self.l1_calculator.run(joints_rec.copy())
            # self.gt_l1_calculator.run(joints_tar.copy())

            joints_rec_rootnorm = joints_rec.reshape(n, 55, 3).copy()
            joints_rec_rootnorm = joints_rec_rootnorm - joints_rec_rootnorm[:1, :1]

            joints_tar_rootnorm = joints_tar.reshape(n, 55, 3).copy()
            joints_tar_rootnorm = joints_tar_rootnorm - joints_tar_rootnorm[:1, :1]
            # if retrieval is not None:
            #     # breakpoint()
                
            #     joints_retrieval = joints_retrieval.reshape(n, 55, 3).copy()

            #     # start from the origin (relative to the first frame)
                
            #     joints_retrieval = joints_retrieval - joints_retrieval[:1, :1]

            #     mpjpe_mask = mpjpe_mask.cpu().numpy()
            #     mpjpe_for_pred = self.mpjpe_calculator.compute_error(joints_rec_rootnorm, joints_retrieval, mpjpe_mask)


            
            pred_all.append(joints_rec_rootnorm[np.newaxis, :])
            gt_all.append(joints_tar_rootnorm[np.newaxis, :])
            # pred_all.append(joints_rec[np.newaxis, :])
            # gt_all.append(joints_tar[np.newaxis, :])
            
            # if self.alignmenter is not None:
            #     in_audio_eval = audio
            #     # breakpoint() # should be 130133/16000 = 8.13 seconds
            #     in_audio_eval = in_audio_eval[:int(self.args.audio_sr / self.args.pose_fps*n)]
            #     a_offset = int(self.align_mask * (self.args.audio_sr / self.args.pose_fps))
            #     onset_bt = self.alignmenter.load_audio(
            #         in_audio_eval, 
            #         a_offset, 
            #         len(in_audio_eval)-a_offset, 
            #         True)
            #     beat_vel = self.alignmenter.load_pose(joints_rec, self.align_mask, n-self.align_mask, 30, True)
            #     gt_beat_vel = self.alignmenter.load_pose(joints_tar, self.align_mask, n-self.align_mask, 30, True)
            #     # print(beat_vel)
            #     align += (self.alignmenter.calculate_align(onset_bt, beat_vel, 30) * (n-2*self.align_mask))
            #     gt_align += (self.alignmenter.calculate_align(onset_bt, gt_beat_vel, 30) * (n-2*self.align_mask))


            # if self.args.calculate_srgr:
            #     # breakpoint()
            #     data_sample = self.test_dataset[data_idx]
            #     tar_sem = data_sample["sem_score"].unsqueeze(0).unsqueeze(0) #.to(self.args.device)
            #     if self.test_cfg.motion_fps != 30:
            #         assert 30 % self.test_cfg.motion_fps == 0
            #         # tar_sem = tar_sem.repeat(1, 1, 30//self.test_cfg.motion_fps, 1)
            #         tar_sem = torch.nn.functional.interpolate(
            #             tar_sem, scale_factor=30/self.test_cfg.motion_fps, mode='linear')
                    
            #     tar_sem = tar_sem.squeeze(0).squeeze(0)
                

            #     _ = self.srgr_calculator.run(joints_rec, joints_tar, tar_sem.cpu().numpy())
            
            # total_length += n
            # done_count += 1

        # print(f"l2 loss: {l2_all/total_length}")
        # print(f"lvel loss: {lvel/total_length}")

        # latent_out_all = np.concatenate(latent_out, axis=0)
        # latent_ori_all = np.concatenate(latent_ori, axis=0)
        # fid = metric.FIDCalculator.frechet_distance(latent_out_all, latent_ori_all)
        # print(f"fid score: {fid}")
        
        # align_avg = align/(total_length-2*done_count*self.align_mask)
        # gt_align_avg = gt_align/(total_length-2*done_count*self.align_mask)
        # print(f"align score: {align_avg}")
        # print(f"gt align score: {gt_align_avg}")

        # l1div = self.l1_calculator.avg()
        # print(f"l1div score: {l1div}")

        # if self.args.calculate_srgr:
        #     srgr = self.srgr_calculator.avg()
        #     print(f"srgr score: {srgr}")

        # gt_l1div = self.gt_l1_calculator.avg()
        # print(f"gt l1div score: {gt_l1div}")

        # # if retrieval is not None:
        # mpjpe = self.mpjpe_calculator.get_average_error()
        # print(f"mpjpe score: {mpjpe}")

        pred_all = np.concatenate(pred_all, axis=0)
        gt_all = np.concatenate(gt_all, axis=0)
        preddiv = metric.calculate_avg_distance(pred_all)
        gt_div = metric.calculate_avg_distance(gt_all)

        print(f"pred div: {preddiv}")
        print(f"gt div: {gt_div}")



if __name__ == "__main__":
    arguments = parse_args()
    
    evaluator = Evaluator(arguments)
    evaluator.evaluate()
