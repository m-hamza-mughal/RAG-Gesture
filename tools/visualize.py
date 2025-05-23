import argparse

import os
import os.path as osp
import mmcv
import numpy as np
import torch
from mogen.models import build_architecture
from mogen.datasets import build_dataset, build_dataloader
from mogen.apis import set_random_seed
import soundfile as sf
from mmcv.runner import get_dist_info, init_dist
from mogen.utils import collect_env, str2bool
from mogen.models.utils import rotation_conversions as rc

from mmcv.runner import load_checkpoint
from mmcv.parallel import MMDataParallel


def parse_args():
    parser = argparse.ArgumentParser(description="mogen evaluation")
    parser.add_argument("config", help="test config file path")
    parser.add_argument("checkpoint", help="checkpoint file")
    parser.add_argument("--retrieval_method", help="retrieval method", default="discourse")
    parser.add_argument("--out", help="output directory", default="./results/")
    parser.add_argument("--use_retrieval", help="whether to use retrieval for testing", action="store_true")
    parser.add_argument("--outpaint", help="whether to outpaint the motion using the base model", action="store_true")
    parser.add_argument("--use_inversion", help="whether to use inversion to inject retr motion", action="store_true")
    parser.add_argument("--inversion_start_time", help="time to start the inversion (in negative vals from -1 being noise to -50 being clean sample)", type=int, default=-1)
    parser.add_argument("--visualize_inversion", help="whether to visualize the inversion results", action="store_true")
    parser.add_argument("--use_insertion_guidance", help="whether to use insertion guidance to help the inversion/generation process", action="store_true")
    parser.add_argument("--guidance_iters", help="list of number of iterations at each diffusion timestep to use the guidance", type=str, default="all_one")
    parser.add_argument("--guidance_lr", help="learning rate for the guidance", type=float, default=0.1)
    parser.add_argument("--test_batchsize", help="batch size for testing", type=int, default=32)
    # parser.add_argument('--pose_npy', help='output pose sequence file', default=None)
    parser.add_argument("--seed", type=int, default=None, help="random seed")
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="whether to set deterministic options for CUDNN backend.",
    )
    parser.add_argument(
        "--launcher",
        choices=["none", "pytorch", "slurm", "mpi"],
        default="none",
        help="job launcher",
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cuda",
        help="device used for testing",
    )
    args = parser.parse_args()
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)
    return args


def main():
    args = parse_args()

    if args.guidance_iters == "all_one":
        args.guidance_iters = [1] * 50
    elif args.guidance_iters == "all_zero":
        args.guidance_iters = [0] * 50
    elif args.guidance_iters == "all_10":
        args.guidance_iters = [10] * 50
    
    # things are reversed because the diffusion process is reversed
    elif args.guidance_iters == "decreasing": # less guidance as you go from noisy to clean
        args.guidance_iters = list(range(50))
    elif args.guidance_iters == "increasing": # more guidance as you go from noisy to clean
        args.guidance_iters = list(range(49, -1, -1))
    elif args.guidance_iters == "drop_decreasing_till_25": # as you want to keep last 25 iterations unguided and decrease the guidance from noisy to clean
        args.guidance_iters = [0]*25 + list(range(50))[25:50]
    elif args.guidance_iters == "step_increasing_from_25": # you want to keep first 10 iterations unguided and increase the guidance from noisy to clean
        args.guidance_iters = list(range(49, -1, -1))[:25] + [0]*25
    elif args.guidance_iters == "decreasing_till_25": # as you want to keep last 25 iterations unguided and decrease the guidance from noisy to clean
        args.guidance_iters = [0]*25 + list(range(25))
    elif args.guidance_iters == "increasing_from_25": # you want to keep first 10 iterations unguided and increase the guidance from noisy to clean
        args.guidance_iters = list(range(24, -1, -1)) + [0]*25
    else:
        raise ValueError("Invalid guidance_iters value")


    cfg = mmcv.Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get("cudnn_benchmark", False):
        torch.backends.cudnn.benchmark = True
    # cfg.data.test.test_mode = True

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == "none":
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = "\n".join([(f"{k}: {v}") for k, v in env_info_dict.items()])
    dash_line = "-" * 60 + "\n"
    print("Environment info:\n" + dash_line + env_info + "\n" + dash_line)
    meta["env_info"] = env_info

    # set random seeds
    if args.seed is not None:
        print(
            f"Set random seed to {args.seed}, " f"deterministic: {args.deterministic}"
        )
        set_random_seed(args.seed, deterministic=args.deterministic)
    cfg.seed = args.seed
    meta["seed"] = args.seed

    # breakpoint()
    # build the dataloader
    # cfg.data.train.training_speakers = list(range(1, 31))
    # cfg.data.train.cache_path = "/CT/GestureSynth1/work/DiscourseAwareGesture/RAGGesture_BEATX/cache/beatx_cache_backup/"
    # cfg.model.model.retrieval_cfg.lmdb_paths = cfg.model.model.retrieval_cfg.lmdb_paths.replace("_spk2", "")
    
    # breakpoint()
    train_dataset = build_dataset(cfg.data.train)

    # breakpoint()
    cfg.model.use_retrieval_for_test = args.use_retrieval

    # build the model and load checkpoint
    model = build_architecture(cfg.model, database=train_dataset)

    
    load_checkpoint(model, args.checkpoint, map_location="cpu")

    if args.device == "cpu":
        model = model.cpu()
    else:
        model = MMDataParallel(model, device_ids=[0])
    model.eval()

    test_dataset = build_dataset(cfg.data.test)

    body_upper_mask = test_dataset.upper_mask
    body_lower_mask = test_dataset.lower_mask
    body_hands_mask = test_dataset.hands_mask
    body_face_mask = test_dataset.face_mask
    
    test_dataloader = build_dataloader(
        test_dataset, samples_per_gpu=args.test_batchsize, workers_per_gpu=1, dist=False, shuffle=True
    )

    
    device = args.device
    
    model = model.to(device)
    # breakpoint()
    exp_dir = osp.join(
        args.out,
        osp.basename(osp.dirname(args.checkpoint)) + "_" + args.retrieval_method,
    )
    os.makedirs(exp_dir, exist_ok=True)
    if args.use_inversion and args.visualize_inversion:
        os.makedirs(osp.join(exp_dir, "inversion"), exist_ok=True)

    for i, data in enumerate(test_dataloader):
        
        gt_text = data["raw_word"]

        gt_audio = data["raw_audio"].numpy()

        gt_sample_name = data["sample_name"]
        
        gt_gesture_labels = data["gesture_labels"]

        if "_" in args.retrieval_method:
            data["retrieval_method"] = args.retrieval_method.split("_")[0]
        else:
            data["retrieval_method"] = args.retrieval_method

        
        with torch.no_grad():
            data["inference_kwargs"] = {
                "use_inversion": args.use_inversion,
                "outpaint": args.outpaint,
                "inversion_start_time": args.inversion_start_time,
                "visualize_inversion": args.visualize_inversion,
                "insertion_guidance": args.use_insertion_guidance,
                "guidance_iters": args.guidance_iters,
                "guidance_lr": args.guidance_lr,
            }
            # output_list = []
            output = model(**data)
            re_dict = output["retrieval_dict"]

        
        pred_upper = output["pred_upper"]
        pred_lower = output["pred_lower"]
        pred_hands = output["pred_hands"]
        pred_face = output["pred_facepose"]

        pred_motion = torch.zeros_like(output["motion"])
        pred_motion[..., body_upper_mask.astype(bool)] = pred_upper
        pred_motion[..., body_lower_mask.astype(bool)] = pred_lower
        pred_motion[..., body_hands_mask.astype(bool)] = pred_hands
        pred_motion[..., body_face_mask.astype(bool)] = pred_face
        
        pred_motion = pred_motion
        pred_facial = output["pred_exps"]
        pred_trans = output["pred_transl"]

        gt_motion = output["motion"]
        gt_facial = output["facial"]
        gt_trans = output["trans"]


        if args.use_inversion and args.visualize_inversion:
            inv_upper = output["inverted_output_upper"]
            inv_lower = output["inverted_output_lower"]
            inv_hands = output["inverted_output_hands"]
            inv_facepose = output["inverted_output_facepose"]
            inv_transl = output["inverted_output_transl"]
            inv_exps = output["inverted_output_exps"]

            # breakpoint()
            inv_bs, diff_ts, seq_len, _ = inv_upper.shape
            inv_motion = torch.zeros(inv_bs, diff_ts, seq_len, output["motion"].shape[-1]).to(inv_upper.device)
            inv_motion[..., body_upper_mask.astype(bool)] = inv_upper
            inv_motion[..., body_lower_mask.astype(bool)] = inv_lower
            inv_motion[..., body_hands_mask.astype(bool)] = inv_hands
            inv_motion[..., body_face_mask.astype(bool)] = inv_facepose

            inv_motion = inv_motion
            inv_facial = inv_exps
            inv_transl = inv_transl

            recons_upper = output["reconspair_output_upper"]
            recons_lower = output["reconspair_output_lower"]
            recons_hands = output["reconspair_output_hands"]
            recons_facepose = output["reconspair_output_facepose"]
            recons_transl = output["reconspair_output_transl"]
            recons_exps = output["reconspair_output_exps"]

            recons_bs, two, seq_len, _ = recons_upper.shape
            recons_motion = torch.zeros(recons_bs, two, seq_len, output["motion"].shape[-1]).to(recons_upper.device)
            recons_motion[..., body_upper_mask.astype(bool)] = recons_upper
            recons_motion[..., body_lower_mask.astype(bool)] = recons_lower
            recons_motion[..., body_hands_mask.astype(bool)] = recons_hands
            recons_motion[..., body_face_mask.astype(bool)] = recons_facepose

            recons_motion = recons_motion
            recons_facial = recons_exps
            recons_transl = recons_transl
            

        # breakpoint()
        # interpolate to 30fps
        if (30/cfg.motion_fps) != 1:
            assert 30%cfg.motion_fps == 0
            bs, n, dim = pred_motion.shape
            new_n = n * int(30/cfg.motion_fps)
            
            n_joints = dim // 3

            pred_motion = rc.axis_angle_to_matrix(pred_motion.reshape(bs, n, n_joints, 3))
            pred_motion = rc.matrix_to_rotation_6d(pred_motion).reshape(bs, n, n_joints*6)

            gt_motion = rc.axis_angle_to_matrix(gt_motion.reshape(bs, n, n_joints, 3))
            gt_motion = rc.matrix_to_rotation_6d(gt_motion).reshape(bs, n, n_joints*6)

            pred_motion = torch.nn.functional.interpolate(pred_motion.permute(0, 2, 1), scale_factor=30/cfg.motion_fps, mode='linear').permute(0,2,1)
            pred_facial = torch.nn.functional.interpolate(pred_facial.permute(0, 2, 1), scale_factor=30/cfg.motion_fps, mode='linear').permute(0,2,1)
            pred_trans = torch.nn.functional.interpolate(pred_trans.permute(0, 2, 1), scale_factor=30/cfg.motion_fps, mode='linear').permute(0,2,1)

            gt_motion = torch.nn.functional.interpolate(gt_motion.permute(0, 2, 1), scale_factor=30/cfg.motion_fps, mode='linear').permute(0,2,1)
            gt_facial = torch.nn.functional.interpolate(gt_facial.permute(0, 2, 1), scale_factor=30/cfg.motion_fps, mode='linear').permute(0,2,1)
            gt_trans = torch.nn.functional.interpolate(gt_trans.permute(0, 2, 1), scale_factor=30/cfg.motion_fps, mode='linear').permute(0,2,1)

            
            pred_motion = rc.rotation_6d_to_matrix(pred_motion.reshape(bs, new_n, n_joints, 6))
            pred_motion = rc.matrix_to_axis_angle(pred_motion).reshape(bs, new_n, n_joints*3)

            gt_motion = rc.rotation_6d_to_matrix(gt_motion.reshape(bs, new_n, n_joints, 6))
            gt_motion = rc.matrix_to_axis_angle(gt_motion).reshape(bs, new_n, n_joints*3)

            if args.use_inversion and args.visualize_inversion:
                # breakpoint()
                inv_bs, diff_ts, inv_seq_len, inv_dim = inv_motion.shape
                assert inv_seq_len == n and inv_dim == dim

                inv_motion = inv_motion.reshape(inv_bs * diff_ts, inv_seq_len, dim)
                inv_facial = inv_facial.reshape(inv_bs * diff_ts, inv_seq_len, 100)
                inv_transl = inv_transl.reshape(inv_bs * diff_ts, inv_seq_len, 3)

                inv_motion = rc.axis_angle_to_matrix(inv_motion.reshape(inv_bs * diff_ts, inv_seq_len, n_joints, 3))
                inv_motion = rc.matrix_to_rotation_6d(inv_motion).reshape(inv_bs * diff_ts, inv_seq_len, n_joints*6)

                inv_motion = torch.nn.functional.interpolate(inv_motion.permute(0, 2, 1), scale_factor=30/cfg.motion_fps, mode='linear').permute(0,2,1)
                inv_facial = torch.nn.functional.interpolate(inv_facial.permute(0, 2, 1), scale_factor=30/cfg.motion_fps, mode='linear').permute(0,2,1)
                inv_transl = torch.nn.functional.interpolate(inv_transl.permute(0, 2, 1), scale_factor=30/cfg.motion_fps, mode='linear').permute(0,2,1)

                inv_motion = rc.rotation_6d_to_matrix(inv_motion.reshape(inv_bs * diff_ts, new_n, n_joints, 6))
                inv_motion = rc.matrix_to_axis_angle(inv_motion).reshape(inv_bs * diff_ts, new_n, n_joints*3)

                inv_motion = inv_motion.reshape(inv_bs, diff_ts, new_n, n_joints*3)
                inv_facial = inv_facial.reshape(inv_bs, diff_ts, new_n, 100)
                inv_transl = inv_transl.reshape(inv_bs, diff_ts, new_n, 3)

                # ------------------------------

                recons_bs, two, recons_seq_len, recons_dim = recons_motion.shape
                assert recons_seq_len == n and recons_dim == dim and two == 2

                recons_motion = recons_motion.reshape(recons_bs * two, recons_seq_len, dim)
                recons_facial = recons_facial.reshape(recons_bs * two, recons_seq_len, 100)
                recons_transl = recons_transl.reshape(recons_bs * two, recons_seq_len, 3)

                recons_motion = rc.axis_angle_to_matrix(recons_motion.reshape(recons_bs * two, recons_seq_len, n_joints, 3))
                recons_motion = rc.matrix_to_rotation_6d(recons_motion).reshape(recons_bs * two, recons_seq_len, n_joints*6)

                recons_motion = torch.nn.functional.interpolate(recons_motion.permute(0, 2, 1), scale_factor=30/cfg.motion_fps, mode='linear').permute(0,2,1)
                recons_facial = torch.nn.functional.interpolate(recons_facial.permute(0, 2, 1), scale_factor=30/cfg.motion_fps, mode='linear').permute(0,2,1)
                recons_transl = torch.nn.functional.interpolate(recons_transl.permute(0, 2, 1), scale_factor=30/cfg.motion_fps, mode='linear').permute(0,2,1)

                recons_motion = rc.rotation_6d_to_matrix(recons_motion.reshape(recons_bs * two, new_n, n_joints, 6))
                recons_motion = rc.matrix_to_axis_angle(recons_motion).reshape(recons_bs * two, new_n, n_joints*3)

                recons_motion = recons_motion.reshape(recons_bs, two, new_n, n_joints*3)
                recons_facial = recons_facial.reshape(recons_bs, two, new_n, 100)
                recons_transl = recons_transl.reshape(recons_bs, two, new_n, 3)

            if re_dict is not None:
                re_motions_batch = re_dict["raw_motion"]
                re_facial_batch = re_dict["raw_facial"]
                re_trans_batch = re_dict["raw_trans"]

                
                re_motions_batch = re_motions_batch.squeeze(1)
                re_facial_batch = re_facial_batch.squeeze(1)
                re_trans_batch = re_trans_batch.squeeze(1)

                re_motions_batch = rc.axis_angle_to_matrix(re_motions_batch.reshape(bs, n, n_joints, 3))
                re_motions_batch = rc.matrix_to_rotation_6d(re_motions_batch).reshape(bs, n, n_joints*6)

                
                re_motions_batch = torch.nn.functional.interpolate(re_motions_batch.permute(0, 2, 1), scale_factor=30/cfg.motion_fps, mode='linear').permute(0, 2, 1)
                re_facial_batch = torch.nn.functional.interpolate(re_facial_batch.permute(0, 2, 1), scale_factor=30/cfg.motion_fps, mode='linear').permute(0, 2, 1)
                re_trans_batch = torch.nn.functional.interpolate(re_trans_batch.permute(0, 2, 1), scale_factor=30/cfg.motion_fps, mode='linear').permute(0, 2, 1)

                re_motions_batch = rc.rotation_6d_to_matrix(re_motions_batch.reshape(bs, new_n, n_joints, 6))
                re_motions_batch = rc.matrix_to_axis_angle(re_motions_batch).reshape(bs, new_n, n_joints*3)

                re_motions_batch = re_motions_batch.unsqueeze(1)
                re_facial_batch = re_facial_batch.unsqueeze(1)
                re_trans_batch = re_trans_batch.unsqueeze(1)



                re_dict["raw_motion"] = re_motions_batch
                re_dict["raw_facial"] = re_facial_batch
                re_dict["raw_trans"] = re_trans_batch


        
        pred_motion = pred_motion.cpu().detach().numpy()
        pred_facial = pred_facial.cpu().detach().numpy()
        pred_trans = pred_trans.cpu().detach().numpy()

        gt_motion = gt_motion.cpu().detach().numpy()
        gt_facial = gt_facial.cpu().detach().numpy()
        gt_trans = gt_trans.cpu().detach().numpy()

        if args.use_inversion and args.visualize_inversion:
            inv_motion = inv_motion.cpu().detach().numpy()
            inv_facial = inv_facial.cpu().detach().numpy()
            inv_transl = inv_transl.cpu().detach().numpy()

            recons_motion = recons_motion.cpu().detach().numpy()
            recons_facial = recons_facial.cpu().detach().numpy()
            recons_transl = recons_transl.cpu().detach().numpy()

            
            for inv_j in range(inv_motion.shape[0]):
                os.makedirs(osp.join(exp_dir, "inversion", f"sample_{inv_j}"), exist_ok=True)
                for diff_t in range(inv_motion.shape[1]):
                    np.savez(
                        osp.join(exp_dir, "inversion", f"sample_{inv_j}", f"inv_motion_{diff_t}.npz"),
                        betas=np.zeros(300,),
                        poses=inv_motion[inv_j, diff_t],
                        expressions=inv_facial[inv_j, diff_t],
                        trans=inv_transl[inv_j, diff_t],
                        model='smplx2020',
                        gender='neutral',
                        mocap_frame_rate = 30 #self.args.pose_fps ,
                    )

                # ------------------------------
                # save the recons pair

                start_img_motion = recons_motion[inv_j, 0]
                start_img_facial = recons_facial[inv_j, 0]
                start_img_transl = recons_transl[inv_j, 0]
                inv_rec_motion = recons_motion[inv_j, 1]
                inv_rec_facial = recons_facial[inv_j, 1]
                inv_rec_transl = recons_transl[inv_j, 1]

                np.savez(
                    osp.join(exp_dir, "inversion", f"sample_{inv_j}", "start_img_motion.npz"),
                    betas=np.zeros(300,),
                    poses=start_img_motion,
                    expressions=start_img_facial,
                    trans=start_img_transl,
                    model='smplx2020',
                    gender='neutral',
                    mocap_frame_rate = 30 #self.args.pose_fps ,
                )
                np.savez(
                    osp.join(exp_dir, "inversion", f"sample_{inv_j}", "inv_rec_motion.npz"),
                    betas=np.zeros(300,),
                    poses=inv_rec_motion,
                    expressions=inv_rec_facial,
                    trans=inv_rec_transl,
                    model='smplx2020',
                    gender='neutral',
                    mocap_frame_rate = 30 #self.args.pose_fps ,
                )
                np.savez(
                    osp.join(exp_dir, "inversion", f"sample_{inv_j}", "inv_rec_motion_notrans.npz"),
                    betas=np.zeros(300,),
                    poses=inv_rec_motion,
                    expressions=inv_rec_facial,
                    trans=inv_rec_transl-inv_rec_transl,
                    model='smplx2020',
                    gender='neutral',
                    mocap_frame_rate = 30 # self.args.pose_fps ,
                )
            
            print(f"Saved inversion results to {osp.join(exp_dir, 'inversion')} for batch {i}")
            breakpoint()
            


        for j, _ in enumerate(gt_sample_name):

            q_gesture_types = [lbl["name"] for lbl in gt_gesture_labels[j]]
            

            os.makedirs(osp.join(exp_dir, gt_sample_name[j]), exist_ok=True)
            print(f"Processing  {exp_dir} / {gt_sample_name[j]}")
            # breakpoint()
            np.savez(osp.join(exp_dir, gt_sample_name[j], "pred_motion.npz"),
                    betas=np.zeros(300,),
                    poses=pred_motion[j],
                    expressions=pred_facial[j],
                    trans=pred_trans[j],
                    model='smplx2020',
                    gender='neutral',
                    mocap_frame_rate = 30 #self.args.pose_fps ,
                )
            
            if args.use_inversion: # TODO: Remove this after newer model
                np.savez(osp.join(exp_dir, gt_sample_name[j], "pred_motion_notrans.npz"),
                    betas=np.zeros(300,),
                    poses=pred_motion[j],
                    expressions=pred_facial[j],
                    trans=pred_trans[j] - pred_trans[j],
                    model='smplx2020',
                    gender='neutral',
                    mocap_frame_rate = 30 #self.args.pose_fps ,
                )
            
            np.savez(osp.join(exp_dir, gt_sample_name[j], "gt_motion.npz"),
                    betas=np.zeros(300,),
                    poses=gt_motion[j],
                    expressions=gt_facial[j],
                    trans=gt_trans[j],
                    model='smplx2020',
                    gender='neutral',
                    mocap_frame_rate = 30 #self.args.pose_fps ,
                )
            with open(
                osp.join(exp_dir, gt_sample_name[j], "gt_text.txt"),
                "w",
                encoding="utf-8",
            ) as f:
                f.write(gt_text[j])
            sf.write(
                osp.join(exp_dir, gt_sample_name[j], "gt_audio.wav"), gt_audio[j], 16000
            )
            
            if re_dict is None:
                continue

            

            # breakpoint()
            re_motions = re_dict["raw_motion"][j].cpu().detach().numpy()
            re_trans = re_dict["raw_trans"][j].cpu().detach().numpy()
            re_facial = re_dict["raw_facial"][j].cpu().detach().numpy()

            re_sample_names = re_dict["raw_sample_names"][j]
            re_type2words = re_dict["raw_type2words"][j]

            q_pt_list = []
            for q_point_idx in re_type2words:
                q_word, q_type, r_word, r_type = re_type2words[q_point_idx]
                if q_word not in re_sample_names:
                    continue
                sample_name = re_sample_names[q_word]

                file_str = f"q_word:{q_word} q_type:{q_type} r_word:{r_word} r_type:{r_type} sample_name:{sample_name}"
                q_pt_list.append(file_str)

            with open(
                osp.join(exp_dir, gt_sample_name[j], "retrieval_list.txt"), "w", encoding="utf-8"
                ) as f:
                f.write("\n".join(q_pt_list))

            # for k, _ in enumerate(re_motions):
            k = 0
            if re_motions[k].sum() == 0:
                continue

            

            cropped_motion = re_motions[k] 
            cropped_facial = re_facial[k] 
            cropped_trans = re_trans[k] 

            
            
            np.savez(osp.join(exp_dir, gt_sample_name[j], f"retrieval_{k}.npz"),
                betas=np.zeros(300,),
                poses=cropped_motion,
                expressions=cropped_facial,
                trans=cropped_trans,
                model='smplx2020',
                gender='neutral',
                mocap_frame_rate = 30 #self.args.pose_fps ,
            )
            sf.write(
                osp.join(exp_dir, gt_sample_name[j], f"retrieval_{k}_audio.wav"),
                gt_audio[j],
                16000,
            )

if __name__ == "__main__":
    main()
