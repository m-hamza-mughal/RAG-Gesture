import argparse

import os
import os.path as osp
import mmcv
import copy
import numpy as np
import torch
from mogen.models import build_architecture
from mogen.datasets import build_dataset, build_dataloader
from mogen.apis import set_random_seed
import soundfile as sf
from mmcv.runner import get_dist_info, init_dist
from mogen.utils import collect_env, str2bool
from transformers import AutoTokenizer, AutoModel
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
    parser.add_argument("--test_batchsize", help="batch size for testing", type=int, default=1)
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

def get_text_feature(bert_model, tokenizer, datasetobj, textsegs):
    # assert len(textsegs) == 1, "Only one text segment is supported for now"
    merged_seg_text = datasetobj.merge_disco_textsegs(textsegs)
    text_data = [seg[1] for seg in merged_seg_text]
    # breakpoint()
    assert len(text_data) == len(merged_seg_text)
    sentence = " ".join(text_data)

    layers = [-4, -3, -2, -1]
    encoded = tokenizer.encode_plus(sentence, return_tensors="pt").to("cuda")
    with torch.no_grad():
        output = bert_model(**encoded)
    # Get all hidden states
    states = output.hidden_states
    # Stack and sum all requested layers as reprsentation (last four by default)
    output = torch.stack([states[i] for i in layers]).sum(0).squeeze()

    return [sentence], [output]

def get_wav2vec2_feature(wav2vec_model, wav2vec2_processor, audio):
    wav_inputs = wav2vec2_processor(
        audio, 
        sampling_rate=16000, 
        return_tensors="pt"
        ).to("cuda")
    with torch.no_grad():
        wav_outputs = wav2vec_model(**wav_inputs)
        # breakpoint() # check output shape -> frames, 768
        sample_audenc = wav_outputs.last_hidden_state.squeeze(0)

    return sample_audenc

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
    
    
    data_bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-cased",
                                                        max_length=512,
                                                        max_position_embeddings=1024,
                                                        use_fast=True
                                                        )
    data_bert_model = AutoModel.from_pretrained("bert-base-cased", output_hidden_states=True)
    data_bert_model.training = False
    for p in data_bert_model.parameters():
        p.requires_grad = False
    data_bert_model = data_bert_model.cuda()
    data_bert_model.eval()

    from transformers import AutoProcessor, Wav2Vec2Model

    data_wav2vec2_processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base-960h")
    data_wav2vec2_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
    data_wav2vec2_model.feature_extractor._freeze_parameters()
    data_wav2vec2_model = data_wav2vec2_model.cuda()
    data_wav2vec2_model.eval()

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

    # dataset_name = cfg.data.test.dataset_name
    # assert dataset_name == ""
    #
    # breakpoint()
    # cfg.data.test.training_speakers = [2]   
    breakpoint()
    test_dataset = build_dataset(cfg.data.test)

    body_upper_mask = test_dataset.upper_mask
    body_lower_mask = test_dataset.lower_mask
    body_hands_mask = test_dataset.hands_mask
    body_face_mask = test_dataset.face_mask

    assert args.test_batchsize == 1, "Batch size should be 1 for long form inference"
    test_dataloader = build_dataloader(
        test_dataset, samples_per_gpu=args.test_batchsize, workers_per_gpu=1, dist=False, shuffle=False
    )

    # train_dataloader = build_dataloader(
    #     train_dataset, samples_per_gpu=args.test_batchsize, workers_per_gpu=1, dist=False, shuffle=True
    # )

    # torch.manual_seed(2809) # for reproducibility of the test set

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

    overlap_framelen = model.module.model.frame_chunk_size
    motion_seqlen = cfg.max_seq_len
    

    for i, data in enumerate(test_dataloader):
        

        # breakpoint()
        gt_text = data["raw_word"]

        gt_audio = data["raw_audio"].numpy()

        gt_sample_name = data["sample_name"]
        
        gt_gesture_labels = data["gesture_labels"]
        # breakpoint()

        if "_" in args.retrieval_method:
            data["retrieval_method"] = args.retrieval_method.split("_")[0]
        else:
            data["retrieval_method"] = args.retrieval_method

        if data["sample_name"][0].replace("/0", "") not in ["17_itoi_0_1_1",  "17_itoi_0_2_2",  "2_scott_0_65_65",  "5_stewart_0_1_1", "15_carlos_0_111_111", "17_itoi_0_73_73", "25_goto_0_87_87", "6_carla_0_73_73", "17_itoi_0_2_2", "27_yingqing_0_65_65"]:
            continue
        

        # make subsets of the data
        
        sample_motion_len = data["motion"].shape[1]
        chunk_starts = [0] + list(range(motion_seqlen - overlap_framelen, sample_motion_len, motion_seqlen - overlap_framelen))
        # chunk_starts = [i for i in chunk_starts if i + motion_seqlen <= sample_motion_len]
        chunk_ends = [i + motion_seqlen for i in chunk_starts]
        ori_audio = copy.deepcopy(data["raw_audio"])
        if chunk_ends[-1] > sample_motion_len:
            # breakpoint()
            remainder = chunk_ends[-1] - sample_motion_len
            data["motion"] = torch.cat([data["motion"], torch.zeros((1, remainder, data["motion"].shape[2])).to(data["motion"].device)], dim=1)
            data["motion_upper"] = torch.cat([data["motion_upper"], torch.zeros((1, remainder, data["motion_upper"].shape[2])).to(data["motion_upper"].device)], dim=1)
            data["motion_lower"] = torch.cat([data["motion_lower"], torch.zeros((1, remainder, data["motion_lower"].shape[2])).to(data["motion_lower"].device)], dim=1)
            data["motion_face"] = torch.cat([data["motion_face"], torch.zeros((1, remainder, data["motion_face"].shape[2])).to(data["motion_face"].device)], dim=1)
            data["motion_hands"] = torch.cat([data["motion_hands"], torch.zeros((1, remainder, data["motion_hands"].shape[2])).to(data["motion_hands"].device)], dim=1)
            data["motion_mask"] = torch.cat([data["motion_mask"], data["motion_mask"][:, -remainder:]], dim=1)
            data["contact"] = torch.cat([data["contact"], torch.zeros((1, remainder, data["contact"].shape[2])).to(data["contact"].device)], dim=1)
            data["trans"] = torch.cat([data["trans"], torch.zeros((1, remainder, data["trans"].shape[2])).to(data["trans"].device)], dim=1)
            data["facial"] = torch.cat([data["facial"], torch.zeros((1, remainder, data["facial"].shape[2])).to(data["facial"].device)], dim=1)
            data["beta"] = torch.cat([data["beta"], torch.zeros((1, remainder, data["beta"].shape[2])).to(data["beta"].device)], dim=1)

            # pad with silence (not zeroes) to the end of the audio
            noise_level = 1e-6  # Very small amplitude (~-120 dB)
            silence = torch.randn(1, remainder * 16000 // cfg.motion_fps, device=data["raw_audio"].device) * noise_level
            data["raw_audio"] = torch.cat([data["raw_audio"], silence], dim=1)

            data["word"] = torch.cat([data["word"], torch.zeros((1, remainder, data["word"].shape[2])).to(data["word"].device)], dim=1)
            data["speaker_ids"] = torch.cat([data["speaker_ids"], data["speaker_ids"][:, -remainder:]], dim=1)

        previous_latent = None

        pred_sample_motion = None
        pred_sample_facial = None
        pred_sample_transl = None

        gt_sample_motion = None
        gt_sample_facial = None
        gt_sample_transl = None
        

        for cidx, (chunk_start, chunk_end) in enumerate(zip(chunk_starts, chunk_ends)):
            # if cidx == len(chunk_starts) - 1:
            #     breakpoint()

            chunk_timestart = chunk_start / cfg.motion_fps
            chunk_timeend = chunk_end / cfg.motion_fps
            chunk_data = dict()
            chunk_data["motion"] = data["motion"][:, chunk_start:chunk_end]
            chunk_data["motion_upper"] = data["motion_upper"][:, chunk_start:chunk_end]
            chunk_data["motion_lower"] = data["motion_lower"][:, chunk_start:chunk_end]
            chunk_data["motion_face"] = data["motion_face"][:, chunk_start:chunk_end]
            chunk_data["motion_hands"] = data["motion_hands"][:, chunk_start:chunk_end]
            chunk_data["motion_length"] = [motion_seqlen] * len(data["motion_length"])
            chunk_data["motion_mask"] = data["motion_mask"][:, chunk_start:chunk_end]
            chunk_data["contact"] = data["contact"][:, chunk_start:chunk_end]
            chunk_data["trans"] = data["trans"][:, chunk_start:chunk_end]
            chunk_data["facial"] = data["facial"][:, chunk_start:chunk_end]
            chunk_data["beta"] = data["beta"][:, chunk_start:chunk_end]

            
            assert chunk_data["motion"].shape[1] == motion_seqlen

            audio_start = int(chunk_timestart * 16000)
            audio_end = int(chunk_timeend * 16000)
            chunk_data["raw_audio"] = data["raw_audio"][:, audio_start:audio_end]
            audio_chunk = chunk_data["raw_audio"][0].numpy()
            audio_features = get_wav2vec2_feature(data_wav2vec2_model, data_wav2vec2_processor, audio_chunk)
            chunk_data["audio"] = audio_features.unsqueeze(0)

            assert chunk_data["audio"].shape[1] == 499
            assert chunk_data["raw_audio"].shape[1] == 16000 * (motion_seqlen / cfg.motion_fps)
            # breakpoint()
            chunk_data["word"] = data["word"][:, chunk_start:chunk_end]
            relevant_textsegs = []
            for textseg in data["text_segments"][0]:
                text_seg_start = textseg[0][0]
                text_seg_end = textseg[0][1]
                if text_seg_start >= chunk_timestart and text_seg_end <= chunk_timeend:
                    new_textseg = [[text_seg_start - chunk_timestart, text_seg_end - chunk_timestart], textseg[1]]
                    relevant_textsegs.append(new_textseg)
            chunk_data["text_segments"] = [relevant_textsegs]
            chunktext, chunk_textfeatures = get_text_feature(data_bert_model, data_bert_tokenizer, train_dataset, relevant_textsegs)      
            chunk_data["raw_word"] = chunktext
            chunk_data["text_features"] = chunk_textfeatures  

            chunk_data["speaker_ids"] = data["speaker_ids"][:, chunk_start:chunk_end]
            chunk_data["sample_name"] = [data["sample_name"][0].replace("/0", f"/{cidx}")]

            relevant_glabels = []
            for glabel in data["gesture_labels"][0]:
                glabel_start = glabel["start"]
                glabel_end = glabel["end"]
                if glabel_start >= chunk_timestart and glabel_end <= chunk_timeend:
                    new_glabel = {"start": glabel_start - chunk_timestart, "end": glabel_end - chunk_timestart, "name": glabel["name"], 'word': glabel["word"]}
                    relevant_glabels.append(new_glabel)
            chunk_data["gesture_labels"] = [relevant_glabels]

            relevent_discourse = []
            for disc in data["discourse"][0]:
                conn = disc[0]
                sense = disc[1]
                arg1 = disc[2]
                arg2 = disc[3]
                rel_start = disc[4]
                rel_end = disc[5]
                conn_start = disc[6]
                conn_end = disc[7]
                if rel_start >= chunk_timestart and rel_end <= chunk_timeend:
                    new_disc = (conn, sense, arg1, arg2, rel_start - chunk_timestart, rel_end - chunk_timestart, conn_start - chunk_timestart, conn_end - chunk_timestart)
                    relevent_discourse.append(new_disc)
            chunk_data["discourse"] = [relevent_discourse]

            relevent_prominence = []
            for prom in data["prominence"][0]:
                prom_word = prom[0]
                prom_start = prom[1]
                prom_end = prom[2]
                prom_score = prom[3]
                if prom_start >= chunk_timestart and prom_end <= chunk_timeend:
                    new_prom = (prom_word, prom_start - chunk_timestart, prom_end - chunk_timestart, prom_score)
                    relevent_prominence.append(new_prom)
            chunk_data["prominence"] = [relevent_prominence]

            chunk_data["sample_idx"] = [data["sample_idx"]] 

            chunk_data["retrieval_method"] = data["retrieval_method"]
            # breakpoint()

            
            with torch.no_grad():
                chunk_data["inference_kwargs"] = {
                    "use_inversion": args.use_inversion,
                    "outpaint": args.outpaint,
                    "inversion_start_time": args.inversion_start_time,
                    "visualize_inversion": args.visualize_inversion,
                    "insertion_guidance": args.use_insertion_guidance,
                    "guidance_iters": args.guidance_iters,
                    "guidance_lr": args.guidance_lr,
                    "use_prev_latent": True,
                    "prev_latent": previous_latent,
                }
                # output_list = []
                output = model(**chunk_data)
                previous_latent = output["prev_latentout"]
                re_dict = output["retrieval_dict"]

            # breakpoint()
            
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
            # breakpoint()


            # blend motion on the overlap frames
            if cidx > 0:
                bs, n, dim = pred_motion.shape
                
                n_joints = dim // 3

                prev_length = pred_sample_motion.shape[1]

                # predicted motion
                # breakpoint() # check if the correct indices are selected by below 6 lines
                previous_pred_motion = pred_sample_motion[:, :-overlap_framelen, :]
                previous_pred_facial = pred_sample_facial[:, :-overlap_framelen, :]
                previous_pred_trans = pred_sample_transl[:, :-overlap_framelen, :]

                overlapprev_pred_motion = pred_sample_motion[:, -overlap_framelen:, :]
                overlapprev_pred_facial = pred_sample_facial[:, -overlap_framelen:, :]
                overlapprev_pred_trans = pred_sample_transl[:, -overlap_framelen:, :]

                pred_motion = rc.axis_angle_to_matrix(pred_motion.reshape(bs, n, n_joints, 3))
                pred_motion = rc.matrix_to_rotation_6d(pred_motion).reshape(bs, n, n_joints*6)

                overlapprev_pred_motion = rc.axis_angle_to_matrix(overlapprev_pred_motion.reshape(bs, overlap_framelen, n_joints, 3))
                overlapprev_pred_motion = rc.matrix_to_rotation_6d(overlapprev_pred_motion).reshape(bs, overlap_framelen, n_joints*6)


                new_weights = torch.linspace(0, 1, overlap_framelen).unsqueeze(0).unsqueeze(-1).to(pred_motion.device)
                prev_weights = 1 - new_weights

                # blended_pred_motion = torch.zeros_like(pred_motion[:, :overlap_framelen, :])
                blended_pred_motion = (overlapprev_pred_motion * prev_weights + pred_motion[:, :overlap_framelen, :] * new_weights)
                pred_motion[:, :overlap_framelen, :] = blended_pred_motion

                # blended_pred_facial = torch.zeros_like(pred_facial[:, :overlap_framelen, :])
                blended_pred_facial = (overlapprev_pred_facial * prev_weights + pred_facial[:, :overlap_framelen, :] * new_weights)
                pred_facial[:, :overlap_framelen, :] = blended_pred_facial

                # blended_pred_trans = torch.zeros_like(pred_trans[:, :overlap_framelen, :])
                blended_pred_trans = (overlapprev_pred_trans * prev_weights + pred_trans[:, :overlap_framelen, :] * new_weights)
                pred_trans[:, :overlap_framelen, :] = blended_pred_trans

                pred_motion = rc.rotation_6d_to_matrix(pred_motion.reshape(bs, n, n_joints, 6))
                pred_motion = rc.matrix_to_axis_angle(pred_motion).reshape(bs, n, n_joints*3)

                pred_sample_motion = torch.cat([previous_pred_motion, pred_motion], dim=1)
                pred_sample_facial = torch.cat([previous_pred_facial, pred_facial], dim=1)
                pred_sample_transl = torch.cat([previous_pred_trans, pred_trans], dim=1)
                assert pred_sample_motion.shape[1] == prev_length + motion_seqlen - overlap_framelen
                assert pred_sample_facial.shape[1] == prev_length + motion_seqlen - overlap_framelen
                assert pred_sample_transl.shape[1] == prev_length + motion_seqlen - overlap_framelen

                # ground truth motion
                previous_gt_motion = gt_sample_motion[:, :-overlap_framelen, :]
                previous_gt_facial = gt_sample_facial[:, :-overlap_framelen, :]
                previous_gt_trans = gt_sample_transl[:, :-overlap_framelen, :]

                overlapprev_gt_motion = gt_sample_motion[:, -overlap_framelen:, :]
                overlapprev_gt_facial = gt_sample_facial[:, -overlap_framelen:, :]
                overlapprev_gt_trans = gt_sample_transl[:, -overlap_framelen:, :]

                gt_motion = rc.axis_angle_to_matrix(gt_motion.reshape(bs, n, n_joints, 3))
                gt_motion = rc.matrix_to_rotation_6d(gt_motion).reshape(bs, n, n_joints*6)

                overlapprev_gt_motion = rc.axis_angle_to_matrix(overlapprev_gt_motion.reshape(bs, overlap_framelen, n_joints, 3))
                overlapprev_gt_motion = rc.matrix_to_rotation_6d(overlapprev_gt_motion).reshape(bs, overlap_framelen, n_joints*6)

                new_weights = torch.linspace(0, 1, overlap_framelen).unsqueeze(0).unsqueeze(-1).to(gt_motion.device)
                prev_weights = new_weights.flip(1)
                # blended_gt_motion = torch.zeros_like(gt_motion[:, :overlap_framelen, :])
                blended_gt_motion = (overlapprev_gt_motion * prev_weights + gt_motion[:, :overlap_framelen, :] * new_weights)
                gt_motion[:, :overlap_framelen, :] = blended_gt_motion

                # blended_gt_facial = torch.zeros_like(gt_facial[:, :overlap_framelen, :])
                blended_gt_facial = (overlapprev_gt_facial * prev_weights + gt_facial[:, :overlap_framelen, :] * new_weights)
                gt_facial[:, :overlap_framelen, :] = blended_gt_facial

                # blended_gt_trans = torch.zeros_like(gt_trans[:, :overlap_framelen, :])
                blended_gt_trans = (overlapprev_gt_trans * prev_weights + gt_trans[:, :overlap_framelen, :] * new_weights)
                gt_trans[:, :overlap_framelen, :] = blended_gt_trans

                gt_motion = rc.rotation_6d_to_matrix(gt_motion.reshape(bs, n, n_joints, 6))
                gt_motion = rc.matrix_to_axis_angle(gt_motion).reshape(bs, n, n_joints*3)

                gt_sample_motion = torch.cat([previous_gt_motion, gt_motion], dim=1)
                gt_sample_facial = torch.cat([previous_gt_facial, gt_facial], dim=1)
                gt_sample_transl = torch.cat([previous_gt_trans, gt_trans], dim=1)

                assert gt_sample_motion.shape[1] == prev_length + motion_seqlen - overlap_framelen
                assert gt_sample_facial.shape[1] == prev_length + motion_seqlen - overlap_framelen
                assert gt_sample_transl.shape[1] == prev_length + motion_seqlen - overlap_framelen


            else:
                # breakpoint()
                pred_sample_motion = pred_motion.clone()
                pred_sample_facial = pred_facial.clone()
                pred_sample_transl = pred_trans.clone()

                gt_sample_motion = gt_motion.clone()
                gt_sample_facial = gt_facial.clone()
                gt_sample_transl = gt_trans.clone()


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

                

                if re_dict is not None:
                    re_motions_batch = re_dict["raw_motion"]
                    re_facial_batch = re_dict["raw_facial"]
                    re_trans_batch = re_dict["raw_trans"]

                    # breakpoint()

                    re_motions_batch = re_motions_batch.squeeze(1)
                    re_facial_batch = re_facial_batch.squeeze(1)
                    re_trans_batch = re_trans_batch.squeeze(1)

                    re_motions_batch = rc.axis_angle_to_matrix(re_motions_batch.reshape(bs, n, n_joints, 3))
                    re_motions_batch = rc.matrix_to_rotation_6d(re_motions_batch).reshape(bs, n, n_joints*6)

                    # breakpoint()

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

            
            gt_sample_name = chunk_data["sample_name"]
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
                    f.write(chunk_data["raw_word"][j])
                sf.write(
                    osp.join(exp_dir, gt_sample_name[j], "gt_audio.wav"), chunk_data["raw_audio"].numpy()[j], 16000
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

                # breakpoint()
                # crop out the non-zero part

                # get start and end idx of all non-zero rows
                # start_idx = np.where(re_motions[k].sum(axis=1) != 0)[0][0]
                # end_idx = np.where(re_motions[k].sum(axis=1) != 0)[0][-1]


                cropped_motion = re_motions[k] #[re_motions[k].sum(axis=1) != 0]
                cropped_facial = re_facial[k] #[re_facial[k].sum(axis=1) != 0]
                cropped_trans = re_trans[k] #[re_trans[k].sum(axis=1) != 0]

                
                
                np.savez(osp.join(exp_dir, gt_sample_name[j], f"retrieval_{k}.npz"),
                    betas=np.zeros(300,),
                    poses=cropped_motion,
                    expressions=cropped_facial,
                    trans=cropped_trans,
                    model='smplx2020',
                    gender='neutral',
                    mocap_frame_rate = 30 #self.args.pose_fps ,
                )
                


        # handle the larger motion
        # breakpoint() # check shapes oif larger motion
        if (30/cfg.motion_fps) != 1:
            assert 30%cfg.motion_fps == 0
            bs, n, dim = pred_sample_motion.shape
            new_n = n * int(30/cfg.motion_fps)
            
            n_joints = dim // 3

            pred_sample_motion = rc.axis_angle_to_matrix(pred_sample_motion.reshape(bs, n, n_joints, 3))
            pred_sample_motion = rc.matrix_to_rotation_6d(pred_sample_motion).reshape(bs, n, n_joints*6)

            gt_sample_motion = rc.axis_angle_to_matrix(gt_sample_motion.reshape(bs, n, n_joints, 3))
            gt_sample_motion = rc.matrix_to_rotation_6d(gt_sample_motion).reshape(bs, n, n_joints*6)

            pred_sample_motion = torch.nn.functional.interpolate(pred_sample_motion.permute(0, 2, 1), scale_factor=30/cfg.motion_fps, mode='linear').permute(0,2,1)
            pred_sample_facial = torch.nn.functional.interpolate(pred_sample_facial.permute(0, 2, 1), scale_factor=30/cfg.motion_fps, mode='linear').permute(0,2,1)
            pred_sample_transl = torch.nn.functional.interpolate(pred_sample_transl.permute(0, 2, 1), scale_factor=30/cfg.motion_fps, mode='linear').permute(0,2,1)

            gt_sample_motion = torch.nn.functional.interpolate(gt_sample_motion.permute(0, 2, 1), scale_factor=30/cfg.motion_fps, mode='linear').permute(0,2,1)
            gt_sample_facial = torch.nn.functional.interpolate(gt_sample_facial.permute(0, 2, 1), scale_factor=30/cfg.motion_fps, mode='linear').permute(0,2,1)
            gt_sample_transl = torch.nn.functional.interpolate(gt_sample_transl.permute(0, 2, 1), scale_factor=30/cfg.motion_fps, mode='linear').permute(0,2,1)

            
            pred_sample_motion = rc.rotation_6d_to_matrix(pred_sample_motion.reshape(bs, new_n, n_joints, 6))
            pred_sample_motion = rc.matrix_to_axis_angle(pred_sample_motion).reshape(bs, new_n, n_joints*3)

            gt_sample_motion = rc.rotation_6d_to_matrix(gt_sample_motion.reshape(bs, new_n, n_joints, 6))
            gt_sample_motion = rc.matrix_to_axis_angle(gt_sample_motion).reshape(bs, new_n, n_joints*3)

        sample_motion_len = sample_motion_len * int(30/cfg.motion_fps)

        # breakpoint() # check shapes of larger motion probably would need to squeeze the batch size
        pred_sample_motion = pred_sample_motion[0, :sample_motion_len].cpu().detach().numpy()
        pred_sample_facial = pred_sample_facial[0, :sample_motion_len].cpu().detach().numpy()
        pred_sample_transl = pred_sample_transl[0, :sample_motion_len].cpu().detach().numpy()

        gt_sample_motion = gt_sample_motion[0, :sample_motion_len].cpu().detach().numpy()
        gt_sample_facial = gt_sample_facial[0, :sample_motion_len].cpu().detach().numpy()
        gt_sample_transl = gt_sample_transl[0, :sample_motion_len].cpu().detach().numpy()

        
        gt_sample_name = data["sample_name"][0].replace("/0", "")

        # os.makedirs(osp.join(exp_dir, gt_sample_name[j]), exist_ok=True)
        print(f"finalizing  {exp_dir}/{gt_sample_name}")
        # breakpoint()
        np.savez(osp.join(exp_dir, gt_sample_name, "full_pred_motion.npz"),
                betas=np.zeros(300,),
                poses=pred_sample_motion,
                expressions=pred_sample_facial,
                trans=pred_sample_transl,
                model='smplx2020',
                gender='neutral',
                mocap_frame_rate = 30 #self.args.pose_fps ,
            )
        
        
        np.savez(osp.join(exp_dir, gt_sample_name, "full_gt_motion.npz"),
                betas=np.zeros(300,),
                poses=gt_sample_motion,
                expressions=gt_sample_facial,
                trans=gt_sample_transl,
                model='smplx2020',
                gender='neutral',
                mocap_frame_rate = 30 #self.args.pose_fps ,
            )
        with open(
            osp.join(exp_dir, gt_sample_name, "gt_text.txt"),
            "w",
            encoding="utf-8",
        ) as f:
            f.write(data["raw_word"][0])
        sf.write(
            osp.join(exp_dir, gt_sample_name, "gt_audio.wav"), ori_audio.numpy()[0], 16000
        )
        # break


if __name__ == "__main__":
    main()
