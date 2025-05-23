import torch
import torch.nn as nn
import torch.nn.functional as F
import time

from .base_architecture import BaseArchitecture
from ..builder import ARCHITECTURES, build_architecture, build_submodule, build_loss
from ..utils.gaussian_diffusion import (
    GaussianDiffusion,
    get_named_beta_schedule,
    create_named_schedule_sampler,
    ModelMeanType,
    ModelVarType,
    LossType,
    space_timesteps,
    SpacedDiffusion,
)
from ..utils import rotation_conversions as rc

import copy

torch.autograd.set_detect_anomaly(True)


def build_diffusion(cfg):
    beta_scheduler = cfg["beta_scheduler"]
    diffusion_steps = cfg["diffusion_steps"]
    classifier_free_guidance_scale = cfg.get("classifier_free_guidance_scale", 0)

    betas = get_named_beta_schedule(beta_scheduler, diffusion_steps)
    model_mean_type = {
        "start_x": ModelMeanType.START_X,
        "previous_x": ModelMeanType.PREVIOUS_X,
        "epsilon": ModelMeanType.EPSILON,
        "v_pred": ModelMeanType.V_PRED,
    }[cfg["model_mean_type"]]
    model_var_type = {
        "learned": ModelVarType.LEARNED,
        "fixed_small": ModelVarType.FIXED_SMALL,
        "fixed_large": ModelVarType.FIXED_LARGE,
        "learned_range": ModelVarType.LEARNED_RANGE,
    }[cfg["model_var_type"]]
    if cfg.get("respace", None) is not None:
        num_inference_timesteps = cfg.get("num_inference_timesteps", None)
        diffusion = SpacedDiffusion(
            use_timesteps=space_timesteps(diffusion_steps, cfg["respace"],num_inference_timesteps),
            betas=betas,
            model_mean_type=model_mean_type,
            model_var_type=model_var_type,
            loss_type=LossType.MSE,
            classifier_free_guidance_scale=classifier_free_guidance_scale,
        )
    else:
        diffusion = GaussianDiffusion(
            betas=betas,
            model_mean_type=model_mean_type,
            model_var_type=model_var_type,
            loss_type=LossType.MSE,
            classifier_free_guidance_scale=classifier_free_guidance_scale,
        )
    return diffusion


@ARCHITECTURES.register_module()
class MotionDiffusion(BaseArchitecture):

    def __init__(
        self,
        model=None,
        loss_recon=None,
        loss_gen=None,
        loss_contact=None,
        loss_laplace=None,
        diffusion_train=None,
        diffusion_test=None,
        init_cfg=None,
        inference_type="ddpm",
        genloss_acceleration_weight=True,
        genloss_hands_weight=2,
        genloss_smooth=True,
        body_part_lossweights=None,
        **kwargs
    ):
        super().__init__(init_cfg=init_cfg)
        
        self.loss_recon = build_loss(loss_recon)
        self.loss_contact = build_loss(loss_contact) if loss_contact is not None else None
        self.loss_gen = build_loss(loss_gen) if loss_gen is not None else None
        self.loss_laplace = (
            build_loss(loss_laplace) if loss_laplace is not None else None
        )

        if self.loss_contact is not None:
            # import smplx
            dataset = kwargs.get("database", None)
            self.body_part_masks = {
                "upper": dataset.upper_mask.copy(),
                "lower": dataset.lower_mask.copy(),
                "face": dataset.face_mask.copy(),
                "hands": dataset.hands_mask.copy(),
            }
            self.smplx = dataset.smplx

        self.model = build_submodule(model, **kwargs)


        self.diffusion_train = build_diffusion(diffusion_train)
        self.diffusion_test = build_diffusion(diffusion_test)
        self.sampler = create_named_schedule_sampler("uniform", self.diffusion_train)
        self.inference_type = inference_type
        self.genloss_acceleration_weight = genloss_acceleration_weight
        self.genloss_hands_weight = genloss_hands_weight
        self.genloss_smooth = genloss_smooth
        self.body_part_lossweights = body_part_lossweights
        

    def forward(self, **kwargs):
        # DATA is passed here as kwargs
        motion, motion_mask = kwargs["motion"].float(), kwargs["motion_mask"].float()
        
        sample_idx = kwargs.get("sample_idx", None)
        sample_name = kwargs.get("sample_name", None)

        motion_framelen = motion.shape[1]

        if self.model.gesture_rep_encoder is not None:
            motion_upper = kwargs["motion_upper"]
            motion_lower = kwargs["motion_lower"]
            motion_face = kwargs["motion_face"]
            motion_facial = kwargs["facial"]
            motion_hands = kwargs["motion_hands"]
            motion_transl = kwargs["trans"]
            motion_contact = kwargs["contact"]

            # vae class should get all motion parts and motion_mask in the forward function
            motion, motion_mask = self.model.gesture_rep_encoder.encode(
                motion_upper, motion_lower, motion_face, motion_hands, motion_transl, motion_facial, motion_contact, motion_mask
            )
        else:
            raise NotImplementedError("not supported")
            

        B, T = motion.shape[:2]
        
        
        upper_indices = list(range(0, (T-3)//4))
        hands_indices = list(range((T-3)//4 + 1, 2*(T-3)//4 + 1))
        face_indices = list(range(2*(T-3)//4 + 2, 3*(T-3)//4 + 2))
        lowertrans_indices = list(range(3*(T-3)//4 + 3, T))

        if self.model.gesture_rep_encoder is not None and \
            self.model.gesture_rep_encoder.body_part_cat_axis == "time":
            
            crossatt_bodypart_mask = torch.ones_like(motion_mask)
            spkatt_bodypart_mask = torch.ones_like(motion_mask)
            # skip cross attention on seperators
            sep_indices = [(T-3)//4, 2*(T-3)//4, 3*(T-3)//4]
            crossatt_bodypart_mask[:, sep_indices] = 0 
            spkatt_bodypart_mask[:, sep_indices] = 0

            # no stopping cross att on lowertrans_indices
            query_masks_per_cond = {
                "xf_text": crossatt_bodypart_mask,
                "xf_audio": crossatt_bodypart_mask,
                "xf_spk": spkatt_bodypart_mask,
            }
            lossweight_mask = torch.ones_like(motion_mask)
            if self.body_part_lossweights is not None:  
                lossweight_mask[:, upper_indices] = self.body_part_lossweights["upper"]
                # lossweight_mask[:, lower_indices] = self.body_part_lossweights["lower"]
                lossweight_mask[:, hands_indices] = self.body_part_lossweights["hands"]
                lossweight_mask[:, face_indices] = self.body_part_lossweights["face"]
                lossweight_mask[:, lowertrans_indices] = self.body_part_lossweights["lowertransl"]
        
        else:
            raise NotImplementedError("Only time axis is supported for body part categorization")

        if self.training:
            t, _ = self.sampler.sample(B, motion.device)
        
            output = self.diffusion_train.training_losses(
                model=self.model,
                x_start=motion,
                t=t,
                model_kwargs={
                    "motion_mask": motion_mask,
                    "motion_length": kwargs["motion_length"],
                    "audio": kwargs["audio"],
                    "raw_audio": kwargs["raw_audio"],
                    "text": kwargs["word"],
                    "raw_text": kwargs["raw_word"],
                    "text_features": kwargs["text_features"],
                    "text_times": kwargs["text_segments"],
                    "discourse": kwargs["discourse"],
                    "prominence": kwargs["prominence"],
                    "speaker_ids": kwargs["speaker_ids"],
                    "gesture_labels": kwargs["gesture_labels"],
                    "sample_idx": sample_idx,
                    "sample_name": sample_name,
                    "gen_loss": True if self.loss_gen is not None else False,
                    "query_mask": query_masks_per_cond,
                },
            )
            
            pred, target = output["pred"], output["target"]
            recon_loss = self.loss_recon(pred, target, reduction_override="none")
            masked_loss = recon_loss.mean(dim=-1) * motion_mask
            masked_loss = masked_loss * lossweight_mask
            recon_loss = (masked_loss).sum() / motion_mask.sum()
            loss = {"recon_loss": recon_loss}

            return loss
        else:
            inference_kwargs = kwargs.get("inference_kwargs", {})

            use_outpaint = inference_kwargs.pop("outpaint", False)
            use_inversion = inference_kwargs.pop("use_inversion", False)
            inversion_start_time = inference_kwargs.pop("inversion_start_time", -1)
            visualize_inversion = inference_kwargs.pop("visualize_inversion", False)
            use_insertion_guidance = inference_kwargs.pop("insertion_guidance", False)
            guidance_iters = inference_kwargs.pop("guidance_iters", [10]*50)
            guidance_lr = inference_kwargs.pop("guidance_lr", 0.1)

            use_prev_latent = inference_kwargs.pop("use_prev_latent", False)
            prev_latent = inference_kwargs.pop("prev_latent", None)

            if use_prev_latent:
                # assert prev_latent is not None
                assert not use_outpaint

            # sanity checks for the inference kwargs
            if use_outpaint:
                assert not use_inversion
                assert not use_insertion_guidance
                
            if use_inversion:
                assert not use_outpaint
                
            if use_insertion_guidance:
                assert not use_outpaint
                assert use_inversion

            print("Using outpainting: ", use_outpaint)
            print("Using inversion: ", use_inversion)
            print("Using insertion guidance: ", use_insertion_guidance)
            print("Using prev latent: ", use_prev_latent)
            if use_insertion_guidance:
                print("Guidance iters: ", guidance_iters)
                print("Guidance lr: ", guidance_lr)
            
            
            dim_pose = (
                motion.shape[-1] if self.model.gesture_rep_encoder is None else self.model.gesture_rep_encoder.vae_latent_dim # * 5 change for dim cat
            )
            
            kwargs.update({
                    "motion_mask": motion_mask,
                    "text": kwargs["word"],
                    "raw_text": kwargs["raw_word"],
                    "text_times": kwargs["text_segments"],
                })
            
            
            model_kwargs = self.model.get_precompute_condition(
                device=motion.device, **kwargs
            )
            
            model_kwargs["query_mask"] = query_masks_per_cond
            
            model_kwargs["motion_mask"] = motion_mask
            model_kwargs["sample_idx"] = sample_idx
            retrieval_dict = model_kwargs["re_dict"]
            results = kwargs
            
            results["retrieval_dict"] = copy.deepcopy(retrieval_dict)

            # gen_time = time.time()

            # outpainting:
            if use_outpaint:
                retrieval_motion_latents = retrieval_dict["raw_motion_latents"]
                assert retrieval_motion_latents.shape[1] == 1
                retrieval_motion_latents = retrieval_motion_latents.squeeze(1)

            # long form motion synthesis
            if use_prev_latent and prev_latent is not None:
                # breakpoint()
                # use the previous latent as the start point
                # zero out the all indices except the first in time and 
                # replace with the previous latent's last time step
                masked_prevlatent = torch.zeros_like(prev_latent)
                masked_prevlatent[:, upper_indices[0]] = prev_latent[:, upper_indices[-1]]
                masked_prevlatent[:, hands_indices[0]] = prev_latent[:, hands_indices[-1]]
                masked_prevlatent[:, face_indices[0]] = prev_latent[:, face_indices[-1]]
                masked_prevlatent[:, lowertrans_indices[0]] = prev_latent[:, lowertrans_indices[-1]]

                prev_latent = masked_prevlatent
                
                

            # invert the retrieval latents using ddim reverse sampling
            start_noise = None
            invlats_per_difft = None
            if use_inversion:
                # you will get three lists (retr_latents, retr_startends, query_startends)
                retr_startends_list = retrieval_dict["retr_startends"]
                query_startends_list = retrieval_dict["query_startends"]
                retr_uncropped_latent_list = retrieval_dict["retr_uncropped_latents"]
                
                assert self.inference_type == "ddim", "Inversion is only supported for ddim for now"

                # 0. generate a noise tensor of same shape as the query_latents
                start_noise = torch.randn((B, T, dim_pose), device=motion.device)
                latent_len = (T - 3) // 4

                if visualize_inversion:
                    all_inverted_latent_lists = []
                    start_recons_pairs = []

                if use_insertion_guidance:
                    invlats_per_batch = []
                
                for batch_idx, retr_startends in enumerate(retr_startends_list):
                    query_startends = query_startends_list[batch_idx]
                    retr_uncropped_lats = retr_uncropped_latent_list[batch_idx]

                    zero_invlat = [torch.zeros((1, T, dim_pose), device=motion.device) for _ in range(self.diffusion_test.num_timesteps)]

                    for q_idx in retr_uncropped_lats.keys():
                        
                        # 1. invert retr_latents using the model.ddim_reverse_sample and store the inverted latents
                        retr_uncrop_lat_dict = retr_uncropped_lats[q_idx]
                        retr_uncrop_lat = retr_uncrop_lat_dict["retr_motion_latent"] # 1, T, D
                        retr_model_kwargs = self.model.get_precompute_condition(device=motion.device,
                                                                                text=retr_uncrop_lat_dict["retr_text"],
                                                                                audio=retr_uncrop_lat_dict["retr_audio"],
                                                                                speaker_ids=retr_uncrop_lat_dict["retr_spkid"],
                                                                                re_dict=1, # not None
                                                                                )
                        
                        assert "re_dict" in retr_model_kwargs and "xf_out" in retr_model_kwargs
                        retr_model_kwargs["query_mask"] = {k: v[batch_idx:batch_idx+1] for k, v in query_masks_per_cond.items()}
                        retr_model_kwargs["motion_mask"] = retr_uncrop_lat_dict["retr_motion_mask"]

                        inverted_latent_list = self.diffusion_test.ddim_reverse_sample_loop(
                            self.model,
                            start_img=retr_uncrop_lat,
                            clip_denoised=False,
                            progress=False,
                            model_kwargs=retr_model_kwargs, # this should be specific to each retr_latent
                            eta = 0,
                            return_all_timesteps=True,
                            **inference_kwargs
                        )

                         
                        if visualize_inversion:
                            # just for testing and sanity checks
                            final_inverted_latent = inverted_latent_list[-1]

                            all_inverted_latent_lists.append(inverted_latent_list)
                            
                            # difference between inverted_latent_list[0] and start_img and it should increase with time
                            print("Difference between inverted_latent_list and start_img")
                            print([((retr_uncrop_lat - inverted_latent_list[xj])**2).mean(dim=(1, 2)).round().item() for xj in range(50)])

                            # test the reconstruction from the inverted latent
                            reconstructed_invlat = self.diffusion_test.ddim_sample_loop(
                                self.model,
                                shape=tuple(final_inverted_latent.shape),
                                noise=final_inverted_latent,
                                clip_denoised=False,
                                progress=False,
                                model_kwargs=retr_model_kwargs,
                                eta=0,
                                in_seq=None,
                                **inference_kwargs
                            )
                            # difference between reconstructed_invlat and final_inverted_latent should be small
                            print("Difference between reconstructed_invlat and final_inverted_latent")
                            print(((retr_uncrop_lat - reconstructed_invlat)**2).mean(dim=(1, 2)).item())
                            start_recons_pairs.append((retr_uncrop_lat, reconstructed_invlat))
                
                        
                        # # 2. take the inverted latents at a certain timestep
                        start_latents = inverted_latent_list[inversion_start_time] # default is -1 (full noise)
                
                        # 3. use retr_startends to crop out the start_latents at the timestep
                        # 4. use query_startends to replace the noise tensor with the cropped retr_latents
                        r_start_idx, r_end_idx = retr_startends[q_idx]
                        q_start_idx, q_end_idx = query_startends[q_idx]
                        assert r_end_idx - r_start_idx == q_end_idx - q_start_idx
                        
                        # upper body
                        upper_latents = start_latents[:, r_start_idx:r_end_idx]
                        start_noise[batch_idx:batch_idx+1, q_start_idx:q_end_idx] = upper_latents
                        # hands
                        hand_latents = start_latents[:,  latent_len + 1 + r_start_idx:  latent_len + 1 + r_end_idx]
                        start_noise[batch_idx:batch_idx+1,  latent_len + 1 + q_start_idx:  latent_len + 1 + q_end_idx] = hand_latents

                        if use_insertion_guidance:
                            for diff_t, inverted_latent_at_t in enumerate(inverted_latent_list): # this goes from 0 to 49 (clean to noisy)
                                # upper body
                                zero_invlat[diff_t][:, q_start_idx:q_end_idx] = inverted_latent_at_t[:, r_start_idx:r_end_idx]
                                # hands
                                zero_invlat[diff_t][:, latent_len + 1 + q_start_idx: latent_len + 1 + q_end_idx] = \
                                    inverted_latent_at_t[:, latent_len + 1 + r_start_idx: latent_len + 1 + r_end_idx]
                                
                    if use_insertion_guidance:
                        zero_invlat = torch.cat(zero_invlat, dim=0) # diffusion_ts x T x dim
                        invlats_per_batch.append(zero_invlat)
                                
                if use_insertion_guidance:
                    invlats_per_batch = torch.stack(invlats_per_batch, dim=0) # B x diffusion_ts x T x dim
                    invlats_per_difft = invlats_per_batch.permute(1, 0, 2, 3) # diffusion_ts x B x T x dim
                    if use_prev_latent and prev_latent is not None:
                        invlats_per_difft[:, :, upper_indices[0], :] = 0
                        invlats_per_difft[:, :, hands_indices[0], :] = 0
                        invlats_per_difft[:, :, face_indices[0], :] = 0
                        invlats_per_difft[:, :, lowertrans_indices[0], :] = 0
                        
                # 6. use the model.ddim_sample_loop to generate the output
            
            if self.inference_type == "ddpm":
                output = self.diffusion_test.p_sample_loop(
                    self.model,
                    (B, T, dim_pose),
                    clip_denoised=False,
                    progress=True,
                    model_kwargs=model_kwargs,
                    **inference_kwargs
                )

            elif use_insertion_guidance:
                output = self.diffusion_test.ddim_guided_sample_loop(
                    self.model,
                    (B, T, dim_pose),
                    noise=start_noise if use_inversion else None,
                    clip_denoised=False,
                    progress=False,
                    model_kwargs=model_kwargs,
                    eta=0,
                    in_seq=prev_latent if use_prev_latent else None,
                    guidance_iters = guidance_iters,
                    inverted_latent_list = invlats_per_difft,
                    guidance_lr = guidance_lr,
                    **inference_kwargs
                )
            elif use_prev_latent and not use_insertion_guidance:
                if not use_inversion:
                    print("only latent inversion without ret guidance isn't tested yet.")
                output = self.diffusion_test.ddim_sample_loop(
                    self.model,
                    (B, T, dim_pose),
                    noise=start_noise if use_inversion else None,
                    clip_denoised=False,
                    progress=False,
                    model_kwargs=model_kwargs,
                    eta=0,
                    in_seq=prev_latent,
                    **inference_kwargs
                )
            else:
                output = self.diffusion_test.ddim_sample_loop(
                    self.model,
                    (B, T, dim_pose),
                    noise=start_noise if use_inversion else None,
                    clip_denoised=False,
                    progress=False,
                    model_kwargs=model_kwargs,
                    eta=0,
                    in_seq=retrieval_motion_latents if use_outpaint else None,
                    **inference_kwargs
                )
            if getattr(self.model, "post_process") is not None:
                output = self.model.post_process(output)

            # 
            results["prev_latentout"] = output

            if self.model.gesture_rep_encoder is not None:
                output_upper, output_lower, output_face, output_hands, output_transl, output_exps, output_contact = self.model.gesture_rep_encoder.decode(output)
            else:
                raise NotImplementedError("not supported")

            # print("Time taken for only generation: ", time.time() - gen_time)
            
            if use_inversion and visualize_inversion:
                # concat all the one sized batches and then stack them
                # len(all_inverted_latent_lists) x diffusion_ts x T x dim
                batched_invlist = torch.stack([torch.cat(inv_list, dim=0) for inv_list in all_inverted_latent_lists], dim=0)
                batched_invlist = batched_invlist.reshape(len(all_inverted_latent_lists) * self.diffusion_test.num_timesteps,  T, dim_pose)
                if getattr(self.model, "post_process") is not None:
                    batched_invlist = self.model.post_process(batched_invlist)

                if batched_invlist.shape[0] > 128:
                    u_list = []
                    l_list = []
                    f_list = []
                    h_list = []
                    t_list = []
                    e_list = []
                    c_list = []
                    for i in range(0, batched_invlist.shape[0], 128):
                        smp_inverted_output_upper, smp_inverted_output_lower, smp_inverted_output_face, \
                            smp_inverted_output_hands, smp_inverted_output_transl, smp_inverted_output_exps, \
                                smp_inverted_output_contact = self.model.gesture_rep_encoder.decode(batched_invlist[i:i+128])
                        u_list.append(smp_inverted_output_upper)
                        l_list.append(smp_inverted_output_lower)
                        f_list.append(smp_inverted_output_face)
                        h_list.append(smp_inverted_output_hands)
                        t_list.append(smp_inverted_output_transl)
                        e_list.append(smp_inverted_output_exps)
                        c_list.append(smp_inverted_output_contact)
                    inverted_output_upper = torch.cat(u_list, dim=0)
                    inverted_output_lower = torch.cat(l_list, dim=0)
                    inverted_output_face = torch.cat(f_list, dim=0)
                    inverted_output_hands = torch.cat(h_list, dim=0)
                    inverted_output_transl = torch.cat(t_list, dim=0)
                    inverted_output_exps = torch.cat(e_list, dim=0)
                    inverted_output_contact = torch.cat(c_list, dim=0)

                else:
                    inverted_output_upper, inverted_output_lower, inverted_output_face, \
                        inverted_output_hands, inverted_output_transl, inverted_output_exps, \
                            inverted_output_contact = self.model.gesture_rep_encoder.decode(batched_invlist)
                    
                
                
                inverted_output_upper = inverted_output_upper.reshape(len(all_inverted_latent_lists), self.diffusion_test.num_timesteps, motion_framelen, -1)
                inverted_output_lower = inverted_output_lower.reshape(len(all_inverted_latent_lists), self.diffusion_test.num_timesteps, motion_framelen, -1)
                inverted_output_face = inverted_output_face.reshape(len(all_inverted_latent_lists), self.diffusion_test.num_timesteps, motion_framelen, -1)
                inverted_output_hands = inverted_output_hands.reshape(len(all_inverted_latent_lists), self.diffusion_test.num_timesteps, motion_framelen, -1)
                inverted_output_transl = inverted_output_transl.reshape(len(all_inverted_latent_lists), self.diffusion_test.num_timesteps, motion_framelen, -1)
                inverted_output_exps = inverted_output_exps.reshape(len(all_inverted_latent_lists), self.diffusion_test.num_timesteps, motion_framelen, -1)

                results["inverted_output_upper"] = inverted_output_upper
                results["inverted_output_lower"] = inverted_output_lower
                results["inverted_output_facepose"] = inverted_output_face
                results["inverted_output_hands"] = inverted_output_hands
                results["inverted_output_transl"] = inverted_output_transl
                results["inverted_output_exps"] = inverted_output_exps


                # ------------------- #
                # len(all_inverted_latent_lists) x 2 x T x dim
                batched_reconspair_list = torch.stack([torch.cat(recons_pair, dim=0) for recons_pair in start_recons_pairs], dim=0)
                batched_reconspair_list = batched_reconspair_list.reshape(len(start_recons_pairs) * 2, T, dim_pose)
                if getattr(self.model, "post_process") is not None:
                    batched_reconspair_list = self.model.post_process(batched_reconspair_list)

                reconspair_output_upper, reconspair_output_lower, reconspair_output_face, \
                    reconspair_output_hands, reconspair_output_transl, reconspair_output_exps, \
                        reconspair_output_contact = self.model.gesture_rep_encoder.decode(batched_reconspair_list)
                
                reconspair_output_upper = reconspair_output_upper.reshape(len(start_recons_pairs), 2, motion_framelen, -1)
                reconspair_output_lower = reconspair_output_lower.reshape(len(start_recons_pairs), 2, motion_framelen, -1)
                reconspair_output_face = reconspair_output_face.reshape(len(start_recons_pairs), 2, motion_framelen, -1)
                reconspair_output_hands = reconspair_output_hands.reshape(len(start_recons_pairs), 2, motion_framelen, -1)
                reconspair_output_transl = reconspair_output_transl.reshape(len(start_recons_pairs), 2, motion_framelen, -1)
                reconspair_output_exps = reconspair_output_exps.reshape(len(start_recons_pairs), 2, motion_framelen, -1)

                results["reconspair_output_upper"] = reconspair_output_upper
                results["reconspair_output_lower"] = reconspair_output_lower
                results["reconspair_output_facepose"] = reconspair_output_face
                results["reconspair_output_hands"] = reconspair_output_hands
                results["reconspair_output_transl"] = reconspair_output_transl
                results["reconspair_output_exps"] = reconspair_output_exps
            

            if self.model.gesture_rep_encoder is not None:
                results["pred_upper"] = output_upper
                results["pred_lower"] = output_lower
                results["pred_facepose"] = output_face
                results["pred_hands"] = output_hands
                results["pred_transl"] = output_transl
                results["pred_exps"] = output_exps
            else:
                raise NotImplementedError("not supported")

            
            return results
