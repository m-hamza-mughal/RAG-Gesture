_base_ = ["../_base_/datasets/beatx_len150_15fps.py"]

# checkpoint saving
checkpoint_config = dict(interval=2)

dist_params = dict(backend="nccl")
log_level = "INFO"
load_from = None
resume_from = None
workflow = [("train", 1)]

# optimizer
optimizer = dict(type="Adam", lr=1e-4)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy="CosineAnnealing", min_lr_ratio=1e-6, by_epoch=False)
runner = dict(type="EpochBasedRunner", max_epochs=500)

log_config = dict(
    interval=10, hooks=[dict(type="TextLoggerHook"), dict(type="TensorboardLoggerHook")]
)

custom_hooks = [
    dict(type="VAE_FreezeHook"),
    # dict(type="BERT_FreezeHook"),
    dict(
        type="DatabaseSaveHook",
        save_dir="/CT/GestureSynth1/work/DiscourseAwareGesture/RAGGesture_BEATX/experiments/retrieval_dicts",
    ),
]

input_feats = 189
max_seq_len = 150
frame_chunk_size = 15
motion_fps = 15
latent_dim = 512
time_embed_dim = 2048
inp_text_latent_dim = 768
ff_size = 1024
num_heads = 16
num_layers = 8
dropout = 0

# model settings
model = dict(
    type="MotionDiffusion",
    model=dict(
        type="ReGestureTransformer",
        input_feats=input_feats,
        max_seq_len=max_seq_len,
        frame_chunk_size=frame_chunk_size,
        latent_dim=latent_dim,
        time_embed_dim=time_embed_dim,
        num_layers=num_layers,
        body_part_cat_axis = "time", # "dim"
        sa_block_cfg=dict(
            type="EfficientSelfAttention",
            latent_dim=latent_dim,
            num_heads=num_heads,
            dropout=dropout,
            time_embed_dim=time_embed_dim, # IMPORTANT TO SET THIS
        ),
        # ca_block_cfg=dict(
        #     type="SemanticsModulatedAttention",
        #     latent_dim=latent_dim,
        #     text_latent_dim=latent_dim,
        #     num_heads=num_heads,
        #     dropout=dropout,
        #     time_embed_dim=time_embed_dim,
        # ),
        ca_block_cfg=dict(
            type="EfficientCrossAttention",
            latent_dim=latent_dim,
            text_latent_dim=latent_dim,
            num_heads=num_heads,
            dropout=dropout,
            time_embed_dim=time_embed_dim,
        ),
        ffn_cfg=dict(
            latent_dim=latent_dim,
            ffn_dim=ff_size,
            dropout=dropout,
            time_embed_dim=time_embed_dim,
        ),
        vae_cfg=dict(
            upper_cfg = "/CT/GestureSynth1/work/GestureGPT/GestureRep/experiments/0903_020101_gesture_lexicon_transformer_vae_upper_allspk_len256_l8h4_fchunksize15/0903_020101_gesture_lexicon_transformer_vae_upper_allspk_len256_l8h4_fchunksize15.yaml",
            # lower_cfg = "/CT/GestureSynth1/work/GestureGPT/GestureRep/experiments/0909_131733_gesture_lexicon_transformer_vae_lower_allspk_len10s_l8h4_fchunksize15/0909_131733_gesture_lexicon_transformer_vae_lower_allspk_len10s_l8h4_fchunksize15.yaml",
            lowertrans_cfg = "/CT/GestureSynth1/work/GestureGPT/GestureRep/experiments/1031_142417_gesture_lexicon_transformer_vae_lowerplustrans_allspk_len10s_l8h8_fchunksize15_run2/1031_142417_gesture_lexicon_transformer_vae_lowerplustrans_allspk_len10s_l8h8_fchunksize15_run2.yaml",
            face_cfg = "/CT/GestureSynth1/work/GestureGPT/GestureRep/experiments/0909_130750_gesture_lexicon_transformer_vae_face_allspk_len10s_l8h4_fchunksize15/0909_130750_gesture_lexicon_transformer_vae_face_allspk_len10s_l8h4_fchunksize15.yaml",
            hands_cfg = "/CT/GestureSynth1/work/GestureGPT/GestureRep/experiments/0909_132647_gesture_lexicon_transformer_vae_hands_allspk_len10s_l8h4_fchunksize15/0909_132647_gesture_lexicon_transformer_vae_hands_allspk_len10s_l8h4_fchunksize15.yaml",
            # transl_cfg = "/CT/GestureSynth1/work/GestureGPT/GestureRep/experiments/1001_173313_gesture_lexicon_transformer_vae_trans_allspk_len10s_l8h8_fchunksize15_xznorm_ldim512_lowkl/1001_173313_gesture_lexicon_transformer_vae_trans_allspk_len10s_l8h8_fchunksize15_xznorm_ldim512_lowkl.yaml",
            latent_dim=latent_dim,
            frame_chunk_size=frame_chunk_size,
        ),
        text_encoder=dict(
            pretrained_model=None, #"bert",
            latent_dim=inp_text_latent_dim,
            num_layers=0, #2, # no transformer layers
            ff_size=2048,
            dropout=dropout,
            use_text_proj=False,
        ),
        audio_encoder=dict(
            pretrained_model=None, # "wav2vec",
            latent_dim=inp_text_latent_dim,
            num_layers=0, #2, # no transformer layers
            dropout=0.1,
            # cp_path="/CT/GroupGesture/work/DiscourseAwareGesture/fairseq/weights/wav2vec_large.pt",
        ),
        speaker_embedding=dict(num_speakers=25), # beatx has 25 speakers instead of 30
        retrieval_train=False,
        retrieval_cfg=dict(
            motion_feat_dim=input_feats,
            num_retrieval=1,
            stride=4,
            num_layers=2,
            num_motion_layers=2,
            kinematic_coef=0.1,
            topk=2,
            # retrieval_file='data/database/t2m_text_train.npz',
            latent_dim=latent_dim,
            text_latent_dim=inp_text_latent_dim,
            output_dim=latent_dim,
            max_seq_len=max_seq_len,
            motion_fps=motion_fps,
            motion_framechunksize=frame_chunk_size,
            num_heads=num_heads,
            ff_size=ff_size,
            dropout=dropout,
            lmdb_paths="/scratch/inf0/user/mmughal/DiscourseAwareGesture/retrieval_cache_stratified/",
            new_lmdb_cache=False,
            stratified_db_creation=True,
            stratification_interval=15, # (max_seq_len // data_stride) // 2) = 150 // 5 // 2 = 15
            ffn_cfg=dict(
                latent_dim=latent_dim,
                ffn_dim=ff_size,
                dropout=dropout,
            ),
            sa_block_cfg=dict(
                type="EfficientSelfAttention",
                latent_dim=latent_dim,
                num_heads=num_heads,
                dropout=dropout,
            ),
        ),
        scale_func_cfg=dict(
            coarse_scale=6.5, both_coef=0.52351, text_coef=-0.28419, retr_coef=2.39872
        ),
    ),
    loss_recon=dict(type="MSELoss", loss_weight=1, reduction="none"),
    # body_part_lossweights=dict(upper=1.0,lower=1.0,hands=1.0,face=1.0,transl=1.0),
    body_part_lossweights=dict(upper=1.0,hands=1.0,face=1.0,lowertransl=1.0),
    # loss_contact=dict(type="MSELoss", loss_weight=10, reduction="none"),
    # loss_gen=dict(
    #     type="MSELoss",
    #     loss_weight=1,
    #     reduction="none",
    # ),
    # genloss_smooth=False,
    diffusion_train=dict(
        beta_scheduler="scaled_linear",
        diffusion_steps=1000,
        # model_mean_type="epsilon",
        # model_var_type="fixed_small",
        model_mean_type="start_x",
        model_var_type="fixed_large",
    ),
    diffusion_test=dict(
        beta_scheduler="scaled_linear",
        diffusion_steps=1000,
        # model_mean_type="epsilon",
        # model_var_type="fixed_small",
        model_mean_type="start_x",
        model_var_type="fixed_large",
        respace="15,15,8,6,6", # can also be leading, trailing, 
        num_inference_timesteps=50,
        classifier_free_guidance_scale=0,
    ),
    inference_type="ddim",
)
