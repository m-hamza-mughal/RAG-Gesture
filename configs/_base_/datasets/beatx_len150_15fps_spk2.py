data_keys = [
    "motion",
    "motion_mask",
    "motion_length",
    "melspec",
    "audio",
    "text",
    "gest_types",
    "discourse",
    "speaker_id",
]
# meta_keys = ['text', 'token']
train_pipeline = [
    # dict(type='Collect', keys=data_keys)
]

motion_length = 150
motion_fps = 15
audio_sr = 16000

base_data_cfg = dict(
    type="BEATXDataset",
    pose_rep="smplxflame_30",
    facial_rep="smplxflame_30",
    data_path="/CT/GestureSynth1/work/GestureGPT/PantoMatrix/BEAT2/beat_english_v2.0.0/",
    cache_path="/scratch/inf0/user/mmughal/DiscourseAwareGesture/dataset_cache/beatx_cache_spk2/", # 
    debug=False,
    tiny=False,
    face_joint_idx=[18, 13, 9, 5],
    sample_rate=audio_sr,
    num_mels=80,
    hop_length=512,
    fps=motion_fps,
    stride=5,
    pose_length=motion_length,
    ori_joints="beat_smplx_joints",
    deps_path="/CT/GestureSynth1/work/GestureGPT/GestureRep/deps/",
    training_speakers=[2],
    additional_data=True,
    pose_fps=motion_fps,
    audio_sr=audio_sr,
    audio_fps=audio_sr,
    new_cache=False,
    # beat_align=True,
    clean_first_seconds=0,
    clean_final_seconds=0,
    audio_rep="wav2vec", # melspec, wav2vec for mel change audio_fps
    word_rep="bert_framealigned", # bert_framealigned, bert, glove_framealigned
    id_rep="idx", # idx, onehot
    sem_rep="info", # info, score
    prom_rep="prom",
    emo_rep="emo",
)

train_cfg = base_data_cfg.copy()
train_cfg["split"] = "train"
val_cfg = base_data_cfg.copy()
val_cfg["split"] = "val"
test_cfg = base_data_cfg.copy()
test_cfg["split"] = "test"

data = dict(
    samples_per_gpu=128,
    workers_per_gpu=8,
    train=train_cfg,
    val=val_cfg,
    test=test_cfg,
)
