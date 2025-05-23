#!/bin/bash
#SBATCH -p gpu20
#SBATCH --signal=B:SIGTERM@120
#SBATCH --gres gpu:1
#SBATCH -t 3-00:00:00
#SBATCH -o /CT/GestureSynth1/work/DiscourseAwareGesture/RAGGesture_BEATX/experiments/logs/slurm-%j.out
trap "trap ' ' TERM INT; kill -TERM 0; wait" TERM INT

eval "$(conda shell.bash hook)"

cd /CT/GestureSynth1/work/DiscourseAwareGesture/RAGGesture_BEATX

conda activate remodiffuse

# BASE DIFFUSION MODEL
# PYTHONPATH=".":$PYTHONPATH python tools/train.py configs/raggesture_beatx/basegesture_len150_beat.py --work-dir ./experiments/base_beatx_len150fps15_transmaskfix_simpleclfguide/ --no-validate
# PYTHONPATH=".":$PYTHONPATH python tools/train.py configs/raggesture_beatx/basegesture_len150_beat.py --work-dir ./experiments/base_beatx_len150fps15_transmaskfix_remodiffuseclfguide/ --no-validate
# PYTHONPATH=".":$PYTHONPATH python tools/train.py configs/raggesture_beatx/basegesture_len150_beat.py --work-dir ./experiments/base_beatx_len150fps15_transmaskfix_remodiffuseclfguide_hwtrans/ --no-validate
# PYTHONPATH=".":$PYTHONPATH python tools/train.py configs/raggesture_beatx/basegesture_len150_beat.py --work-dir ./experiments/base_beatx_len150fps15_transmaskfix_remodiffuseclfguide_contactlossw10/ --no-validate

# base model with one speaker
# PYTHONPATH=".":$PYTHONPATH python tools/train.py configs/raggesture_beatx/basegesture_len150_beat_spk2.py --work-dir ./experiments/base_beatxspk2_len150fps15_lowertrans4lats_remodiffuseclfguide_8l16h/ --no-validate
# continue above model
# PYTHONPATH=".":$PYTHONPATH python tools/train.py experiments/base_beatxspk2_len150fps15_lowertrans4lats_remodiffuseclfguide_8l16h/basegesture_len150_beat_spk2.py \
#     --work-dir ./experiments/base_beatxspk2_len150fps15_lowertrans4lats_remodiffuseclfguide_8l16h/ \
#     --resume-from  ./experiments/base_beatxspk2_len150fps15_lowertrans4lats_remodiffuseclfguide_8l16h/latest.pth \
#     --no-validate



# base model with lower trans
# PYTHONPATH=".":$PYTHONPATH python tools/train.py configs/raggesture_beatx/basegesture_len150_beat.py --work-dir ./experiments/base_beatx_len150fps15_lowertrans4lats_remodiffuseclfguide_8l16h_lowbs/ --no-validate

PYTHONPATH=".":$PYTHONPATH python tools/train.py experiments/base_beatx_len150fps15_lowertrans4lats_remodiffuseclfguide_8l16h/basegesture_len150_beat.py \
    --work-dir ./experiments/base_beatx_len150fps15_lowertrans4lats_remodiffuseclfguide_8l16h/ \
    --resume-from  ./experiments/base_beatx_len150fps15_lowertrans4lats_remodiffuseclfguide_8l16h/latest.pth \
    --no-validate



# RAG model with lowertrans
# PYTHONPATH=".":$PYTHONPATH python tools/train.py configs/raggesture_beatx/raggesture_len150_beat.py --work-dir ./experiments/rag_beatx_len150fps15_lowertrans4lats_remodiffuseclfguide_nogenloss/ --no-validate


# RAG model with lowertrans with gen loss
# PYTHONPATH=".":$PYTHONPATH python tools/train.py configs/raggesture_beatx/raggesture_len150_beat.py --work-dir ./experiments/rag_beatx_len150fps15_lowertrans4lats_remodiffuseclfguide_genloss/ --no-validate
# continue above model
PYTHONPATH=".":$PYTHONPATH python tools/train.py experiments/rag_beatx_len150fps15_lowertrans4lats_remodiffuseclfguide_genloss/raggesture_len150_beat.py \
    --work-dir ./experiments/rag_beatx_len150fps15_lowertrans4lats_remodiffuseclfguide_genloss/ \
    --resume-from  ./experiments/rag_beatx_len150fps15_lowertrans4lats_remodiffuseclfguide_genloss/latest.pth \
    --no-validate


# RAG GESTURE LIKE REMODIFFUSE
# PYTHONPATH=".":$PYTHONPATH python tools/train.py configs/raggesture_beatx/raggesture_len150_beat.py --work-dir ./experiments/rag_beatx_len150fps15_transmaskfix_remodiffuseclfguide_nogenloss/ --no-validate

# REMODIFFUSE NO LATENT remo RETRIEVAL
# PYTHONPATH=".":$PYTHONPATH python tools/train.py configs/raggesture_beatx/remodiffuse_len150_beat.py --work-dir ./experiments/remodiffuse_beatx_len150fps15_nolat_remoret_run2/ --no-validate

# REMODIFFUSE NO LATENT OUR RETRIEVAL GEN LOSS
# PYTHONPATH=".":$PYTHONPATH python tools/train.py configs/raggesture_beatx/remodiffuse_len150_beat_ourret.py --work-dir ./experiments/remodiffuse_beatx_len150fps15_nolat_ourret_genloss_run2/ --no-validate

# REMODIFFUSE NO LATENT OUR RETRIEVAL
# PYTHONPATH=".":$PYTHONPATH python tools/train.py configs/raggesture_beatx/remodiffuse_len150_beat_ourret.py --work-dir ./experiments/remodiffuse_beatx_len150fps15_nolat_ourret_run2/ --no-validate

wait