#!/bin/bash
#SBATCH -p gpu20
#SBATCH --signal=B:SIGTERM@120
#SBATCH -t 0-04:00:00
#SBATCH --gres gpu:1
#SBATCH -o /CT/GestureSynth1/work/DiscourseAwareGesture/RAGGesture_BEATX/experiments/logs/slurm-%j.out
trap "trap ' ' TERM INT; kill -TERM 0; wait" TERM INT

eval "$(conda shell.bash hook)"

cd /CT/GestureSynth1/work/DiscourseAwareGesture/RAGGesture_BEATX

conda activate remodiffuse

# PYTHONPATH=".":$PYTHONPATH python tools/visualize.py experiments/remolatdiffuse_ahd16lyr8_latdim256_sched-scalinear_sepattcond-audiotextspk_retr-maskdiscomasktextzerofix-lr1e-4-bs32-mlen256x189-scalenorm-retrloss_w1_smooth-all3full/remolatdiffuse256_beat.py experiments/remolatdiffuse_ahd16lyr8_latdim256_sched-scalinear_sepattcond-audiotextspk_retr-maskdiscomasktextzerofix-lr1e-4-bs32-mlen256x189-scalenorm-retrloss_w1_smooth-all3full/latest.pth --retrieval_method discourse --out ./results/

# cp -r beatx_cache/ /scratch/inf0/user/mmughal/DiscourseAwareGesture/dataset_cache/

# PYTHONPATH=".":$PYTHONPATH python tools/visualize_remodiffuse.py experiments/remodiffuse_beatx_len150fps15_nolat_ourret_genloss_run2/remodiffuse_len150_beat_ourret.py experiments/remodiffuse_beatx_len150fps15_nolat_ourret_genloss_run2/latest.pth --retrieval_method discourse


# PYTHONPATH=".":$PYTHONPATH python tools/visualize_remodiffuse.py experiments/remodiffuse_beatx_len150fps15_nolat_remoret_run2/remodiffuse_len150_beat.py experiments/remodiffuse_beatx_len150fps15_nolat_remoret_run2/latest.pth --retrieval_method remodiffuse


# inc guidance discourse incfrom25

# PYTHONPATH=".":$PYTHONPATH python tools/visualize.py experiments/base_beatx_len150fps15_lowertrans4lats_remodiffuseclfguide_8l16h/basegesture_len150_beat.py experiments/base_beatx_len150fps15_lowertrans4lats_remodiffuseclfguide_8l16h/latest.pth --retrieval_method discourse_guidance_incfrom25_run2 --use_retrieval --use_inversion --use_insertion_guidance --guidance_iters increasing_from_25

# llm short window guidance dec till 25

# PYTHONPATH=".":$PYTHONPATH python tools/visualize.py experiments/base_beatx_len150fps15_lowertrans4lats_remodiffuseclfguide_8l16h/basegesture_len150_beat.py experiments/base_beatx_len150fps15_lowertrans4lats_remodiffuseclfguide_8l16h/latest.pth --retrieval_method llm_guidance_dectill25_shortwindow --use_retrieval --use_inversion --use_insertion_guidance --guidance_iters decreasing_till_25

# speaker 2 only model
# dec guidance llm
# PYTHONPATH=".":$PYTHONPATH python tools/visualize.py experiments/base_beatxspk2_len150fps15_lowertrans4lats_remodiffuseclfguide_8l16h/basegesture_len150_beat_spk2.py experiments/base_beatxspk2_len150fps15_lowertrans4lats_remodiffuseclfguide_8l16h/latest.pth --retrieval_method llm_guidance_dectill25_shortwindow --use_retrieval --use_inversion --use_insertion_guidance --guidance_iters decreasing_till_25

# dec guidance discourse

# PYTHONPATH=".":$PYTHONPATH python tools/visualize.py experiments/base_beatxspk2_len150fps15_lowertrans4lats_remodiffuseclfguide_8l16h/basegesture_len150_beat_spk2.py experiments/base_beatxspk2_len150fps15_lowertrans4lats_remodiffuseclfguide_8l16h/latest.pth --retrieval_method discourse_guidance_dectill25 --use_retrieval --use_inversion --use_insertion_guidance --guidance_iters decreasing_till_25

# PYTHONPATH=".":$PYTHONPATH python tools/visualize.py experiments/base_beatx_len150fps15_lowertrans4lats_remodiffuseclfguide_8l16h/basegesture_len150_beat.py experiments/base_beatx_len150fps15_lowertrans4lats_remodiffuseclfguide_8l16h/latest.pth --retrieval_method llm_guidance_dectill25_mm1 --use_retrieval --use_inversion --use_insertion_guidance --guidance_iters decreasing_till_25 --seed 3455
# PYTHONPATH=".":$PYTHONPATH python tools/visualize.py experiments/base_beatx_len150fps15_lowertrans4lats_remodiffuseclfguide_8l16h/basegesture_len150_beat.py experiments/base_beatx_len150fps15_lowertrans4lats_remodiffuseclfguide_8l16h/latest.pth --retrieval_method llm_guidance_dectill25_mm2 --use_retrieval --use_inversion --use_insertion_guidance --guidance_iters decreasing_till_25 --seed 2333
# PYTHONPATH=".":$PYTHONPATH python tools/visualize.py experiments/base_beatx_len150fps15_lowertrans4lats_remodiffuseclfguide_8l16h/basegesture_len150_beat.py experiments/base_beatx_len150fps15_lowertrans4lats_remodiffuseclfguide_8l16h/latest.pth --retrieval_method llm_guidance_dectill25_mm3 --use_retrieval --use_inversion --use_insertion_guidance --guidance_iters decreasing_till_25 --seed 4567
# PYTHONPATH=".":$PYTHONPATH python tools/visualize.py experiments/base_beatx_len150fps15_lowertrans4lats_remodiffuseclfguide_8l16h/basegesture_len150_beat.py experiments/base_beatx_len150fps15_lowertrans4lats_remodiffuseclfguide_8l16h/latest.pth --retrieval_method llm_guidance_dectill25_mm4 --use_retrieval --use_inversion --use_insertion_guidance --guidance_iters decreasing_till_25 --seed 9875
# PYTHONPATH=".":$PYTHONPATH python tools/visualize.py experiments/base_beatx_len150fps15_lowertrans4lats_remodiffuseclfguide_8l16h/basegesture_len150_beat.py experiments/base_beatx_len150fps15_lowertrans4lats_remodiffuseclfguide_8l16h/latest.pth --retrieval_method llm_guidance_dectill25_mm5 --use_retrieval --use_inversion --use_insertion_guidance --guidance_iters decreasing_till_25 --seed 7643
# PYTHONPATH=".":$PYTHONPATH python tools/visualize.py experiments/base_beatx_len150fps15_lowertrans4lats_remodiffuseclfguide_8l16h/basegesture_len150_beat.py experiments/base_beatx_len150fps15_lowertrans4lats_remodiffuseclfguide_8l16h/latest.pth --retrieval_method llm_guidance_dectill25_mm6 --use_retrieval --use_inversion --use_insertion_guidance --guidance_iters decreasing_till_25 --seed 5789
# PYTHONPATH=".":$PYTHONPATH python tools/visualize.py experiments/base_beatx_len150fps15_lowertrans4lats_remodiffuseclfguide_8l16h/basegesture_len150_beat.py experiments/base_beatx_len150fps15_lowertrans4lats_remodiffuseclfguide_8l16h/latest.pth --retrieval_method llm_guidance_dectill25_mm7 --use_retrieval --use_inversion --use_insertion_guidance --guidance_iters decreasing_till_25 --seed 6807
# PYTHONPATH=".":$PYTHONPATH python tools/visualize.py experiments/base_beatx_len150fps15_lowertrans4lats_remodiffuseclfguide_8l16h/basegesture_len150_beat.py experiments/base_beatx_len150fps15_lowertrans4lats_remodiffuseclfguide_8l16h/latest.pth --retrieval_method llm_guidance_dectill25_mm8 --use_retrieval --use_inversion --use_insertion_guidance --guidance_iters decreasing_till_25 --seed 3455
# PYTHONPATH=".":$PYTHONPATH python tools/visualize.py experiments/base_beatx_len150fps15_lowertrans4lats_remodiffuseclfguide_8l16h/basegesture_len150_beat.py experiments/base_beatx_len150fps15_lowertrans4lats_remodiffuseclfguide_8l16h/latest.pth --retrieval_method llm_guidance_dectill25_mm9 --use_retrieval --use_inversion --use_insertion_guidance --guidance_iters decreasing_till_25 --seed 5367
# PYTHONPATH=".":$PYTHONPATH python tools/visualize.py experiments/base_beatx_len150fps15_lowertrans4lats_remodiffuseclfguide_8l16h/basegesture_len150_beat.py experiments/base_beatx_len150fps15_lowertrans4lats_remodiffuseclfguide_8l16h/latest.pth --retrieval_method llm_guidance_dectill25_mm10 --use_retrieval --use_inversion --use_insertion_guidance --guidance_iters decreasing_till_25 --seed 4375


PYTHONPATH=".":$PYTHONPATH python tools/longform_synthesis.py experiments/base_beatx_len150fps15_lowertrans4lats_remodiffuseclfguide_8l16h/basegesture_len150_beat.py experiments/base_beatx_len150fps15_lowertrans4lats_remodiffuseclfguide_8l16h/latest.pth --retrieval_method norag_genea2 --test_batchsize 1

wait