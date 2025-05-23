from cv2 import norm
import torch
import torch.nn.functional as F
from torch import layer_norm, nn
import numpy as np
# import clip
import random
import math
from tqdm import tqdm
import time
import copy
import json
import os
import lmdb
import pyarrow
import warnings
import shutil

from torch.nn.utils.rnn import pad_sequence

# ignore Future warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

from ..builder import SUBMODULES, build_attention
from .diffusion_transformer import DiffusionTransformer
from .rag.utils import TextFeatureExtractor, map_conns_to_prominence
from .rag.discourse_retrieval import discourse_retrieval
from .rag.llm_retrieval import llm_retrieval
from .rag.gesture_type_retrieval import gesture_type_retrieval
from ..utils.detr_utils import PositionEmbeddingLearned1D, PositionEmbeddingSine1D
from ..utils import rotation_conversions as rc


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad for all the networks.

    Args:
        nets (nn.Module | list[nn.Module]): A list of networks or a single
            network.
        requires_grad (bool): Whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


class FFN(nn.Module):

    def __init__(self, latent_dim, ffn_dim, dropout):
        super().__init__()
        self.linear1 = nn.Linear(latent_dim, ffn_dim)
        self.linear2 = zero_module(nn.Linear(ffn_dim, latent_dim))
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, **kwargs):
        y = self.linear2(self.dropout(self.activation(self.linear1(x))))
        y = x + y
        return y


class EncoderLayer(nn.Module):

    def __init__(self, sa_block_cfg=None, ca_block_cfg=None, ffn_cfg=None):
        super().__init__()
        self.sa_block = build_attention(sa_block_cfg)
        self.ffn = FFN(**ffn_cfg)

    def forward(self, **kwargs):
        if self.sa_block is not None:
            x = self.sa_block(**kwargs)
            kwargs.update({"x": x})
        if self.ffn is not None:
            x = self.ffn(**kwargs)
        return x
    

class LMDBDict:
    """
    A class to store the a dictionary as lmdb database file. As a new key is added
    the database is updated with the new key and value.
    for access, the key is used to retrieve the value from the lmdb database.
    """
    
    def __init__(self, db_path, torch_converter=False):
        self.db_path = db_path
        self.db = lmdb.open(
            db_path,
            map_size=int(1024 ** 3 * 300),
            readonly=False,
            lock=False,
        )
        self.torch_converter = torch_converter
    
    def __len__(self):
        with self.db.begin(write=False) as txn:
            return txn.stat()["entries"]

    def __setitem__(self, key, value):
        with self.db.begin(write=True) as txn:
            if self.torch_converter:
                if isinstance(value, torch.Tensor):
                    value = value.numpy()
                
                if isinstance(value, (list, tuple)):
                    value = [v.numpy() if isinstance(v, torch.Tensor) else v for v in value]

            v = pyarrow.serialize(value).to_buffer()
            txn.put(key.encode("ascii"), v)

    def __getitem__(self, key):
        with self.db.begin(write=False) as txn:
            value = txn.get(key.encode("ascii"))
            value = pyarrow.deserialize(value)

        if self.torch_converter:
            if isinstance(value, np.ndarray):
                value = torch.from_numpy(value)
                
            elif isinstance(value, list):
                value = [torch.from_numpy(v) if isinstance(v, np.ndarray) else v for v in value]
            
        return value
        

    def __del__(self):
        self.db.sync()
        self.db.close()

    def keys(self):
        with self.db.begin(write=False) as txn:
            return [key.decode("ascii") for key, _ in txn.cursor()]
    
    def values(self):
        raise NotImplementedError
    
    def items(self):
        for key in self.keys():
            yield key, self[key]

    def to_dict(self):
        return {k: v for k, v in self.items()}


class RetrievalDatabase(nn.Module):

    def __init__(
        self,
        dataset,
        motion_feat_dim=189,
        num_retrieval=None,
        topk=None,
        latent_dim=512,
        text_latent_dim=768,
        output_dim=512,
        num_layers=2,
        num_motion_layers=4,
        kinematic_coef=0.1,
        max_seq_len=150,
        motion_fps=15,
        motion_framechunksize=15,
        num_heads=8,
        ff_size=1024,
        stride=4,
        sa_block_cfg=None,
        ffn_cfg=None,
        dropout=0,
        lmdb_paths=None,
        new_lmdb_cache=False,
        stratified_db_creation=False,
        stratification_interval=15,
        # retrieval_dict_path=None,
    ):
        super().__init__()
        self.num_retrieval = num_retrieval
        self.topk = topk
        self.latent_dim = latent_dim
        self.stride = stride
        self.kinematic_coef = kinematic_coef
        self.num_layers = num_layers
        self.num_motion_layers = num_motion_layers
        self.max_seq_len = max_seq_len

        self.retrieval_method = {
            "discourse": discourse_retrieval,
            "gesture_type": gesture_type_retrieval,
            "llm": llm_retrieval,
        }
        
        self.train_indexes = {}
        self.test_indexes = {}
        self.train_dbounds = {}
        self.test_dbounds = {}
        self.train_qbounds = {}
        self.test_qbounds = {}

        # breakpoint()

        self.dataset = dataset

        if new_lmdb_cache and os.path.exists(lmdb_paths):
            breakpoint()
            shutil.rmtree(lmdb_paths)
            os.makedirs(lmdb_paths)
        elif not os.path.exists(lmdb_paths):
            os.makedirs(lmdb_paths)
        
        

        
        self.idx_2_text = LMDBDict(os.path.join(lmdb_paths, "idx_2_text"), torch_converter=True)
        self.idx_2_sense = LMDBDict(os.path.join(lmdb_paths, "idx_2_sense"))
        self.idx_2_discbounds = LMDBDict(os.path.join(lmdb_paths, "idx_2_discbounds"))
        self.idx_2_gesture_labels = LMDBDict(os.path.join(lmdb_paths, "idx_2_gesture_labels"))
        self.idx_2_prominence = LMDBDict(os.path.join(lmdb_paths, "idx_2_prominence"))
        self.idx_2_gestprom = LMDBDict(os.path.join(lmdb_paths, "idx_2_gestprom"))

        # breakpoint()
        if new_lmdb_cache:
            print("Creating retrieval databases")
            for smp_idx, smp in tqdm(enumerate(dataset)):
                # do random selection from self.idx_2_sense, self.idx_2_discbounds, self.idx_2_prominence, self.idx_2_text, idx_2_gesture_labels
                # sample every 30th/15th example from one sample sequence
                # verify that those sampled actually amount upto the length of the sequence
                # breakpoint()
                if stratified_db_creation:
                    per_sample_idx = smp["sample_name"].split("/")[1]
                    if int(per_sample_idx) % stratification_interval != 0:
                        if smp_idx == len(dataset) - 1: break
                        continue

                # breakpoint()  # check the dataset
                speaker_id = int(smp["speaker_id"][0].item())
                # breakpoint()
                if new_lmdb_cache or not os.path.exists(os.path.join(lmdb_paths, "idx_2_text")):
                    self.idx_2_text[smp["sample_name"]] = (smp["text_feature"], speaker_id)

                if new_lmdb_cache or not os.path.exists(os.path.join(lmdb_paths, "idx_2_sense")):
                    self.idx_2_sense[smp["sample_name"]] = [speaker_id] + [(d[1], d[0]) for d in smp["discourse"]]  # speaker_id, sense, text
                
                if new_lmdb_cache or not os.path.exists(os.path.join(lmdb_paths, "idx_2_discbounds")):
                    self.idx_2_discbounds[smp["sample_name"]] = [(d[1], d[0], d[4], d[5], d[6], d[7]) for d in smp["discourse"]]  # sense, text, disco_start, disco_end, conn_start, conn_end

                if new_lmdb_cache or not os.path.exists(os.path.join(lmdb_paths, "idx_2_gesture_labels")):
                    self.idx_2_gesture_labels[smp["sample_name"]] = [speaker_id] + smp["gesture_labels"]
                

                # filter out the relevant prominance values according to
                # the connectives in disco conns
                # breakpoint()  # check following code
                if new_lmdb_cache or not os.path.exists(os.path.join(lmdb_paths, "idx_2_prominence")):
                    smp_conns = []
                    for disco in smp["discourse"]:
                        smp_conns.append(disco[0])
                    relevant_dps = map_conns_to_prominence(smp_conns, smp["prominence"])

                    # if len(relevant_dps) > 1 and relevant_dps[0] is not None:
                    #     if "." in relevant_dps[0][0]:
                    #         breakpoint()

                    self.idx_2_prominence[smp["sample_name"]] = relevant_dps

                if new_lmdb_cache or not os.path.exists(os.path.join(lmdb_paths, "idx_2_gestprom")):
                    smp_gest_words = [s["word"] for s in smp["gesture_labels"]]
                    relevant_gest_dps = map_conns_to_prominence(smp_gest_words, smp["prominence"])
                    self.idx_2_gestprom[smp["sample_name"]] = relevant_gest_dps

                if smp_idx == len(dataset) - 1:
                    
                    break

            print("Retrival databases creation finished")

        # TODO: stratify existing databases 

        # load the databases into memory as dictionaries
        self.idx_2_text = self.idx_2_text.to_dict()
        self.idx_2_sense = self.idx_2_sense.to_dict()
        self.idx_2_discbounds = self.idx_2_discbounds.to_dict()
        self.idx_2_gesture_labels = self.idx_2_gesture_labels.to_dict()
        self.idx_2_prominence = self.idx_2_prominence.to_dict()
        self.idx_2_gestprom = self.idx_2_gestprom.to_dict()
        

        
        self.feature_cache_tensor = pad_sequence([f[0] for f in self.idx_2_text.values()], batch_first=True)
        self.sample_names = {i: s for i, s in enumerate(self.idx_2_text.keys())}



        # breakpoint()
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.text_latent_dim = text_latent_dim

        self.motion_fps = motion_fps
        self.motion_framechunksize = motion_framechunksize

    def generate_src_mask(self, T, length):
        B = len(length)
        src_mask = torch.ones(B, T)
        for i in range(B):
            for j in range(length[i], T):
                src_mask[i, j] = 0
        return src_mask

    def retrieve(
        self,
        retr_method,
        text,
        text_features,
        audio,
        discourse,
        gesture_labels,
        text_times,
        prominence,
        speaker_id,
        # length,
        idx=None,
    ):
        # train cache is in form of two dicts: self.train_indexes and self.train_dbounds
        # test cache is in form of two dicts: self.test_indexes and self.test_dbounds

        # self.train_indexes is a dict of query idx to dict of retrival types to list of smp indexes
        # self.train_dbounds is a dict of query idx to dict of retrival types to dict of smp indexes to bounds
        # breakpoint()
        assert retr_method in ["gesture_type", "discourse", "llm"] # "llm"
        # print(f"Retrieval method: {retr_method}")
        if self.training and idx in self.train_indexes and idx is not None:
            # idx = idx.item()
            # breakpoint()
            multiple_db_indexes = self.train_indexes[idx]
            multiple_db_bounds = self.train_dbounds[idx]
            multiple_query_bounds = self.train_qbounds[idx]

            # select the db_indexes and db_bounds randomly during training
            per_idx_retrmethods = list(multiple_db_indexes.keys())
            if len(per_idx_retrmethods) == 0:
                return {}, {}, {}
            train_retr_method = random.choice(per_idx_retrmethods)

            db_indexes = multiple_db_indexes[train_retr_method]
            db_bounds = multiple_db_bounds[train_retr_method]
            query_bounds = multiple_query_bounds[train_retr_method]

            data = {}
            # bounds = {}
            for query_w_idx, smp_idxs in db_indexes.items():
                data[query_w_idx] = [s_i for s_i in smp_idxs if s_i != idx]
                data[query_w_idx] = data[query_w_idx][: self.topk]
                random.shuffle(data[query_w_idx])
                data[query_w_idx] = data[query_w_idx][: self.num_retrieval]

            return data, db_bounds, query_bounds

        elif not self.training and idx in self.test_indexes and idx is not None:
            
            multiple_db_indexes = self.test_indexes[idx]
            multiple_db_bounds = self.test_indexes[idx]
            multiple_query_bounds = self.test_qbounds[idx]

            if retr_method not in multiple_db_indexes:
                print(
                    f"WARNUNG: Retrieval method {retr_method} not found for idx {idx}"
                )
                return {}, {}, {}

            # select the db_indexes and db_bounds based on the retr_method during testing
            db_indexes = multiple_db_indexes[retr_method]
            db_bounds = multiple_db_bounds[retr_method]
            query_bounds = multiple_query_bounds[retr_method]

            if len(db_indexes) == 0:
                print(
                    f"WARNUNG: No samples found for idx {idx} for retr_method {retr_method}"
                )
                # return {}, {}, {}

            data = {}
            # bounds = {}
            for query_w_idx, smp_idxs in db_indexes.items():
                data[query_w_idx] = [s_i for s_i in smp_idxs if s_i != idx]
                # data[query_w_idx] = data[query_w_idx][: self.topk]
                # random.shuffle(data[query_w_idx])
                data[query_w_idx] = data[query_w_idx][: self.num_retrieval]

            return data, db_bounds, query_bounds
        else:
            # base_method_args =
            # breakpoint() 
            method_args = {}
            for retr_m in self.retrieval_method:

                
                method_args[retr_m] = {
                    "text": text,
                    "speaker_id": speaker_id,
                    "encoded_text": text_features,
                    "text_feat_cache": self.idx_2_text,
                }
                if retr_m == "gesture_type":
                    method_args[retr_m]["gesture_labels"] = gesture_labels
                    method_args[retr_m][
                        "db_idx_2_gesture_labels"
                    ] = self.idx_2_gesture_labels
                elif retr_m == "llm":
                    method_args[retr_m]["text_times"] = text_times
                    method_args[retr_m][
                        "db_idx_2_gesture_labels"
                    ] = self.idx_2_gesture_labels
                    method_args[retr_m]["prominence"] = prominence
                    method_args[retr_m]["db_idx_2_prominence"] = self.idx_2_gestprom

                elif retr_m == "discourse":
                    method_args[retr_m]["discourse"] = discourse
                    method_args[retr_m]["prominence"] = prominence
                    method_args[retr_m]["db_idx_2_sense"] = self.idx_2_sense
                    method_args[retr_m]["db_idx_2_discbounds"] = self.idx_2_discbounds
                    method_args[retr_m]["db_idx_2_prominence"] = self.idx_2_prominence
                elif retr_m == "prosody":
                    method_args[retr_m]["audio"] = audio
                    method_args[retr_m]["prominence"] = prominence
                    method_args[retr_m]["db_idx_2_prominence"] = self.idx_2_prominence
                    raise NotImplementedError
                
                else:
                    raise NotImplementedError

            self.train_indexes[idx] = {}
            self.train_dbounds[idx] = {}
            self.train_qbounds[idx] = {}
            self.test_indexes[idx] = {}
            self.test_dbounds[idx] = {}
            self.test_qbounds[idx] = {}

            if not self.training:
                # use retrieval method to get the sample indexes and bounds
                # instead of randomly selecting one
                sample_indexes, sample_bounds, query_bounds = self.retrieval_method[
                    retr_method
                ](**method_args[retr_method])

                self.test_indexes[idx].update({retr_method: sample_indexes})
                self.test_dbounds[idx].update({retr_method: sample_bounds})
                self.test_qbounds[idx].update({retr_method: query_bounds})

            else:
                raise NotImplementedError("Not released for training for retrieval")

                per_idx_retrmethods = list(self.train_indexes[idx].keys())
                if len(per_idx_retrmethods) == 0:
                    return {}, {}, {}
                
                
                train_retr_method = random.choice(per_idx_retrmethods)
                # print(f"-----Selected retrieval method: {train_retr_method}")

                sample_indexes = self.train_indexes[idx][train_retr_method]
                sample_bounds = self.train_dbounds[idx][train_retr_method]
                query_bounds = self.train_qbounds[idx][train_retr_method]

                # breakpoint()

            data = {}
            # bounds = {}
            for query_w_idx, smp_idxs in sample_indexes.items():
                data[query_w_idx] = [s_i for s_i in smp_idxs if s_i != idx]

                data[query_w_idx] = data[query_w_idx][: self.num_retrieval]

            return data, sample_bounds, query_bounds

    def forward(
        self, conditions, lengths, device, idx=None, retrieval_method="gesture_type", gesture_rep_encoder=None
    ):
        B = len(conditions["text"])
        all_indexes = []
        all_masked_motions = []
        raw_masked_motions = []
        raw_masked_motions_aa = []
        raw_masked_trans = []
        raw_masked_facial = []
        all_words = []
        all_raw_words = []
        all_type2words = []

        all_retr_startends = []
        all_query_startends = []
        all_retr_latents = []
        start = time.time()
        # timess = []
        for b_ix in range(B):
            retr_indexes, retr_bounds, query_bounds = self.retrieve(
                retrieval_method,
                text=conditions["text"][b_ix],
                text_features=conditions["text_features"][b_ix],
                audio=conditions["audio"][b_ix],
                discourse=conditions["discourse"][b_ix],
                gesture_labels=conditions["gesture_labels"][b_ix],
                text_times=conditions["text_times"][b_ix],
                prominence=conditions["prominence"][b_ix],
                speaker_id=conditions["speaker_ids"][b_ix, 0].item(),
                # lengths[b_ix],
                idx=idx[b_ix] if idx is not None else None,
            )
            all_indexes.append(retr_indexes)
            
            

            batch_masked_motions = []
            batch_words = []
            batch_type2words = {}

            
            
            zero_motion = torch.zeros((self.max_seq_len // self.motion_framechunksize * 4 + 3, self.latent_dim)).to(device)

            zero_text = torch.zeros((self.max_seq_len // self.motion_framechunksize * 4 + 3, self.text_latent_dim)).to(device)

            zero_raw_motion = torch.zeros_like(self.dataset[0]["motion"]).to(device)
            zero_raw_text = torch.zeros_like(self.dataset[0]["word"]).to(device)
            zero_raw_motion_aa = torch.zeros_like(self.dataset[0]["motion"]).to(device)
            
            zero_raw_trans = torch.zeros_like(self.dataset[0]["trans"]).to(device)
            zero_raw_facial = torch.zeros_like(self.dataset[0]["facial"]).to(device)

            text_encoded = conditions["text_enc"][b_ix]
            # breakpoint() 
            prev_end_frame = -1

            retr_startend = {}
            query_startend = {}
            retrlatents_uncropped = {}

            # breakpoint()
            for query_point_idx, smp_idxs in retr_indexes.items():
                if len(smp_idxs) == 0:
                    continue

                if query_point_idx not in query_bounds:
                    continue

                query_bound = query_bounds[query_point_idx]
                query_word, query_type, query_start, query_end = query_bound

                if query_start > query_end:
                    continue

                assert len(smp_idxs) == self.num_retrieval == 1
                for smp_idx in smp_idxs:
                    
                    retr_motion = self.dataset[smp_idx]["motion"].unsqueeze(0).to(device)
                    retr_motion_upper = (self.dataset[smp_idx]["motion_upper"]).unsqueeze(0).to(device)
                    retr_motion_lower = (self.dataset[smp_idx]["motion_lower"]).unsqueeze(0).to(device)
                    retr_motion_face = (self.dataset[smp_idx]["motion_face"]).unsqueeze(0).to(device)
                    retr_motion_facial = (self.dataset[smp_idx]["facial"]).unsqueeze(0).to(device)
                    retr_motion_hands = (self.dataset[smp_idx]["motion_hands"]).unsqueeze(0).to(device)
                    retr_motion_transl = (self.dataset[smp_idx]["trans"]).unsqueeze(0).to(device)
                    retr_motion_contact = (self.dataset[smp_idx]["contact"]).unsqueeze(0).to(device)
                    retr_motion_mask = (self.dataset[smp_idx]["motion_mask"]).unsqueeze(0).to(device)

                    retr_text = self.dataset[smp_idx]["word"].unsqueeze(0).to(device)
                    retr_audio = self.dataset[smp_idx]["audio"].unsqueeze(0).to(device)
                    retr_spkid = self.dataset[smp_idx]["speaker_id"].unsqueeze(0).to(device)
                    retr_motion_aa = retr_motion.clone()
                    

                    
                    if retr_motion.shape[0] == 0:
                        continue

                    assert gesture_rep_encoder is not None
                    
                    retr_motion_latent, re_lat_motion_mask = gesture_rep_encoder.encode(
                        retr_motion_upper, retr_motion_lower, retr_motion_face, retr_motion_hands, retr_motion_transl, retr_motion_facial, retr_motion_contact, retr_motion_mask
                    )
                    

                    retr_motion_latent = retr_motion_latent.squeeze(0)
                    retr_motion = retr_motion.squeeze(0) # .detach()
                    retr_motion_aa = retr_motion_aa.squeeze(0) #.detach()
                    retr_motion_transl = retr_motion_transl.squeeze(0) #.detach()
                    retr_motion_facial = retr_motion_facial.squeeze(0) #.detach()

                    motion_len = self.max_seq_len #zero_motion.shape[0] # change this according to zero_motion # check axis of zero_motion

                    

                    retr_word, retr_type, retr_start, retr_end = retr_bounds[query_point_idx][smp_idx]
                    
                    # logging for testing
                    batch_type2words[query_point_idx] = (
                        query_word,
                        query_type,
                        retr_word,
                        retr_type,
                    )

                    # motion features Rm
                    query_start = max(0, query_start)
                    query_end = min(motion_len / self.motion_fps, query_end)

                    query_start = int(query_start * self.motion_fps)
                    query_end = int(query_end * self.motion_fps)

                    # if query_start == query_end:
                    #     continue
                    

                    query_lat_start = query_start // self.motion_framechunksize
                    query_lat_end = query_end // self.motion_framechunksize + 1
                    if query_lat_start >= query_lat_end:
                        breakpoint() # check wth happened here
                    
                    # 0.6 before and 0.3 sec after the stroke. 
                    # currently assuming stroke in the middle. 
                    # time reference frame
                    # breakpoint()
                    if (retrieval_method == "gesture_type" or retrieval_method == "llm") \
                        and (retr_end - retr_start) > 0.9:
                        # reduced padding for gesture retrieval 
                        # because of large annotation duration in BEAT dataset
                        retr_start = max(0, retr_start - 0.2)
                        retr_end = min(motion_len / self.motion_fps, retr_end + 0.1)
                    else:
                        retr_start = max(0, retr_start - 0.666)  # half second padding 
                        retr_end = min(motion_len / self.motion_fps, retr_end + 0.333)  # half second padding
                    # padding also affects how much of an overlap you have with
                    # other retr motions

                    # frame reference frame
                    retr_start = int(retr_start * self.motion_fps)
                    retr_end = int(retr_end * self.motion_fps)

                    # breakpoint()
                    if retr_start == retr_end:
                        continue
                    
                    if retr_end == motion_len:
                        retr_end = motion_len - 1
                        retr_start = max(0, retr_start - 1)

                    retr_lat_start = retr_start // self.motion_framechunksize
                    retr_lat_end = retr_end // self.motion_framechunksize + 1
                    if retr_lat_start >= retr_lat_end:
                        breakpoint() # check wtf happened here

                    query_mid = (query_start + query_end) // 2
                    query_mid_lat = query_mid // self.motion_framechunksize

                    latent_len = (zero_motion.shape[0] - 3) // 4
                    assert latent_len == motion_len // self.motion_framechunksize

                    # breakpoint()

                    
                    retr_window_lat_u = retr_motion_latent[retr_lat_start:retr_lat_end]
                    retr_window_lat_h = retr_motion_latent[latent_len + 1 + retr_lat_start: latent_len + 1 + retr_lat_end]
                    retr_window_lat_f = retr_motion_latent[2 * latent_len + 2 + retr_lat_start: 2 * latent_len + 2 + retr_lat_end]
                    retr_window_lat_lt = retr_motion_latent[3 * latent_len + 3 + retr_lat_start: 3 * latent_len + 3 + retr_lat_end]

                    retr_motion_raw = retr_motion[retr_lat_start*self.motion_framechunksize:retr_lat_end*self.motion_framechunksize]
                    retr_motion_raw_aa = retr_motion_aa[retr_lat_start*self.motion_framechunksize:retr_lat_end*self.motion_framechunksize]
                    retr_trans_raw = retr_motion_transl[retr_lat_start*self.motion_framechunksize:retr_lat_end*self.motion_framechunksize]
                    retr_facial_raw = retr_motion_facial[retr_lat_start*self.motion_framechunksize:retr_lat_end*self.motion_framechunksize]
                    retr_length_raw = retr_motion_raw.shape[0]

                    # breakpoint()
                    retr_length_lat = retr_window_lat_u.shape[0]
                    assert retr_length_lat > 0
                    if retr_length_lat == 1:
                        side_length = retr_length_lat // 2
                        start_lat = query_mid_lat - side_length
                        end_lat = query_mid_lat + side_length + 1
                    elif retr_length_lat == 2:
                        side_length = retr_length_lat // 2
                        start_lat = query_mid_lat
                        end_lat = query_mid_lat + side_length + 1
                    elif retr_length_lat % 2 == 1:
                        side_length = retr_length_lat // 2
                        start_lat = query_mid_lat - side_length - 1
                        end_lat = query_mid_lat + side_length
                    else:
                        side_length = retr_length_lat // 2
                        start_lat = query_mid_lat - side_length
                        end_lat = query_mid_lat + side_length

                    if start_lat < 0:
                        start_lat = 0
                        end_lat = retr_length_lat
                    
                    if end_lat > latent_len:
                        start_lat -= end_lat - latent_len
                        end_lat = latent_len

                    if start_lat < prev_end_frame:
                        start_lat = prev_end_frame
                        end_lat = start_lat + retr_length_lat
                        if end_lat > latent_len:
                            end_lat = latent_len
                            retr_length_lat = end_lat - start_lat
                            # breakpoint() # shouldnt it be retr_window_lat[start_lat:end_lat]?
                            if retr_length_lat <= 0:
                                continue
                            retr_window_lat_u = retr_window_lat_u[:retr_length_lat]
                            # retr_window_lat_l = retr_window_lat_l[:retr_length_lat]
                            
                            retr_window_lat_h = retr_window_lat_h[:retr_length_lat]
                            retr_window_lat_f = retr_window_lat_f[:retr_length_lat]
                            retr_window_lat_lt = retr_window_lat_lt[:retr_length_lat]
                            # retr_window_lat_t = retr_window_lat_t[:retr_length_lat]


                            retr_motion_raw = retr_motion_raw[:retr_length_lat*self.motion_framechunksize]
                            retr_motion_raw_aa = retr_motion_raw_aa[:retr_length_lat*self.motion_framechunksize]
                            retr_trans_raw = retr_trans_raw[:retr_length_lat*self.motion_framechunksize]
                            retr_facial_raw = retr_facial_raw[:retr_length_lat*self.motion_framechunksize]

                            # update retr_lat_end
                            retr_lat_end = retr_lat_start + retr_length_lat


                            # breakpoint()
                            assert retr_window_lat_u.shape[0] == retr_length_lat

                    prev_end_frame = end_lat

                    
                    # append retr_latents to the list of retr_latents
                    retrlatents_uncropped[query_point_idx] = {
                        "retr_motion_latent": retr_motion_latent.unsqueeze(0), # 1, T, D
                        "retr_text": retr_text,
                        "retr_audio": retr_audio,
                        "retr_spkid": retr_spkid,
                        "retr_motion_mask": re_lat_motion_mask,
                    }
                    # append the retr_start and retr_end to the list of retr_startend list
                    retr_startend[query_point_idx] = (retr_lat_start, retr_lat_end)
                    # append the query_start and query_end to the list of query_startend list
                    query_startend[query_point_idx] = (start_lat, end_lat)

                    
                    zero_motion[start_lat:end_lat] = retr_window_lat_u
                    zero_motion[latent_len + 1 + start_lat: latent_len + 1 + end_lat] = retr_window_lat_h
                    zero_motion[2 * latent_len + 2 + start_lat: 2 * latent_len + 2 + end_lat] = retr_window_lat_f
                    zero_motion[3 * latent_len + 3 + start_lat: 3 * latent_len + 3 + end_lat] = retr_window_lat_lt
                    

                    zero_raw_motion[start_lat*self.motion_framechunksize:end_lat*self.motion_framechunksize] = retr_motion_raw
                    zero_raw_motion_aa[start_lat*self.motion_framechunksize:end_lat*self.motion_framechunksize] = retr_motion_raw_aa
                    zero_raw_trans[start_lat*self.motion_framechunksize:end_lat*self.motion_framechunksize] = retr_trans_raw
                    zero_raw_facial[start_lat*self.motion_framechunksize:end_lat*self.motion_framechunksize] = retr_facial_raw
                    
                    
                    if query_start >= query_end:
                        q_s = query_start - 1
                        q_e = query_end + 1
                        q_s = max(0, q_s)
                        q_e = min(text_encoded.shape[0], q_e)
                        # breakpoint()
                    else:
                        q_s = query_start
                        q_e = query_end

                    text_enc_pooled = text_encoded[q_s:q_e]

                    # select end_lat - start_lat equally spaced frames from text_enc_pooled
                    remainder = text_enc_pooled.shape[0] % (end_lat - start_lat)
                    if remainder > 0 and text_enc_pooled.shape[0] > (end_lat - start_lat):
                        text_enc_pooled = text_enc_pooled[:-remainder]

                    if text_enc_pooled.shape[0] // (end_lat - start_lat) == 0:
                        # if text_enc_pooled.shape[0] == 0:
                        #     breakpoint()

                        text_enc_pooled = text_enc_pooled[:1].expand(end_lat - start_lat, -1)
                    else:
                        text_enc_pooled = text_enc_pooled[::text_enc_pooled.shape[0] // (end_lat - start_lat)]


                    zero_text[start_lat:end_lat] = text_enc_pooled
                    zero_text[latent_len + 1 + start_lat: latent_len + 1 + end_lat] = text_enc_pooled
                    zero_text[2 * latent_len + 2 + start_lat: 2 * latent_len + 2 + end_lat] = text_enc_pooled
                    zero_text[3 * latent_len + 3 + start_lat: 3 * latent_len + 3 + end_lat] = text_enc_pooled
                    zero_raw_text[start_lat*self.motion_framechunksize:end_lat*self.motion_framechunksize] = text_enc_pooled.repeat(self.motion_framechunksize, 1)
                    
            
            all_type2words.append(batch_type2words)
            all_masked_motions.append(zero_motion)
            all_words.append(zero_text)
            all_raw_words.append(zero_raw_text)
            raw_masked_motions.append(zero_raw_motion)
            raw_masked_motions_aa.append(zero_raw_motion_aa)
            raw_masked_trans.append(zero_raw_trans)
            raw_masked_facial.append(zero_raw_facial)

            all_retr_startends.append(retr_startend)
            all_query_startends.append(query_startend)
            all_retr_latents.append(retrlatents_uncropped)

        N = len(all_masked_motions)
        all_motions = torch.stack(all_masked_motions, dim=0).to(device)
        
        

        all_raw_motions = torch.stack(raw_masked_motions_aa, dim=0).to(device)
        all_raw_trans = torch.stack(raw_masked_trans, dim=0).to(device)
        all_raw_facial = torch.stack(raw_masked_facial, dim=0).to(device)

        # getting the sample names for the retrieved motions for future reference
        all_sample_names = []
        # all_type2words_ = []
        for b_ix in range(B):
            type2words = all_type2words[b_ix]
            all_sample_names.append({})
            for query_point_idx, smp_idxs in all_indexes[b_ix].items():
                if query_point_idx not in type2words:
                    continue
                q_word, q_type, r_word, r_type = type2words[query_point_idx]

                assert len(smp_idxs) == self.num_retrieval == 1
                all_sample_names[-1][q_word] = self.dataset[smp_idxs[0]]["sample_name"]

        # text feature processing:
        # breakpoint() # check all words shape
        all_text_features = torch.stack(all_words, dim=0).to(device)
        

        T = all_text_features.shape[1]
        # breakpoint()  # check all_text_features shape and self.text_pos_embedding shape
        # T should be equal to self.text_pos_embedding.shape[0]
        

        # motion feature processing:
        T = all_motions.shape[1] 
        src_mask = (all_motions != 0).any(dim=-1).to(torch.int).to(device)
        raw_latent_mask = src_mask.clone() # TODO: check how to change this

        all_motions_reshaped = all_motions
        
        raw_motion_latents = all_motions_reshaped.clone()

        upper_indices = list(range(0, (T-3)//4))
        hands_indices = list(range((T-3)//4 + 1, 2*(T-3)//4 + 1))
        face_indices = list(range(2*(T-3)//4 + 2, 3*(T-3)//4 + 2))
        lowertrans_indices = list(range(3*(T-3)//4 + 3, T))

        
        src_mask[:, face_indices + lowertrans_indices] = 0
        raw_motion_latents[:, face_indices + lowertrans_indices, :] = 0
        

        # breakpoint()  # check the raw motion and raw text features
        raw_motion = all_raw_motions.view(B, self.num_retrieval, self.max_seq_len, -1).contiguous()
        raw_trans = all_raw_trans.view(B, self.num_retrieval, self.max_seq_len, -1).contiguous()
        raw_facial = all_raw_facial.view(B, self.num_retrieval, self.max_seq_len, 100).contiguous()
        raw_motion_latents = raw_motion_latents.view(B, self.num_retrieval, T, -1).contiguous()
        # breakpoint()  # check the raw motion and raw text features

        re_dict = dict(
            re_text=None,
            re_motion=None,
            
            re_mask=src_mask,
            raw_motion_latents=raw_motion_latents,
            raw_motion=raw_motion,
            raw_trans=raw_trans,
            raw_facial=raw_facial,
            raw_sample_names=all_sample_names,
            raw_type2words=all_type2words,
            raw_latent_mask=raw_latent_mask, # without upper selection

            retr_startends=all_retr_startends,
            query_startends=all_query_startends,
            retr_uncropped_latents=all_retr_latents,
        )
        return re_dict


@SUBMODULES.register_module()
class ReGestureTransformer(DiffusionTransformer):
    def __init__(self, 
                 retrieval_cfg=None, 
                 scale_func_cfg=None, 
                 per_joint_scale=None, 
                 retrieval_train=False, 
                 use_retrieval_for_test=False, 
                 **kwargs
                 ):
        
        dataset = kwargs.pop("database")
        super().__init__(**kwargs)
        assert not retrieval_train
        if retrieval_cfg is not None and use_retrieval_for_test:
            self.database = RetrievalDatabase(
                **retrieval_cfg,
                dataset=dataset,
            )
        else:
            self.database = None
        self.scale_func_cfg = scale_func_cfg

        self.per_joint_scale = per_joint_scale
        if self.per_joint_scale is not None:
            T = 43
            upper_indices = list(range(0, (T-3)//4))
            hands_indices = list(range((T-3)//4 + 1, 2*(T-3)//4 + 1))
            face_indices = list(range(2*(T-3)//4 + 2, 3*(T-3)//4 + 2))
            lowertransl_indices = list(range(3*(T-3)//4 + 3, T))

            self.joint_scale_mask = torch.ones(T)
            self.joint_scale_mask[upper_indices] = self.per_joint_scale["upper"]
            self.joint_scale_mask[hands_indices] = self.per_joint_scale["hands"]
            self.joint_scale_mask[face_indices] = self.per_joint_scale["face"]
            self.joint_scale_mask[lowertransl_indices] = self.per_joint_scale["lowertransl"]


    def scale_func_retr(self, timestep):
        coarse_scale = self.scale_func_cfg["coarse_scale"]
        w = (1 - (1000 - timestep) / 1000) * coarse_scale + 1
        if timestep > 100:
            if random.randint(0, 1) == 0:
                output = {
                    "both_coef": w,
                    "text_coef": 0,
                    "retr_coef": 1 - w,
                    "none_coef": 0,
                }
            else:
                output = {
                    "both_coef": 0,
                    "text_coef": w,
                    "retr_coef": 0,
                    "none_coef": 1 - w,
                }
        else:
            both_coef = self.scale_func_cfg["both_coef"]
            text_coef = self.scale_func_cfg["text_coef"]
            retr_coef = self.scale_func_cfg["retr_coef"]
            none_coef = 1 - both_coef - text_coef - retr_coef
            output = {
                "both_coef": both_coef,
                "text_coef": text_coef,
                "retr_coef": retr_coef,
                "none_coef": none_coef,
            }
        return output


    def get_precompute_condition(
        self,
        text=None,
        raw_text=None,
        text_features=None,
        audio=None,
        raw_audio=None,
        discourse=None,
        prominence=None,
        speaker_ids=None,
        gesture_labels=None,
        text_times=None,
        motion_length=None,
        xf_out=None,
        re_dict=None,
        device=None,
        sample_idx=None,
        sample_name=None,
        retrieval_method="gesture_type",
        **kwargs,
    ):
        if xf_out is None:
            xf_out_text = self.encode_text(text, device)
            xf_out_audio = self.encode_audio(audio, device)
            xf_spk = self.encode_spks(speaker_ids, device)
            
            xf_out = {
                "xf_text": xf_out_text,
                "xf_audio": xf_out_audio,
                "xf_spk": xf_spk,
            }
        output = {"xf_out": xf_out}
        
        if re_dict is None and self.database is not None:
            retr_conditions = dict(
                text=raw_text, 
                audio=raw_audio, 
                text_enc=text,
                text_features=text_features,
                audio_enc=audio,
                discourse=discourse,
                prominence=prominence,
                speaker_ids=speaker_ids,
                gesture_labels=gesture_labels,
                text_times=text_times,
            )
            re_dict = self.database(
                retr_conditions,
                motion_length,
                device,
                idx=sample_name,  # sample_idx
                retrieval_method=retrieval_method,
                gesture_rep_encoder=self.gesture_rep_encoder,
            )
        output["re_dict"] = re_dict
        
        return output

    def post_process(self, motion):
        return motion

    def forward_train(
        self, h=None, src_mask=None, emb=None, xf_out=None, query_mask=None, re_dict=None, **kwargs
    ):
        start = time.time()
        B, T = h.shape[0], h.shape[1]
        
        cond_type = torch.randint(0, 100, size=(B, 1, 1)).to(h.device)
        for module in self.temporal_decoder_blocks:
            h = module(
                x=h,
                xf=xf_out,
                emb=emb,
                src_mask=src_mask,
                query_mask=query_mask,
                cond_type=cond_type,
                re_dict=re_dict,
            )

        
        output = self.out(h).view(B, T, -1).contiguous()
        
        return output, re_dict
    
    def forward_test(
        self,
        h=None,
        src_mask=None,
        emb=None,
        xf_out=None,
        query_mask=None,
        timesteps=None,
        do_clf_guidance=False,
        **kwargs,
    ):
        
        B, T = h.shape[0], h.shape[1]
        text_cond_type = torch.zeros(B, 1, 1).to(h.device) + 1
        if do_clf_guidance or self.scale_func_cfg is not None:
            none_cond_type = torch.zeros(B, 1, 1).to(h.device)

            cond_types = (text_cond_type, none_cond_type)
            all_cond_type = torch.cat(cond_types, dim=0)

            h = h.repeat(len(cond_types), 1, 1)
            xf_out = {
                k: v.repeat(len(cond_types), 1, 1) for k, v in xf_out.items()
            }
            emb = emb.repeat(len(cond_types), 1)
            src_mask = src_mask.repeat(len(cond_types), 1, 1)

            if query_mask is not None:
                for k, v in query_mask.items():
                    query_mask[k] = v.repeat(len(cond_types), 1)

        else:
            all_cond_type = text_cond_type
        
        
        for module in self.temporal_decoder_blocks:
            h = module(
                x=h,
                xf=xf_out,
                emb=emb,
                src_mask=src_mask,
                query_mask=query_mask,
                cond_type=all_cond_type,
            )
        
        out = self.out(h)
        if do_clf_guidance or self.scale_func_cfg is not None:
            out = out.view(2 * B, T, -1).contiguous()
            
            if self.scale_func_cfg is not None:

                coef_cfg = self.scale_func_retr(int(timesteps[0]))
                both_coef = coef_cfg["both_coef"]
                text_coef = coef_cfg["text_coef"]
                retr_coef = coef_cfg["retr_coef"]
                none_coef = coef_cfg["none_coef"]

                out_text = out[:B].contiguous()
                out_none = out[B : 2 * B].contiguous()
                
                
                joint_scale_tensor = self.joint_scale_mask.unsqueeze(0).unsqueeze(-1).expand(B, -1, out_text.shape[-1])
                joint_scale_tensor = joint_scale_tensor.to(out_text.device)

                output = (
                    out_text * both_coef * joint_scale_tensor
                    + out_text * text_coef * joint_scale_tensor
                    + out_none * retr_coef * (1/joint_scale_tensor)
                    + out_none * none_coef * (1/joint_scale_tensor)
                )
                out = output

        return out

