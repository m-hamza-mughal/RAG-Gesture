from abc import ABCMeta, abstractmethod
from cv2 import norm
import torch
from torch import layer_norm, nn
import torch.nn.functional as F
from mmcv.runner import BaseModule
import numpy as np
import fairseq
import transformers  # for bert pipeline

from ..builder import SUBMODULES, build_attention
from ..utils.stylization_block import StylizationBlock
from .gesture_vae import TransformerVAE
from ..utils.detr_utils import PositionEmbeddingLearned1D, PositionEmbeddingSine1D
from ..utils import rotation_conversions as rc
import math
import os
import time
import copy
import yaml
from argparse import Namespace
from collections import OrderedDict

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32)
        / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


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


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


class FFN(nn.Module):

    def __init__(self, latent_dim, ffn_dim, dropout, time_embed_dim):
        super().__init__()
        self.linear1 = nn.Linear(latent_dim, ffn_dim)
        self.linear2 = zero_module(nn.Linear(ffn_dim, latent_dim))
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.proj_out = StylizationBlock(latent_dim, time_embed_dim, dropout)

    def forward(self, x, emb, **kwargs):
        y = self.linear2(self.dropout(self.activation(self.linear1(x))))
        y = x + self.proj_out(y, emb)
        return y


class DecoderLayer(nn.Module):

    def __init__(self, sa_block_cfg=None, ca_block_cfg=None, ffn_cfg=None):
        super().__init__()
        self.sa_block = build_attention(sa_block_cfg)
        self.ca_blocks = nn.ModuleDict()
        for ca_block in ["xf_text", "xf_audio", "xf_spk"]:
            self.ca_blocks[ca_block] = build_attention(ca_block_cfg)

        # TODO: Move this 3 to a config
        self.ca_mix = nn.Linear(
            ffn_cfg["latent_dim"] * 3, ffn_cfg["latent_dim"]
        )  # 3 for Text, audio, speakerid
        self.ffn = FFN(**ffn_cfg)

    def forward(self, **kwargs):
        if self.sa_block is not None:
            x = self.sa_block(**kwargs)
            kwargs.update({"x": x})
        if self.ca_blocks is not None:
            # breakpoint()
            condition_xf = kwargs.pop("xf")
            x_out_conds = {}
            # breakpoint()
            query_masks = kwargs.pop("query_mask")
            for cond, xf_cond in condition_xf.items():
                q_mask = query_masks[cond] if query_masks is not None else None
                x_out_cond = self.ca_blocks[cond](xf=xf_cond, query_mask=q_mask, **kwargs)
                x_out_conds[cond] = x_out_cond

            x = self.ca_mix(
                torch.cat([x_out_conds[cond] for cond in x_out_conds.keys()], dim=-1)
            )

            kwargs.update({"x": x})
        if self.ffn is not None:
            x = self.ffn(**kwargs)
        return x


class GestureRepEncoder(nn.Module):
    def __init__(self, vae_cfg, body_part_cat_axis="time"):
        super(GestureRepEncoder, self).__init__()
        self.vae_cfg = vae_cfg
        self.body_part_cat_axis = body_part_cat_axis
        self.frame_chunk_size = vae_cfg["frame_chunk_size"]
        
        upper_cfg = vae_cfg["upper_cfg"]
        self.upper_vae, self.upper_latproj = self.load_vae(upper_cfg)

        face_cfg = vae_cfg["face_cfg"]
        self.face_vae, self.face_latproj = self.load_vae(face_cfg)

        hands_cfg = vae_cfg["hands_cfg"]
        self.hands_vae, self.hands_latproj = self.load_vae(hands_cfg)

        lowertrans_cfg = vae_cfg["lowertrans_cfg"]
        self.lowertrans_vae, self.lowertrans_latproj = self.load_vae(lowertrans_cfg)

        self.vae_latent_dim = vae_cfg["latent_dim"]
    
    def load_vae(self, cfg_path):
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)
        args = Namespace(**cfg)
        ckpt_path = os.path.dirname(cfg_path) + "/" + os.path.basename(args.test_ckpt)
        model = TransformerVAE(args)
        self.load_checkpoints(model, ckpt_path)
        model.eval()
        model.training = False
        for p in model.parameters():
            p.requires_grad = False
        
        if self.vae_cfg['latent_dim'] != args.latent_dim:
            proj = nn.Identity() 
        else:
            proj = nn.Identity()
        return model, proj
    
    @staticmethod
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
                model.load_state_dict(states['model_state'])
        else:
            model.load_state_dict(states['model_state'])
        print(f"load self-pretrained checkpoints for {load_name}")

    def encode(self, motion_upper, motion_lower, motion_face, motion_hands, motion_transl, motion_facial, motion_contact, motion_mask):
        # check vae device and training/eval condition
        # upper latent
        bs, n, uj = motion_upper.shape
        uj = uj // 3
        motion_upper = rc.axis_angle_to_matrix(motion_upper.reshape(bs, n, uj, 3))
        motion_upper = rc.matrix_to_rotation_6d(motion_upper).reshape(bs, n, uj*6)
        in_upper = motion_upper
        z_upper, dist_upper = self.upper_vae.encode_to_dist(in_upper) # bs, n_chunks, dim
        
        self.uj = uj
        
        # lower latent
        bs, n, lj = motion_lower.shape
        lj = lj // 3
        motion_lower = rc.axis_angle_to_matrix(motion_lower.reshape(bs, n, lj, 3))
        motion_lower = rc.matrix_to_rotation_6d(motion_lower).reshape(bs, n, lj*6)

        self.lj = lj

        # hands latent
        bs, n, hj, = motion_hands.shape
        hj = hj // 3
        motion_hands = rc.axis_angle_to_matrix(motion_hands.reshape(bs, n, hj, 3))
        motion_hands = rc.matrix_to_rotation_6d(motion_hands).reshape(bs, n, hj*6)
        in_hands = motion_hands
        z_hands, dist_hands = self.hands_vae.encode_to_dist(in_hands) # bs, n_chunks, dim
        
        self.hj = hj

        # facial latent
        bs, n, fj = motion_face.shape
        fj = fj // 3
        motion_face = rc.axis_angle_to_matrix(motion_face.reshape(bs, n, fj, 3))
        motion_face = rc.matrix_to_rotation_6d(motion_face).reshape(bs, n, fj*6)
        in_face = torch.cat([motion_face, motion_facial], dim=-1)
        z_face, dist_face = self.face_vae.encode_to_dist(in_face) # bs, n_chunks, dim
        
        self.fj = fj

        # translation latent
        motion_transl[:, :, 0] = motion_transl[:, :, 0] - motion_transl[:, 0:1, 0]
        motion_transl[:, :, 2] = motion_transl[:, :, 2] - motion_transl[:, 0:1, 2]
        tj = motion_transl.shape[-1]
        in_lowertrans = torch.cat([motion_lower, motion_transl, motion_contact], dim=-1)
        z_lowertrans, dist_lowertrans = self.lowertrans_vae.encode_to_dist(in_lowertrans) 

        self.tj = tj

        if self.body_part_cat_axis == "time":

            seperator = torch.zeros_like(z_upper[:, :1, :])
            motion = torch.cat([
                z_upper,
                seperator,
                z_hands,
                seperator,
                z_face,
                seperator,
                z_lowertrans
            ], dim=1) 
            
            motion_mask = motion_mask[:, ::self.frame_chunk_size]
            mask_sep = torch.zeros_like(motion_mask[:, :1])
            motion_mask = torch.cat([motion_mask, mask_sep, motion_mask, mask_sep, motion_mask, mask_sep, motion_mask], dim=1) #, mask_sep, motion_mask], dim=1)
        else:
            seperator = torch.zeros_like(z_upper[:, :, :1])
            motion = torch.cat([
                z_upper,
                seperator,
                z_hands,
                seperator,
                z_face,
                seperator,
                z_lowertrans
            ], dim=-1)
            motion_mask = motion_mask[:, ::self.frame_chunk_size]

        return motion, motion_mask
    
    def decode(self, z_output):
        bs, ntokens, dim = z_output.shape

        if self.body_part_cat_axis == "time":
            ntokens = (ntokens - 3) // 4
            outputz_upper = z_output[:, :ntokens, :]
            outputz_hands = z_output[:, ntokens + 1 : 2 * ntokens + 1, :]
            outputz_face = z_output[:, 2 * ntokens + 2 : 3 * ntokens + 2, :]
            outputz_lowertrans = z_output[:, 3 * ntokens + 3 :, :]

            assert outputz_upper.shape[1] == outputz_lowertrans.shape[1] == outputz_face.shape[1] == outputz_hands.shape[1]
            
        else:
            dim = (dim - 3) // 4
            outputz_upper = z_output[:, :, :dim]
            outputz_hands = z_output[:, :, dim + 1 : 2 * dim + 1]
            outputz_face = z_output[:, :, 2 * dim + 2 : 3 * dim + 2]
            outputz_lowertrans = z_output[:, :, 3 * dim + 3 :]

            assert outputz_upper.shape[2] == self.vae_latent_dim
            assert outputz_upper.shape[2] == outputz_lowertrans.shape[2] == outputz_face.shape[2] == outputz_hands.shape[2] 


        # upper latent
        output_upper = self.upper_vae.decode(outputz_upper)
        bs, n, _ = output_upper.shape
        output_upper = output_upper.reshape(bs, n, self.uj, 6)
        output_upper = rc.rotation_6d_to_matrix(output_upper)
        output_upper = rc.matrix_to_axis_angle(output_upper).reshape(bs, n, self.uj * 3)

        # hands latent
        output_hands = self.hands_vae.decode(outputz_hands)
        bs, n, _ = output_hands.shape
        output_hands = output_hands.reshape(bs, n, self.hj, 6)
        output_hands = rc.rotation_6d_to_matrix(output_hands)
        output_hands = rc.matrix_to_axis_angle(output_hands).reshape(bs, n, self.hj * 3)

        # facial latent
        output_face = self.face_vae.decode(outputz_face)
        output_facej = output_face[:, :, :self.fj * 6]
        output_exps = output_face[:, :, self.fj * 6:]
        bs, n, _ = output_facej.shape
        output_facej = output_facej.reshape(bs, n, self.fj, 6)
        output_facej = rc.rotation_6d_to_matrix(output_facej)
        output_facej = rc.matrix_to_axis_angle(output_facej).reshape(bs, n, self.fj * 3)

        output_exps = output_exps.reshape(bs, n, 100)

        # lower translation latent
        assert self.tj == 3
        outputz_lowertranscontact = self.lowertrans_vae.decode(outputz_lowertrans)
        output_lower = outputz_lowertranscontact[:, :, :self.lj * 6]
        output_transl = outputz_lowertranscontact[:, :, self.lj * 6: self.lj * 6 + self.tj]
        output_contact = outputz_lowertranscontact[:, :, self.lj * 6 + self.tj:]

        bs, n, _ = output_lower.shape
        output_lower = output_lower.reshape(bs, n, self.lj, 6)
        output_lower = rc.rotation_6d_to_matrix(output_lower)
        output_lower = rc.matrix_to_axis_angle(output_lower).reshape(bs, n, self.lj * 3)

        return output_upper, output_lower, output_facej, output_hands, output_transl, output_exps, output_contact



class DiffusionTransformer(BaseModule, metaclass=ABCMeta):
    def __init__(
        self,
        input_feats,
        max_seq_len=240,
        frame_chunk_size=16,
        latent_dim=512,
        time_embed_dim=2048,
        num_layers=8,
        sa_block_cfg=None,
        ca_block_cfg=None,
        vae_cfg=None,
        ffn_cfg=None,
        text_encoder=None,
        audio_encoder=None,
        speaker_embedding=None,
        use_cache_for_text=False,
        init_cfg=None,
        body_part_cat_axis="time",
    ):
        super().__init__(init_cfg=init_cfg)
        self.input_feats = input_feats
        # self.max_seq_len = max_seq_len
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.time_embed_dim = time_embed_dim

        # vae_cfg = None
        if vae_cfg is not None:
            self.gesture_rep_encoder = GestureRepEncoder(vae_cfg, body_part_cat_axis)
        else:
            self.gesture_rep_encoder = None

        max_seq_len = max_seq_len if self.gesture_rep_encoder is None else (max_seq_len // frame_chunk_size) # * 2
        self.max_seq_len = max_seq_len
        self.frame_chunk_size = frame_chunk_size

        self.body_part_cat_axis = body_part_cat_axis

        self.sequence_embedding = PositionEmbeddingSine1D(
            latent_dim,
            max_seq_len,
            batch_first=True
        )
        self.global_positional_embedding = PositionEmbeddingLearned1D(
            latent_dim, 
            max_seq_len * 4 + 3 if body_part_cat_axis == "time" else max_seq_len,
            batch_first=True
        )

        self.use_cache_for_text = use_cache_for_text
        if use_cache_for_text:
            self.text_cache = {}
        self.build_text_encoder(text_encoder)

        # Add wav2vec2 encoder
        self.build_audio_encoder(audio_encoder)

        self.build_spk_embedding(speaker_embedding)

        # Input Embedding
        if self.gesture_rep_encoder is None:
            self.joint_embed = nn.Linear(self.input_feats, self.latent_dim)
        else:
            self.joint_embed = nn.Linear(
                self.gesture_rep_encoder.vae_latent_dim if self.body_part_cat_axis == "time" else self.gesture_rep_encoder.vae_latent_dim * 4 + 3, #* 5 + 4,
                self.latent_dim,
            )
        

        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, self.time_embed_dim),
            nn.SiLU(),
            nn.Linear(self.time_embed_dim, self.time_embed_dim),
        )
        self.build_temporal_blocks(sa_block_cfg, ca_block_cfg, ffn_cfg)

        # Output Module
        if self.gesture_rep_encoder is None:
            self.out = zero_module(nn.Linear(self.latent_dim, self.input_feats))
        else:
            self.out = zero_module(
                nn.Linear(
                    self.latent_dim,
                    self.gesture_rep_encoder.vae_latent_dim if self.body_part_cat_axis == "time" else self.gesture_rep_encoder.vae_latent_dim * 4 + 3, #* 5 + 4,
                )
            )

    def build_temporal_blocks(self, sa_block_cfg, ca_block_cfg, ffn_cfg):
        self.temporal_decoder_blocks = nn.ModuleList()
        for i in range(self.num_layers):
            self.temporal_decoder_blocks.append(
                DecoderLayer(
                    sa_block_cfg=sa_block_cfg,
                    ca_block_cfg=ca_block_cfg,
                    ffn_cfg=ffn_cfg,
                )
            )
    

    def build_text_encoder(self, text_encoder):

        text_latent_dim = text_encoder["latent_dim"]
        num_text_layers = text_encoder.get("num_layers", 0)
        text_ff_size = text_encoder.get("ff_size", 2048)
        pretrained_model = text_encoder["pretrained_model"]
        text_num_heads = text_encoder.get("num_heads", 4)
        dropout = text_encoder.get("dropout", 0.1)
        activation = text_encoder.get("activation", "gelu")
        self.use_text_proj = text_encoder.get("use_text_proj", False)

        if pretrained_model == "bert":
            self.use_audio_model = True
            self.bert_tokenizer = transformers.AutoTokenizer.from_pretrained(
                "google-bert/bert-base-cased",
                max_length=512,
                max_position_embeddings=1024,
            )
            
            self.bert = transformers.BertModel.from_pretrained(
                "google-bert/bert-base-cased"
            )

            # freeze bert
            self.bert.training = False
            for p in self.bert.parameters():
                p.requires_grad = False
            
        else:
            self.use_text_model = False

        if num_text_layers > 0:
            self.use_text_finetune = True
            textTransEncoderLayer = nn.TransformerEncoderLayer(
                d_model=text_latent_dim,
                nhead=text_num_heads,
                dim_feedforward=text_ff_size,
                dropout=dropout,
                activation=activation,
            )
            self.textTransEncoder = nn.TransformerEncoder(
                textTransEncoderLayer, num_layers=num_text_layers
            )
            self.text_ln = nn.LayerNorm(text_latent_dim)
        else:
            self.use_text_finetune = False
        

        if text_latent_dim != self.latent_dim:
            self.text_pre_proj = nn.Linear(text_latent_dim, self.latent_dim)
        else:
            self.text_pre_proj = nn.Identity()
        
        if self.use_text_proj:
            self.text_proj = nn.Sequential(
                nn.Linear(self.latent_dim, self.time_embed_dim)
            )

    def build_audio_encoder(self, audio_encoder):
        # Add wav2vec2 encoder
        latent_dim = audio_encoder["latent_dim"]
        num_layers = audio_encoder.get("num_layers", 0)
        dropout = audio_encoder.get("dropout", 0)
        pretrained_model = audio_encoder["pretrained_model"]
        cp_path = audio_encoder.get("cp_path", None)

        if pretrained_model == "wav2vec":
            self.use_audio_model = True
            audio_model, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task(
                [cp_path]
            )
            audio_model = audio_model[0]
            for param in audio_model.parameters():
                param.requires_grad = False
            audio_model.eval()
            self.audio_model = audio_model
        else:
            self.use_audio_model = False

        if num_layers > 0:
            self.use_audio_finetune = True
            audioTransEncoderLayer = nn.TransformerEncoderLayer(
                d_model=latent_dim,
                nhead=4,
                dim_feedforward=2048,
                dropout=dropout,
                activation="gelu",
            )
            self.audioTransEncoder = nn.TransformerEncoder(
                audioTransEncoderLayer, num_layers=num_layers
            )
            self.audio_ln = nn.LayerNorm(latent_dim)
        else:
            self.use_audio_finetune = False
        

        if latent_dim != self.latent_dim:
            self.audio_pre_proj = nn.Linear(latent_dim, self.latent_dim)
        else:
            self.audio_pre_proj = nn.Identity()
        
        if self.use_text_proj:
            self.audio_proj = nn.Sequential(nn.Linear(self.latent_dim, self.time_embed_dim))

    def build_spk_embedding(self, speaker_embedding):
        self.num_speakers = speaker_embedding["num_speakers"]
        self.speaker_embedding = nn.Embedding(self.num_speakers, self.latent_dim)
        self.speaker_embedding.weight.data.normal_(mean=0, std=1)
        self.speaker_embedding.weight.data /= self.latent_dim

    def encode_spks(self, spk_ids, device):
        if self.num_speakers == 1:
            return torch.zeros((spk_ids.shape[0], spk_ids.shape[0], self.latent_dim), device=spk_ids.device)
        
        return self.speaker_embedding(spk_ids)

    def encode_text(self, text, device):
        B = len(text)
        
        if self.use_text_model:
            tokenized_text = self.bert_tokenizer(
                text, return_tensors="pt", padding=True
            ).to(device)
            x = self.bert(**tokenized_text).last_hidden_state
        else:
            x = text
        
        # B, T, D
        x = self.text_pre_proj(x)

        if self.use_text_finetune:
            x = x.permute(1, 0, 2)  # T, B, D
            xf_out = self.textTransEncoder(x)
            xf_out = self.text_ln(xf_out)
            xf_out = xf_out.permute(1, 0, 2) # B, T, D
        else:
            xf_out = x
        

        if self.use_text_proj:
            xf_proj = self.text_proj(xf_out)
            # B, T, D
            
            return xf_proj, xf_out
        else:
            return xf_out

    def encode_audio(self, audio, device):
        
        B = len(audio)
        
        # breakpoint()
        if self.use_audio_model:
            audio = audio.to(device)
            audio = self.audio_model.feature_extractor(audio)
            audio = audio.permute(0, 2, 1)  # B, D, T -> B, T, D # specific to wav2vec

        audio = self.audio_pre_proj(audio)
        if self.use_audio_finetune:
            audio = audio.permute(1, 0, 2)  # T, B, D
            x = self.audio_ln(self.audioTransEncoder(audio))
            # B, T, D
            xf_out = x.permute(1, 0, 2)
        else:
            x = audio
        
        if self.use_text_proj:
            xf_proj = self.audio_proj(x)
            xf_out = x
            return xf_proj, xf_out
        else:
            xf_out = x
            return xf_out

    @abstractmethod
    def get_precompute_condition(self, **kwargs):
        pass

    @abstractmethod
    def forward_train(self, h, src_mask, emb, **kwargs):
        pass

    @abstractmethod
    def forward_test(self, h, src_mask, emb, **kwargs):
        pass

    def forward(self, motion, timesteps, motion_mask=None, **kwargs):
        """
        motion latent: B, T, D
        """
        do_clf_guidance = kwargs.get("do_clf_guidance", False)
        B, T = motion.shape[0], motion.shape[1]
        
        
        query_att_mask = kwargs.get("query_mask", None)
        query_mask = copy.deepcopy(query_att_mask)
        
        conditions = self.get_precompute_condition(device=motion.device, **kwargs)
        if len(motion_mask.shape) == 2:
            src_mask = motion_mask.clone().unsqueeze(-1)
        else:
            src_mask = motion_mask.clone()
        
        if self.use_text_proj:
            emb = (
                self.time_embed(timestep_embedding(timesteps, self.latent_dim))
                + conditions["xf_proj"]
            )
        else:
            emb = self.time_embed(timestep_embedding(timesteps, self.latent_dim))
        
        # B, T, latent_dim
        h = self.joint_embed(motion)

        # positional embedding for joined body parts
        if self.body_part_cat_axis == "time":
            # here sequence_embedding should have shape of B, T, D
            pos = self.sequence_embedding(h[:, :(T-3)//4, :], return_only_pos=True) # 4 body parts and 3 separators
            sep = torch.zeros_like(pos[:, :1, :])
            pos_cat = torch.cat([pos, sep, pos, sep, pos, sep, pos], dim=1) # 4 body parts and 3 separators
            h = h + pos_cat
        else:
            h = self.sequence_embedding(h)

        # global positional embedding
        h = self.global_positional_embedding(h)
        
        if self.training:
            return self.forward_train(
                h=h, src_mask=src_mask, emb=emb, timesteps=timesteps, query_mask=query_mask, **conditions
            )
        else:
            return self.forward_test(
                h=h, src_mask=src_mask, emb=emb, timesteps=timesteps, query_mask=query_mask, do_clf_guidance=do_clf_guidance, **conditions
            )
