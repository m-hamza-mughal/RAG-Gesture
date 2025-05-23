from functools import reduce
from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, nn
from torch.distributions.distribution import Distribution
from collections import OrderedDict

# import sys
# import os

# SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(os.path.dirname(SCRIPT_DIR))
# sys.path.append(
#     "/CT/GestureSynth1/work/DiscourseAwareGesture/RAGGesture_BEATX/mogen/models/utils"
# )
# sys.path.append(
#     "/CT/GestureSynth1/work/DiscourseAwareGesture/RAGGesture_BEATX/mogen/models/transformers"
# )

from ..utils.detr_utils import (
    TransformerEncoderLayer,
    TransformerDecoderLayer,
    SkipTransformerDecoder,
    SkipTransformerEncoder,
    PositionEmbeddingLearned1D,
    PositionEmbeddingSine1D
)
from ..utils.detr_utils import lengths_to_mask



class TransformerVAE(nn.Module):

    def __init__(self, args, **kwargs) -> None:

        super().__init__()

        self.latent_dim = args.latent_dim
        self.frame_chunk_size = args.frame_chunk_size
        self.arch = args.decoder_arch
        self.pe_type = args.position_embedding
        self.num_frames = args.num_frames

        if args.position_embedding == "learned": 
            self.query_pos_encoder = PositionEmbeddingLearned1D(args.latent_dim)
            self.query_pos_decoder = PositionEmbeddingLearned1D(args.latent_dim)
            self.mem_pos_decoder = PositionEmbeddingLearned1D(args.latent_dim)
        elif args.position_embedding == "sine":
            self.query_pos_encoder = PositionEmbeddingSine1D(args.latent_dim)
            self.query_pos_decoder = PositionEmbeddingSine1D(args.latent_dim)
            self.mem_pos_decoder = PositionEmbeddingSine1D(args.latent_dim)

        encoder_layer = TransformerEncoderLayer(
            self.latent_dim,
            args.num_heads,
            args.ff_size,
            args.dropout,
            args.transformer_activation,
            args.transformer_normalize_before,
        )
        encoder_norm = nn.LayerNorm(self.latent_dim)
        self.encoder = SkipTransformerEncoder(
            encoder_layer, args.num_layers, encoder_norm
        )

        if self.arch == "all_encoder":
            decoder_norm = nn.LayerNorm(self.latent_dim)
            decoder_layer = TransformerEncoderLayer(
                self.latent_dim,
                args.num_heads * 8,
                args.ff_size,
                args.dropout,
                args.transformer_activation,
                args.transformer_normalize_before,
            )
            self.decoder = SkipTransformerEncoder(
                decoder_layer, args.num_layers, decoder_norm
            )
            
        elif self.arch == "encoder_decoder":
            decoder_layer = TransformerDecoderLayer(
                self.latent_dim,
                args.num_heads * 4,
                args.ff_size,
                args.dropout,
                args.transformer_activation,
                args.transformer_normalize_before,
            )
            decoder_norm = nn.LayerNorm(self.latent_dim)
            self.decoder = SkipTransformerDecoder(
                decoder_layer, (args.num_layers - 1) * 4 + 1, decoder_norm
            )
        else:
            raise ValueError("Not support architecture!")

        self.global_motion_token = nn.Parameter(
            torch.randn(2, self.latent_dim) # 2 for mu and logvar
        )

        self.skel_embedding = nn.Linear(args.nfeats, self.latent_dim)
        self.final_layer = nn.Linear(self.latent_dim, args.nfeats)

        self.dist_type = args.vae_dist
        if self.dist_type == "multivariate_normal":
            self.softplus = nn.Softplus()

    def forward(self, features: Tensor, lengths: Optional[List[int]] = None):
        z, dist = self.encode_to_dist(features, lengths)
        feats_rst = self.decode(z, lengths)
        
        return {
            "rec_pose": feats_rst, 
            "poses_feat": z, 
            "rec_dist": dist
        }
    

    def encode_to_dist(self, features: Tensor, lengths: Optional[List[int]] = None):
        # breakpoint()
        bs, nframes, nfeats = features.shape
        n_chunks = nframes // self.frame_chunk_size

        z = self.encode(features, lengths)
        
        z, dist = self.reparameterize(z)
        z = z.reshape(bs, n_chunks, self.latent_dim)
        
        return z, dist


    def encode(
        self, features: Tensor, lengths: Optional[List[int]] = None
    ) -> Union[Tensor]:
        if lengths is None:
            lengths = [len(feature) for feature in features]
        # print(features.shape)
        device = features.device

        bs, nframes, nfeats = features.shape
        mask = lengths_to_mask(lengths, device)

        # breakpoint()
        n_chunks = nframes // self.frame_chunk_size  # changed from 16  # 128 // 16 = 8, 256 // 16 = 16
        motion_feats = features.clone()
        motion_feats = motion_feats.reshape(
            bs * n_chunks, nframes // n_chunks, -1
        )  # 128 // 8 = 16, 256 // 16 = 16

        # breakpoint()
        mask = mask.reshape(bs * n_chunks, nframes // n_chunks)
        bs = bs * n_chunks

        x = motion_feats

        # Switch sequence and batch_size because the input of
        # Pytorch Transformer is [Sequence, Batch size, ...]
        # breakpoint()
        x = x.permute(1, 0, 2)  # now it is [nframes, bs, latent_dim]

        # Embed each human poses into latent vectors
        x = self.skel_embedding(x)

        # Each batch has its own set of tokens
        dist = torch.tile(self.global_motion_token[:, None, :], (1, bs, 1))

        # create a bigger mask, to allow attend to emb
        dist_masks = torch.ones((bs, dist.shape[0]), dtype=bool, device=x.device)
        aug_mask = torch.cat((dist_masks, mask), 1)

        # adding the embedding token for all sequences
        xseq = torch.cat((dist, x), dim=0)

        xseq = self.query_pos_encoder(xseq)
        latent = self.encoder(xseq, src_key_padding_mask=~aug_mask)[:dist.shape[0]]

        # breakpoint()
        latent = latent.permute(1, 0, 2)
        return latent
    
    def reparameterize(self, latent) -> Union[Tensor, Distribution]:
        # content distribution
        mu = latent[:, 0 : 1, :]
        logvar = latent[:, 1 :, :]

        # # resampling
        if self.dist_type == "normal":
            std = logvar.exp().pow(0.5)
            dist = torch.distributions.Normal(mu, std)

        elif self.dist_type == "multivariate_normal":
            scale = self.softplus(logvar) + 1e-8
            scale_tril = torch.diag_embed(scale)
            dist = torch.distributions.MultivariateNormal(mu, scale_tril=scale_tril)
        else:
            raise ValueError("Not support distribution type!")

        latent = dist.rsample()
        
        
        return latent, dist

    def decode(self, z: Tensor, lengths: Optional[List[int]] = None):
        # print(z.shape)
        bs, n_chunks, _ = z.shape
        
        if lengths is None:
            lengths = [self.num_frames] * bs

        mask = lengths_to_mask(lengths, z.device)
        bs, nframes = mask.shape

        queries = torch.zeros(nframes, bs, self.latent_dim, device=z.device)
        
        z = z.permute(1, 0, 2) # now it is [nframes, bs, latent_dim]

        # Pass through the transformer decoder
        # with the latent vector for memory
        if self.arch == "all_encoder":
            xseq = torch.cat((z, queries), dim=0)
            z_mask = torch.ones((bs, z.shape[0]), dtype=bool, device=z.device)
            augmask = torch.cat((z_mask, mask), dim=1)

            query_pos = self.query_pos_decoder(xseq)
            output = self.decoder(
                xseq, pos=query_pos, src_key_padding_mask=~augmask
            )[z.shape[0] :]

        elif self.arch == "encoder_decoder":
            queries = self.query_pos_decoder(queries)
            z = self.mem_pos_decoder(z)
            output = self.decoder(
                tgt=queries,
                memory=z,
                tgt_key_padding_mask=~mask,
                # query_pos=query_pos,
                # pos=mem_pos,
            ).squeeze(0)

        output = self.final_layer(output)  # [nframes, bs, body_nfeats]

        # zero for padded area
        output[~mask.T] = 0

        feats = output.permute(1, 0, 2)

        return feats


if __name__ == "__main__":
    import yaml
    import os
    from argparse import Namespace

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

    config_path = "/CT/GestureSynth1/work/GestureGPT/GestureRep/experiments/0903_020101_gesture_lexicon_transformer_vae_upper_allspk_len256_l8h4_fchunksize15/0903_020101_gesture_lexicon_transformer_vae_upper_allspk_len256_l8h4_fchunksize15.yaml"
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        
    args = Namespace(**config)
    checkpoint_path = args.test_ckpt
    checkpoint_path = os.path.dirname(config_path) + "/" + os.path.basename(checkpoint_path)

    model = TransformerVAE(args).cuda()
    load_checkpoints(model, checkpoint_path)
    model.eval()

    features = torch.randn(2, 150, 78).cuda()
    out = model(features)
    print(out['rec_pose'].shape, out['poses_feat'].shape, out['rec_dist'])