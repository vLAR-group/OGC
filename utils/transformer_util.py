import torch
import torch.nn as nn


class TransformerDecoderLayer(nn.Module):
    """
    A Transformer decoder layer (order adjusted) : Cross-Attention + Self-Attention.
    """
    def __init__(self,
                 embed_dim=256,
                 n_head=8,
                 hidden_dim=256):
        super().__init__()

        # Layer norm
        self.norm_slot1 = nn.LayerNorm(embed_dim)
        self.norm_slot2 = nn.LayerNorm(embed_dim)
        self.norm_pre_ff = nn.LayerNorm(embed_dim)

        # Cross & Self-Attention layers
        self.cross_attn = nn.MultiheadAttention(embed_dim, n_head, batch_first=True)
        self.self_attn = nn.MultiheadAttention(embed_dim, n_head, batch_first=True)

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, embed_dim)
        )


    def forward(self, slot, point_feats, pos_enc=None):
        """
        :param slot: (B, K, C) torch.Tensor.
        :param point_feats: (B, N, C) torch.Tensor.
        :param pos_enc: (B, N, C) torch.Tensor, if available.
        :return:
            slot: (B, K, C) torch.Tensor.
        """
        # Cross-Attention
        slot1 = self.norm_slot1(slot)
        if pos_enc is not None:
            slot1 = self.cross_attn(query=slot1,
                                     key=point_feats+pos_enc,
                                     value=point_feats)[0]
        else:
            slot1 = self.cross_attn(query=slot1,
                                     key=point_feats,
                                     value=point_feats)[0]
        slot = slot + slot1
        
        # Self-Attention
        slot2 = self.norm_slot2(slot)
        slot2 = self.self_attn(query=slot2,
                                key=slot2,
                                value=slot2)[0]
        slot = slot + slot2

        slot = slot + self.mlp(self.norm_pre_ff(slot))
        return slot


class MaskFormerHead(nn.Module):
    """
    Adapt from MaskFormer (NeurIPS'21).
    """
    def __init__(self,
                 n_slot,
                 input_dim=256,
                 n_transformer_layer=2,
                 transformer_embed_dim=256,
                 transformer_n_head=8,
                 transformer_hidden_dim=256,
                 input_pos_enc=False):
        super().__init__()

        self.n_slot = n_slot
        self.query = nn.Embedding(n_slot, transformer_embed_dim)

        self.mlp_input = nn.Sequential(
            nn.Linear(input_dim, transformer_embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(transformer_embed_dim, transformer_embed_dim)
        )
        self.norm_input = nn.LayerNorm(transformer_embed_dim)

        # Positional encoding (if needed)
        if input_pos_enc:
            self.input_pos_enc = nn.Linear(3, transformer_embed_dim)
        else:
            self.input_pos_enc = None

        # Transformer decoders
        self.transformer_layers = nn.ModuleList()
        for l in range(n_transformer_layer):
            self.transformer_layers.append(
                TransformerDecoderLayer(
                    embed_dim=transformer_embed_dim,
                    n_head=transformer_n_head,
                    hidden_dim=transformer_hidden_dim
                )
            )


    def forward(self, point_feats, point_pos):
        """
        :param inputs: (B, N, C) torch.Tensor.
        :param inputs_pos: (B, N, 3) torch.Tensor.
        """
        n_batch = point_feats.shape[0]
        slot = self.query(torch.arange(0, self.n_slot).expand(n_batch, self.n_slot).cuda())

        inputs = self.norm_input(self.mlp_input(point_feats))
        if self.input_pos_enc is not None:
            pos_enc = self.input_pos_enc(point_pos)
        else:
            pos_enc = None

        # Feed into Transformer decoders
        for l in range(len(self.transformer_layers)):
            slot = self.transformer_layers[l](slot, inputs, pos_enc)
        return slot