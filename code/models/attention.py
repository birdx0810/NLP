# -*- coding: utf-8 -*-
"""
This is an example for implementing a Transformer model class using PyTorch.
Notes:
- Most of the code is referenced from the Annotated Transformer: https://nlp.seas.harvard.edu/2018/04/03/attention.html
"""

class Transformer(nn.Module):
    def __init__(self, encoder, decoder, src_embed, tgt_embed):
        super(Transformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

        self.src_embed = torch.nn.Embedding()
        if src_embed is not None:
            self.src_embed.weight = nn.Parameter(src_embed)
        else:
            with torch.no_grad():
                for parameter in self.src_embed.parameters():
                    parameter.normal_(mean=0.0, std=0.1)

        self.tgt_embed = torch.nn.Embedding()
        if tgt_embed is not None:
            self.tgt_embed.weight = nn.Parameter(tgt_embed)
        else:
            with torch.no_grad():
                for parameter in self.tgt_embed.parameters():
                    parameter.normal_(mean=0.0, std=0.1)

    def forward(self, src, tgt, src_mask, tgt_mask):
        """
        Take in and process masked src and target sequences.
        """
        # Encoding step
        # B x S x E
        src_embed = self.src_embed(src)

        # B x S x H
        encoded_tensor = self.encoder(src_embed, src_mask)

        # Decoding Step
        # B x S x E
        tgt_embed = self.tgt_embed(tgt)

        # B x S x H
        decoded_tensor = self.decoder(encoded_tensor, src_mask, tgt, tgt_mask)

        # Project to target vocabulary
        # B x S x V
        logits = decoder_tensor.matmul(tgt_embed.weight.transpose(0,1))

        return logits

def attention(query, key, value, mask=None, dropout=None):
    """Scaled dot product attention
    """
    d_k = query.size(-1)

    # A = (Q x K^T)/ sqrt(d_k)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_filled(mask == 0, -1e10)

    p_attn = F.softmax(scores, dim=-1)

    if dropout is not None:
        p_attn = dropout(p_attn)

    # A x V
    return torch.matmul(p_attn, value), p_attn

class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v == d_k
        self.d_k = d_model // heads
        self.heads = heads
        self.linear = None
        self.attn = None
        self.dropout = dropout

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            # Same mask applied to all h heads
            mask = mask.unsqueeze(1)

        nbatches = query.size(0)

        # 1. Do all linear projection in batch from d_model
        query, key, value = \
            [l(x),view(nbatches, -1, self.heads, self.d_k).transpose(1,2)
             for l, x in zip(self.linear, (query, key, value))]

        # 2. Apply attention on all the projected vectors in batch
        x, self.attention = attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3. "Concat" using a view and apply a final linear
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.heads * self.d_k)

        return self.linears[-1].(x)



