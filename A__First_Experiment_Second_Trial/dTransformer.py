import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


# x A_1 (x A_2).T
# (A_1.T x.T).T (A_2.T x.T)


# TODO maybe need Softmax before each multiplication to prevent explosion and problems with convergence

class dAttntionHead(nn.Module):
    def __init__(self, ninp, nhid, dropout, degree):
        super(dAttntionHead, self).__init__()
        self.ninp, self.nhid, self.dropout, self.degree = ninp, nhid, dropout, degree
        self.embeds = [nn.Linear(nhid, ninp) for _ in range(degree)]
        self.dropout = nn.Dropout(p=dropout, inplace=False)

    def forward(self, x):
        assert x.shape[-1] == self.ninp
        assert x.dim == 3

        x = self.dropout(x)

        calculated_intermidiate = []
        calculated = [x, ]
        for idx, layer in enumerate(self.embeds):
            calculated_intermidiate.append(
                self.dropout(layer(calculated[-1]))
            )
            if idx % 2 == 0:
                calculated.append(calculated_intermidiate[-1].T)
            else:
                calculated.append(calculated_intermidiate[-1])
        return calculated[-1]


class dAttntionLayer(nn.Module):
    def __init__(self, ninp, nhead, nhid, dropout, degree):
        super(dAttntionLayer, self).__init__()
        self.heads = [dAttntionHead(ninp, nhid, dropout, degree) for _ in range(nhead)]
        self.projection = nn.Linear(nhid * nhead, ninp)
        self.normalization_activation = nn.Sigmoid()

    def forward(self, x):
        heads_values = [head(x) for head in self.heads]
        values_ten = torch.stack(heads_values)
        projected = self.projection(values_ten)
        projected_normalized = self.normalization_activation(projected)
        return projected_normalized


class dTransformerModel(nn.Module):
    """Container module with an encoder, a recurrent or transformer module, and a decoder."""

    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5, degree=3):
        super(dTransformerModel, self).__init__()
        try:
            from torch.nn import TransformerEncoder, TransformerEncoderLayer
        except:
            raise ImportError('TransformerEncoder module does not exist in PyTorch 1.1 or lower.')
        self.model_type = 'dTransformer'
        self.degree = degree
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        degree_encoder_layers = self.degree_changer(encoder_layers, ninp, nhead, nhid, dropout, degree)
        self.transformer_encoder = TransformerEncoder(degree_encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, ntoken)

        self.init_weights()

    def degree_changer(self, encoder_layers, ninp, nhead, nhid, dropout, degree):
        encoder_layers.self_attn = dAttntionLayer(ninp, nhead, nhid, dropout, degree)
        return encoder_layers

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.encoder.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.weight)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, src, has_mask=True):
        if has_mask:
            device = src.device
            if self.src_mask is None or self.src_mask.size(0) != len(src):
                mask = self._generate_square_subsequent_mask(len(src)).to(device)
                self.src_mask = mask
        else:
            self.src_mask = None

        src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        output = self.decoder(output)
        return F.log_softmax(output, dim=-1)


###############
if __name__ == '__main__':
    t = dTransformerModel(33278, 200, 2, 200, 2,
                          0.2, degree=3)  # (ntokens, args.emsize, args.nhead, args.nhid, args.nlayers, args.dropout,)
