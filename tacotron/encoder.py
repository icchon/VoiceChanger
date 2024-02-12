# Acknowledgement: some of the code was adapted from ESPnet
#  Copyright 2019 Nagoya University (Tomoki Hayashi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def encoder_init(m):
    if isinstance(m, nn.Conv1d):
        nn.init.xavier_uniform_(m.weight, nn.init.calculate_gain("relu"))


class Encoder(nn.Module):
    """Encoder of Tacotron 2

    Args:
        num_vocab (int): number of vocabularies
        embed_dim (int): dimension of embeddings
        hidden_dim (int): dimension of hidden units
        conv_layers (int): number of convolutional layers
        conv_channels (int): number of convolutional channels
        conv_kernel_size (int): size of convolutional kernel
        dropout (float): dropout rate
    """

    def __init__(
        self,
        num_vocab=51,  # 語彙数
        embed_dim=512,  # 文字埋め込みの次元数
        hidden_dim=512,  # 隠れ層の次元数
        conv_layers=3,  # 畳み込み層数
        conv_channels=512,  # 畳み込み層のチャネル数
        conv_kernel_size=5,  # 畳み込み層のカーネルサイズ
        dropout=0.5,  # Dropout 率
    ):
        super(Encoder, self).__init__()
        # 文字の埋め込み表現
        self.embed = nn.Embedding(num_vocab, embed_dim, padding_idx=0)
        # 1 次元畳み込みの重ね合わせ：局所的な時間依存関係のモデル化
        convs = nn.ModuleList()
        for layer in range(conv_layers):
            in_channels = embed_dim if layer == 0 else conv_channels
            convs += [
                nn.Conv1d(
                    in_channels,
                    conv_channels,
                    conv_kernel_size,
                    padding=(conv_kernel_size - 1) // 2,
                    bias=False,  # この bias は不要です
                ),
                nn.BatchNorm1d(conv_channels),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
        self.convs = nn.Sequential(*convs)
        # Bi-LSTM による長期依存関係のモデル化
        self.blstm = nn.LSTM(
            conv_channels, hidden_dim // 2, 1, batch_first=True, bidirectional=True
        )

        # initialize
        self.apply(encoder_init)

    def forward(self, seqs, in_lens):
        """Forward step

        Args:
            seqs (torch.Tensor): input sequences
            in_lens (torch.Tensor): input sequence lengths

        Returns:
            torch.Tensor: encoded sequences
        """
        emb = self.embed(seqs)
        # 1 次元畳み込みと embedding では、入力の shape が異なるので注意
        out = self.convs(emb.transpose(1, 2)).transpose(1, 2)

        # Bi-LSTM の計算
        out = pack_padded_sequence(out, in_lens.to("cpu"), batch_first=True)
        out, _ = self.blstm(out)
        out, _ = pad_packed_sequence(out, batch_first=True)

        return out
