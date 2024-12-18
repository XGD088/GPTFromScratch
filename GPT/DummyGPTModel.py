import tiktoken
import torch
import torch.nn as nn


GPT_CONFIG_124M = {
    "vocab_size": 50257,  # Vocabulary size
    "context_length": 1024,  # Context length
    "emb_dim": 768,  # Embedding dimension
    "n_heads": 12,  # Number of attention heads
    "n_layers": 12,  # Number of layers
    "drop_rate": 0.1,  # Dropout rate
    "qkv_bias": False  # Query-Key-Value bias
}


class DummyGPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.Sequential(*[DummyTransformerBlock(cfg) for _ in range(cfg["n_layers"])])  # A
        self.final_norm = LayerNorm(cfg["emb_dim"])  # B
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        )

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits


class DummyTransformerBlock(nn.Module):  # C
    def __init__(self, cfg):
        super().__init__()

    def forward(self, x):  # D
        return x


class LayerNorm(nn.Module):  # E
    def __init__(self, emb_dim, eps=1e-5):  # F
        super().__init__()
        self.eps = eps
        #缩放
        self.scale = nn.Parameter(torch.ones(emb_dim))
        #偏移
        self.shift = nn.Parameter(torch.zeros(emb_dim))


    def forward(self, x):
        #平均数
        mean = x.mean(dim = -1, keepdim=True)
        #方差
        var = x.var(dim = -1, keepdim=True, unbiased=False)
        #归一化，通过将数据中轴向0点移动，实现了0均值。并且通过除以标准差，统一了数据分布的尺度。
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return norm_x * self.scale + self.shift


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(cfg["emb_dim"], cfg["emb_dim"] * 4),
                                    nn.GELU(),
                                    nn.Linear(cfg["emb_dim"] * 4, cfg["emb_dim"]))

    def forward(self, x):
        return self.layers(x)

