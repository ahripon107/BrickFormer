# Jamba-style hybrid: Mamba2 layers with one attention layer per 8, MoE on every other layer.
_moe = dict(type="MoE", num_experts=8, top_k=2, hidden=1408, aux_loss_coef=0.01)
_dense = dict(type="SwiGLU", hidden=2816)

model = dict(
    type="CausalLM",
    vocab_size=32000,
    dim=1024,
    # one "period" of 8 layers, tiled 2x -> 16 layers
    pattern=["mamba", "mamba_moe", "mamba", "mamba_moe", "attn", "mamba_moe", "mamba", "mamba_moe"],
    num_layers=16,
    block_types=dict(
        attn=dict(mixer=dict(type="GQAttention", heads=16, kv_heads=8, rope=False), ffn=_dense),
        mamba=dict(mixer=dict(type="Mamba2", d_state=64, headdim=64, expand=2), ffn=_dense),
        mamba_moe=dict(mixer=dict(type="Mamba2", d_state=64, headdim=64, expand=2), ffn=_moe),
    ),
    norm=dict(type="RMSNorm"),
    residual="pre_norm",
    tie_embeddings=False,
)
