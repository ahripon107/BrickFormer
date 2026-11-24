# Gated DeltaNet hybrid: three GDN layers per full-attention layer (Qwen3-Next / Kimi-Linear style).
model = dict(
    type="CausalLM",
    vocab_size=32000,
    dim=1024,
    pattern=["gdn", "gdn", "gdn", "attn"] * 6,  # 24 layers; equivalently pattern="GGG*", num_layers=24
    block_types=dict(
        attn=dict(
            mixer=dict(type="GQAttention", heads=16, kv_heads=4, rope=True, qk_norm=True),
            ffn=dict(type="SwiGLU", hidden=2816),
        ),
        gdn=dict(
            mixer=dict(type="GatedDeltaNet", heads=8, head_dim=128, expand_v=1.0),
            ffn=dict(type="SwiGLU", hidden=2816),
        ),
    ),
    norm=dict(type="RMSNorm"),
    residual="pre_norm",
    tie_embeddings=True,
)
