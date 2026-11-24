# Llama-style dense transformer: GQA + RoPE + SwiGLU, pre-norm RMSNorm. ~125M params.
model = dict(
    type="CausalLM",
    vocab_size=32000,
    dim=768,
    pattern="A",
    num_layers=12,
    block_types=dict(
        attn=dict(
            mixer=dict(type="GQAttention", heads=12, kv_heads=4, rope=dict(type="RoPE", theta=500000.0)),
            ffn=dict(type="SwiGLU", hidden=2048),
        ),
    ),
    norm=dict(type="RMSNorm", eps=1e-6),
    residual="pre_norm",
    tie_embeddings=True,
)
