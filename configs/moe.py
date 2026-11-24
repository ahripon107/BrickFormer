# DeepSeek-V2-lite-style: MLA attention, first layer dense, the rest MoE with a shared expert.
_mla = dict(type="MLA", heads=16, kv_lora_rank=256, q_lora_rank=None,
            nope_head_dim=64, rope_head_dim=32, v_head_dim=64)

model = dict(
    type="CausalLM",
    vocab_size=32000,
    dim=1024,
    pattern=["dense"] + ["moe"] * 11,
    block_types=dict(
        dense=dict(mixer=_mla, ffn=dict(type="SwiGLU", hidden=2816)),
        moe=dict(mixer=_mla, ffn=dict(type="MoE", num_experts=16, top_k=2, hidden=704, shared_experts=1,
                                      aux_loss_coef=0.001, z_loss_coef=1e-4)),
    ),
    norm=dict(type="RMSNorm"),
    residual="pre_norm",
    tie_embeddings=True,
)
