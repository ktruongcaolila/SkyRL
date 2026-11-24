# Faster Rollouts with MTP Speculative Decoding

This example shows how you can train a model's native **Multi-Token Prediction (MTP)** head with a decoupled draft loss, and reuse that head for **vLLM speculative decoding** to speed up rollout. The draft loss is autograd-decoupled — it trains only the MTP head and never perturbs the policy, so the RL dynamics are unchanged.

Requires the `megatron` backend and the `vllm` inference engine, plus an MTP-capable model.

`trainer.mtp` is the single high-level knob; it propagates to both the Megatron MTP heads + draft loss and vLLM's `speculative_config`:

```bash
trainer.mtp.enabled=true \
trainer.mtp.num_speculative_tokens=3 \      # vLLM draft depth
trainer.mtp.loss_weight=0.2 \               # weight of the draft loss
trainer.policy.megatron_config.mtp_loss_topk=256   # top-k soft CE (memory-efficient)
```

Three recipes run on a single 8×H100 node, and one is multi-node:

- **`run_megatron_dapo_mimo_7b_rl_specdecode.sh`** — DAPO on `dapo-math-17k` with `XiaomiMiMo/MiMo-7B-RL`.
- **`run_megatron_dapo_qwen3.5_9b_specdecode.sh`** — DAPO on `dapo-math-17k` with `Qwen/Qwen3.5-9B`.
- **`run_search_megatron_mimo_7b_specdecode.sh`** — multi-turn GRPO SearchR1 with `XiaomiMiMo/MiMo-7B-RL`.
- **`run_megatron_dapo_glm5.3_specdecode.sh`** — DAPO on `dapo-math-17k` with `zai-org/GLM-5.3-BF16` (753B MoE) on 8×8 B200; see [below](#multi-node-recipe-glm-53).

See [the docs](https://docs.skyrl.ai/docs/examples/spec_decode) for the full walkthrough, configuration reference, and results (~2x rollout speedup with no effect on policy training).

## Quick start

```bash
# DAPO recipes
bash examples/train/algorithms/dapo/prepare_dapo_data.sh
export WANDB_API_KEY=<your_key_here>
bash examples/train/spec_decode/run_megatron_dapo_mimo_7b_rl_specdecode.sh

# SearchR1 recipe requires extra setup (dataset + retrieval server) —
# see examples/train/search/README.md, then:
bash examples/train/spec_decode/run_search_megatron_mimo_7b_specdecode.sh
```

## Multi-node recipe (GLM-5.3)

`run_megatron_dapo_glm5.3_specdecode.sh` trains GLM-5.3 (753B, 256 routed experts per layer) with k=4 MTP drafts on 8 nodes of 8×B200. It is not colocated, because neither side fits a smaller footprint:

- **6 trainer nodes** (Megatron TP1 / CP16 / PP3 / EP16, DeepEP, host AdamW). bf16 weights and grads are ~3 TB, ~100 GB per GPU at this layout. At 4 nodes (EP8) model init runs out of memory.
- **2 inference nodes**, one vLLM engine per node (DP8 × EP8). NVFP4 routed experts and an NVFP4 KV cache make a whole engine fit in one node (87 GiB of weights per GPU, ~1.1M KV tokens per rank). With two engines, each of the 16 DP ranks decodes 16 sequences per step, where MTP helps most.
- Weights move over `sharded_rdt`; the MTP drafter rides the target's sessions.

**Ray cluster.** Start Ray on all 8 nodes and launch from the head node, as in the [installation docs](https://docs.skyrl.ai/docs/getting-started/installation). Set `RAY_RUNTIME_ENV_HOOK=ray._private.runtime_env.uv_runtime_env_hook.hook` so every worker re-runs `uv run`.

Settings SkyRL does not forward to the workers are set in the script itself:
- the vLLM KV-offload and DSA-indexer settings;
- DeepEP over EFA between the trainer nodes.

The script writes them to an env file next to itself and passes it with `uv run --env-file`. Ray replays that on every worker, so keep the checkout at the same path on every node (a shared checkout). Add your NCCL / EFA settings, `WANDB_API_KEY` and `HF_TOKEN` with `EXTRA_ENV_FILE=/path/to/.env`. The checkpoint and data must also be at the same path on every node.

```bash
bash examples/train/algorithms/dapo/prepare_dapo_data.sh
bash examples/train/spec_decode/run_megatron_dapo_glm5.3_specdecode.sh                    # MTP
MTP_ENABLED=false bash examples/train/spec_decode/run_megatron_dapo_glm5.3_specdecode.sh  # baseline
```

**Results.** 6 steps per arm, one arm after the other on the same 8 nodes; means over steps 2–5. DAPO on `dapo-math-17k`, 16 prompts × 16 samples per step, `max_generate_length=8192`, depth-4 probabilistic drafting:

| Phase | Baseline | MTP | Speedup |
|---|---:|---:|---:|
| **Total step** | **287 s** | **170 s** | **1.68x** |
| Rollout | 205 s | 81 s | **2.51x** |
| Policy train | 53.4 s | 58.3 s | 0.92x |
| Weight sync | 11.6 s | 11.7 s | 0.99x |

Mean acceptance length was **3.73 tokens**, with 89.0 / 74.7 / 60.6 / 48.3% acceptance at depth 1 / 2 / 3 / 4.

Reward and the rollout/trainer log-prob gap are similar in both arms (one run each, over steps 2-5): 0.24 vs 0.29 raw reward, 0.067 vs 0.072 mean |log-prob diff|.

## Monitoring acceptance

Acceptance determines your speedup. It's logged under `vllm/train/` — most importantly `draft_mean_acceptance_length` (tokens per target forward pass; this *is* the speedup factor) and the per-depth breakdown `draft_acceptance_rate_pos_{1..k}`. The draft loss is logged as `mtp_loss` on the training side. See [the docs](https://docs.skyrl.ai/docs/examples/spec_decode#monitoring-acceptance) for the full list.
