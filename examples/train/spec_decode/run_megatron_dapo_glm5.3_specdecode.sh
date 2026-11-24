set -x

# Disaggregated DAPO training for GLM-5.3 (753B glm_moe_dsa MoE, full fine-tuning) on DAPO data with Megatron,
# with native Multi-Token Prediction (MTP) speculative decoding for faster rollout.
# Runs on 8x8 B200 (multi-node: 6 trainer + 2 inference nodes) on one Ray cluster; see README.md for the setup.
# bash examples/train/algorithms/dapo/prepare_dapo_data.sh
# bash examples/train/spec_decode/run_megatron_dapo_glm5.3_specdecode.sh                    # MTP
# MTP_ENABLED=false bash examples/train/spec_decode/run_megatron_dapo_glm5.3_specdecode.sh  # baseline

MODEL_NAME="${MODEL_NAME:-zai-org/GLM-5.3-BF16}"
DATA_DIR="${DATA_DIR:-$HOME/data/dapo}"
TRAIN_FILE="$DATA_DIR/dapo-math-17k-cleaned.parquet"
TEST_FILE="$DATA_DIR/aime-2024-cleaned.parquet"
# Trainer: 6 nodes (sizing under "megatron config" below).
NUM_NODES=6
NUM_GPUS_PER_NODE=8
# Inference: 2 nodes, one vLLM engine per node, each 8 data-parallel attention ranks sharing 8-way expert
# parallelism (DP8 x EP8, 1 GPU per rank). One node holds a whole engine only because the routed experts are
# NVFP4: ~725B expert params at ~4.5 bits are ~51 GB/GPU under EP8, and with the bf16 attention / dense weights
# that every DP rank keeps a GPU loads 87 GiB. That leaves 1.1M (MTP) to 1.4M tokens of NVFP4 KV per rank at
# gpu_memory_utilization=0.77, against the 16 x 10K tokens a rank holds per step. bf16 experts alone would be
# ~181 GB/GPU and FP8 ~91 GB/GPU.
# Two engines = 16 DP ranks = 16 sequences per rank at the start of each rollout (256 per step), where MTP
# helps most; more engines would lower that further but cost nodes.
NUM_INFERENCE_ENGINES=2
INFERENCE_ENGINE_TENSOR_PARALLEL_SIZE=1
INFERENCE_ENGINE_DATA_PARALLEL_SIZE=8
INFERENCE_ENGINE_EXPERT_PARALLEL_SIZE=8
LOGGER="${LOGGER:-wandb}"  # change to "console" to print to stdout (or set LOGGER env var)

CLIP_RATIO_LOW=0.2
CLIP_RATIO_HIGH=0.28
# use token mean loss reduction
LOSS_REDUCTION="token_mean"
# applies overlong filtering (but not soft overlong punishment)
APPLY_OVERLONG_FILTERING=true
# apply soft overlong punishment with custom trainer impl in main_dapo.py
OVERLONG_BUFFER_LEN=$((1024 * 4))
OVERLONG_BUFFER_PENALTY_FACTOR=1.0

# other DAPO parameters
USE_KL_LOSS=false
TEMPERATURE=1.0
# GLM-5.3's RL sampler: top-k 128 / top-p 0.99 trim the low-probability tail.
TOP_K=128
TOP_P=0.99
EVAL_TOP_P=0.7
CLIP_RATIO_C=10.0
MAX_PROMPT_LENGTH=$((1024 * 2))
MAX_RESPONSE_LENGTH=$((1024 * 8))
# vLLM max_model_len / trainer max_seq_len: 2K prompt + 8K response.
MAX_SEQ_LEN=$((1024 * 10))

# repro run parameters
TRAIN_BATCH_SIZE=16
MINI_BATCH_SIZE=16
N_SAMPLES_PER_PROMPT=16
EVAL_N_SAMPLES_PER_PROMPT=16
ENFORCE_EAGER=false # FULL_DECODE_ONLY CUDA graphs (capture sizes in the MTP block)
LR=1e-6

# megatron config -- GLM-5.3 is a 753B MoE (~725B in 256 routed experts per layer, ~28B dense).
# bf16 weights + bf16 grads are ~3 TB, so the trainer needs 48 GPUs: TP1 / CP16 / PP3 / EP16 puts ~15B expert
# and ~9B dense params on each GPU (~100 GB/GPU with grads, before activations). With 4 trainer nodes (EP8 / PP4)
# it is ~120 GB/GPU and model init runs out of memory. CP16 shards the DSA indexer, which TP would replicate;
# EP16 spans 2 nodes (DeepEP over EFA). PP3 at 26/28/24 layers: every stage starts on a layer that computes its
# own DSA top-k (index_topk_freq 4).
MEGATRON_TP=1
MEGATRON_PP=3
MEGATRON_CP=16
MEGATRON_EP=16
MEGATRON_ETP=1
HYBRID_LAYER_PATTERN="D-D-D-$(printf 'DE%.0s' {1..23})|$(printf 'DE%.0s' {1..28})|$(printf 'DE%.0s' {1..24})"
# One 10K sequence per CP rank group at a time is far below this; packs ~6 sequences per microbatch.
MAX_TOKENS_PER_MICROBATCH=65536

# optimizer offload -- ON. The optimizer step runs on the host with cpu_adamw_bf16 (bf16 moments, fp32 master as
# bf16 weight + int16 remainder: 6 B/param, ~750 GB per trainer node). Megatron's fp32 HybridDeviceOptimizer
# state (12 B/param plus fp32 grad copies) does not fit 6 nodes' host memory.
OPTIMIZER_OFFLOAD=true
OPTIMIZER_OFFLOAD_FRACTION=1.0

# TIS parameters
TIS_IMP_RATIO_CAP=2.0
TIS_TYPE=token


# Multi-Token Prediction (MTP) speculative decoding.
# GLM-5.3 ships 1 native MTP layer (`num_nextn_predict_layers: 1`, layer 78), but Megatron-Bridge's GLM5 bridge
# builds none unless mtp_num_layers=1 is set. NUM_SPECULATIVE_TOKENS is the vLLM *draft depth* only -- values > 1
# reuse the single layer autoregressively at draft time. Here k=4, with probabilistic drafting: each draft token
# is sampled from the head's distribution instead of its argmax. Verification is still vLLM's exact rejection
# sampler, so this is lossless; it raises acceptance at temperature 1. vLLM re-quantizes the drafter's experts to
# NVFP4 like the target's.
MTP_ENABLED="${MTP_ENABLED:-true}"
MTP_NUM_SPECULATIVE_TOKENS=4
MTP_DRAFT_SAMPLE_METHOD=probabilistic
# hard_label (CE on the generated tokens) is the draft loss that runs under CP, PP and the fused LM head.
# NOTE: the draft loss trains ONLY the MTP-head params -- trunk, re-embedding and output weight are detached.
MTP_LOSS_WEIGHT=0.1
if [ "$MTP_ENABLED" = "true" ]; then
  MTP_NUM_LAYERS=1
  SPECULATIVE_CONFIG='{"method": "mtp", "num_speculative_tokens": '"$MTP_NUM_SPECULATIVE_TOKENS"', "draft_sample_method": "'"$MTP_DRAFT_SAMPLE_METHOD"'"}'
  # MTP verifies k+1 tokens per sequence: capture decode graphs up to max_num_seqs * (k+1) tokens.
  CUDAGRAPH_CAPTURE_SIZES="[1, 2, 4, 8, 16, 32, 64, 128, 160, 320, 640]"
else
  MTP_NUM_LAYERS=0
  SPECULATIVE_CONFIG=null
  CUDAGRAPH_CAPTURE_SIZES="[1, 2, 4, 8, 16, 32, 64, 128]"
fi


# GLM-5.3 flags
# Trainer: DSA sparse attention on cuDNN/FlashMLA with the TileLang indexer top-k; reuse the forward top-k in the
# recompute (exact: the indexer is frozen), parallel TileLang logits writes, FP8 indexer scoring as in vLLM.
export SKYRL_DSA_INDEXER_BACKEND=tilelang SKYRL_DSA_TOPK_REUSE=1 SKYRL_DSA_INDEXER_PARALLEL_WRITE=1 SKYRL_DSA_INDEXER_FP8=1
export SKYRL_DSA_CP_LAYOUT_CACHE=1 SKYRL_DISABLE_FA4=1
# Full-recompute checkpoint inputs wait in pinned host memory; attention gets its own checkpoint inside MoE layers.
export SKYRL_OFFLOAD_CHECKPOINT_INPUTS=1 SKYRL_OFFLOAD_CHECKPOINT_INPUTS_PINNED=1 SKYRL_MOE_ATTENTION_RECOMPUTE=1
# Frozen weights (the DSA indexer) offload to file-backed pages in a writable node-local directory.
export SKYRL_FROZEN_OFFLOAD_DIR="${SKYRL_FROZEN_OFFLOAD_DIR:-/mnt/local_storage/skyrl_frozen_offload}"
# Inference: NVFP4 routed experts, re-quantized online from the bf16 weights on every sync (W4A4, per-token
# activation scales, FlashInfer TRT-LLM, round-to-nearest with 4/6 block scales); other layers stay bf16. NVFP4
# KV cache on FlashMLA sparse attention. NVLink one-sided expert all-to-all (each engine's EP group is one node).
# The engines boot from the real checkpoint, so the MTP drafter has its weights before the first sync.
export SKYRL_NVFP4_ROUNDING=nearest SKYRL_NVFP4_BLOCK_SCALE=four_over_six SKYRL_NVFP4_STOCHASTIC_STRENGTH=1.0
ENGINE_INIT_KWARGS='{
  "max_model_len": '"$MAX_SEQ_LEN"', "trust_remote_code": true, "logprobs_mode": "processed_logprobs", "load_format": "auto",
  "quantization": "nvfp4_per_token", "kv_cache_dtype": "nvfp4_ds_mla", "block_size": 64,
  "async_scheduling": false, "prefill_schedule_interval": 4,
  "all2all_backend": "flashinfer_nvlink_one_sided", "enable_eplb": false, "enable_ep_weight_filter": true,
  "kv_offloading_backend": "native", "kv_offloading_size": 96,
  "attention_config": {"backend": "FLASHMLA_SPARSE", "sparse_mla_force_mqa": true, "mla_prefill_backend": null,
                       "sparse_mla_prefill_workspace_factor": 2},
  "kernel_config": {"moe_backend": "flashinfer_trtllm",
                    "ir_op_priority": {"rms_norm": ["vllm_c"], "fused_add_rms_norm": ["vllm_c"]}},
  "compilation_config": {"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": '"$CUDAGRAPH_CAPTURE_SIZES"',
                         "cudagraph_num_of_warmups": 1, "pass_config": {"fuse_allreduce_rms": false}}}'
TRANSFORMER_CONFIG_KWARGS='{
  "recompute_granularity": "full", "recompute_method": "uniform", "recompute_num_layers": 1, "recompute_modules": null,
  "gradient_accumulation_fusion": false, "mlp_chunks_for_training": 1, "deallocate_pipeline_outputs": true,
  "moe_aux_loss_coeff": 0.0, "moe_permute_fusion": true, "moe_router_fusion": false, "moe_router_bias_update_rate": 0.0,
  "moe_router_dtype": "fp32", "moe_router_pre_softmax": true, "moe_router_topk_scaling_factor": 2.5,
  "moe_flex_dispatcher_backend": "deepep", "moe_deepep_num_sms": 20, "moe_shared_expert_overlap": false,
  "dsa_kernel_backend": "cudnn", "dsa_indexer_loss_coeff": 0.0, "dsa_indexer_topk_freq": 4, "dsa_indexer_skip_topk_offset": 3,
  "apply_rope_fusion": false, "bias_activation_fusion": true, "defer_fp32_logits": true, "mtp_num_layers": '"$MTP_NUM_LAYERS"',
  "pipeline_model_parallel_layout": null, "hybrid_layer_pattern": "'"$HYBRID_LAYER_PATTERN"'"}'
# sharded_rdt: the engines pull their own shards from the trainer (the drafter rides the target's sessions);
# tile pulls of tensors above 1.125 GiB (embedding / LM head) instead of failing planning.
export SKYRL_RDT_SPLIT_OVERSIZED_ROWS=1 SKYRL_RDT_MAX_PULL_GB=1.125
DISTRIBUTED_EXECUTOR_BACKEND="ray"
export OMP_NUM_THREADS=12 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
export SKYRL_WORKER_NCCL_TIMEOUT_IN_S=7200 SKYRL_WAIT_UNTIL_INFERENCE_SERVER_HEALTHY_TIMEOUT_S=10800
# vLLM servers above Ray's default worker-port range (10002-19999).
export SKYRL_VLLM_START_PORT="${SKYRL_VLLM_START_PORT:-20000}"
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=7200
# Settings SkyRL does not forward to the Ray workers. The script writes them to an env file next to itself and passes
# it with `uv run --env-file`, which Ray's uv hook replays on every worker, so keep the checkout at the same path on
# every node. Add your own NCCL / EFA settings, WANDB_API_KEY and HF_TOKEN with EXTRA_ENV_FILE=/path/to/.env.
WORKER_ENV_FILE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/.glm5.3_specdecode.env"
cat > "$WORKER_ENV_FILE" <<'EOF'
# vLLM engines: host KV through OffloadingConnector, DSA indexer prefill logits buffer, scale-out endpoints.
VLLM_USE_SIMPLE_KV_OFFLOAD=0
VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=2048
VLLM_ENABLE_SCALE_OUT_ENDPOINTS=1
# Trainer: DeepEP internode (EP16 spans 2 trainer nodes) over EFA through NVSHMEM / libfabric.
NVSHMEM_REMOTE_TRANSPORT=libfabric
NVSHMEM_LIBFABRIC_PROVIDER=efa
NVIDIA_GDRCOPY=enabled
# Trainer: TransformerEngine fused attention, the backend megatron-core's DSA attention runs on.
NVTE_FUSED_ATTN=1
EOF
if [ -n "${EXTRA_ENV_FILE:-}" ]; then cat "$EXTRA_ENV_FILE" >> "$WORKER_ENV_FILE"; fi

uv run --isolated --extra megatron --env-file "$WORKER_ENV_FILE" -m examples.train.algorithms.dapo.main_dapo \
  data.train_data="['$TRAIN_FILE']" \
  data.val_data="['$TEST_FILE']" \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.algorithm.policy_loss_type="dual_clip" \
  trainer.algorithm.overlong_buffer_len=$OVERLONG_BUFFER_LEN \
  trainer.algorithm.overlong_buffer_penalty_factor=$OVERLONG_BUFFER_PENALTY_FACTOR \
  trainer.algorithm.loss_reduction=$LOSS_REDUCTION \
  trainer.algorithm.grpo_norm_by_std=true \
  generator.inference_engine.enforce_eager=$ENFORCE_EAGER \
  generator.apply_overlong_filtering=$APPLY_OVERLONG_FILTERING \
  generator.sampling_params.temperature=$TEMPERATURE \
  generator.sampling_params.top_k=$TOP_K \
  generator.sampling_params.top_p=$TOP_P \
  generator.eval_sampling_params.top_p=$EVAL_TOP_P \
  generator.eval_sampling_params.temperature=$TEMPERATURE \
  generator.eval_sampling_params.max_generate_length=$MAX_RESPONSE_LENGTH \
  trainer.algorithm.use_kl_loss=$USE_KL_LOSS \
  trainer.algorithm.clip_ratio_c=$CLIP_RATIO_C \
  trainer.algorithm.max_seq_len=$MAX_SEQ_LEN \
  trainer.policy.model.path="$MODEL_NAME" \
  trainer.policy.language_model_only=true \
  trainer.placement.colocate_all=false \
  trainer.placement.colocate_policy_ref=false \
  trainer.placement.ref_num_nodes=0 \
  trainer.placement.ref_num_gpus_per_node=0 \
  trainer.placement.critic_num_nodes=0 \
  trainer.strategy=megatron \
  generator.inference_engine.distributed_executor_backend="$DISTRIBUTED_EXECUTOR_BACKEND" \
  trainer.placement.policy_num_nodes=$NUM_NODES \
  trainer.placement.policy_num_gpus_per_node=$NUM_GPUS_PER_NODE \
  generator.inference_engine.num_engines=$NUM_INFERENCE_ENGINES \
  generator.inference_engine.tensor_parallel_size=$INFERENCE_ENGINE_TENSOR_PARALLEL_SIZE \
  generator.inference_engine.data_parallel_size=$INFERENCE_ENGINE_DATA_PARALLEL_SIZE \
  generator.inference_engine.expert_parallel_size=$INFERENCE_ENGINE_EXPERT_PARALLEL_SIZE \
  trainer.policy.megatron_config.tensor_model_parallel_size=$MEGATRON_TP \
  trainer.policy.megatron_config.pipeline_model_parallel_size=$MEGATRON_PP \
  trainer.policy.megatron_config.context_parallel_size=$MEGATRON_CP \
  trainer.policy.megatron_config.expert_model_parallel_size=$MEGATRON_EP \
  trainer.policy.megatron_config.expert_tensor_parallel_size=$MEGATRON_ETP \
  trainer.policy.megatron_config.moe_token_dispatcher_type=flex \
  trainer.policy.megatron_config.moe_router_load_balancing_type=none \
  trainer.policy.megatron_config.moe_router_score_function=sigmoid \
  trainer.policy.megatron_config.moe_router_enable_expert_bias=true \
  trainer.policy.megatron_config.moe_grouped_gemm=true \
  trainer.policy.megatron_config.freeze_dsa_indexer=true \
  trainer.policy.megatron_config.empty_cuda_cache=false \
  trainer.policy.megatron_config.ddp_config.grad_reduce_in_fp32=false \
  trainer.policy.megatron_config.transformer_config_kwargs="$TRANSFORMER_CONFIG_KWARGS" \
  trainer.policy.megatron_config.mtp_num_layers=$MTP_NUM_LAYERS \
  "trainer.policy.megatron_config.model_config_kwargs={\"model_config\": {\"num_nextn_predict_layers\": $MTP_NUM_LAYERS}}" \
  trainer.policy.optimizer_config.name=cpu_adamw_bf16 \
  "trainer.policy.optimizer_config.kwargs={\"num_threads\": 20}" \
  trainer.policy.megatron_config.optimizer_config_kwargs.overlap_cpu_optimizer_d2h_h2d=$OPTIMIZER_OFFLOAD \
  trainer.policy.megatron_config.optimizer_config_kwargs.use_precision_aware_optimizer=$OPTIMIZER_OFFLOAD \
  trainer.policy.megatron_config.optimizer_config_kwargs.optimizer_cpu_offload=$OPTIMIZER_OFFLOAD \
  trainer.policy.megatron_config.optimizer_config_kwargs.optimizer_offload_fraction=$OPTIMIZER_OFFLOAD_FRACTION \
  trainer.policy.megatron_config.optimizer_config_kwargs.pin_cpu_grads=true \
  trainer.policy.megatron_config.optimizer_config_kwargs.pin_cpu_params=true \
  trainer.policy.megatron_config.optimizer_config_kwargs.exp_avg_dtype=bf16 \
  trainer.policy.megatron_config.optimizer_config_kwargs.exp_avg_sq_dtype=bf16 \
  trainer.policy.megatron_config.optimizer_config_kwargs.main_params_dtype=fp32 \
  trainer.policy.megatron_config.optimizer_config_kwargs.store_param_remainders=true \
  trainer.algorithm.off_policy_correction.tis_ratio_type=$TIS_TYPE \
  trainer.algorithm.off_policy_correction.token_tis_ratio_clip_high=$TIS_IMP_RATIO_CAP \
  trainer.fused_lm_head_logprob=true \
  trainer.fused_lm_head_logprob_backend=triton \
  trainer.flash_attn=false \
  trainer.remove_microbatch_padding=true \
  trainer.max_tokens_per_microbatch=$MAX_TOKENS_PER_MICROBATCH \
  trainer.epochs=10 \
  trainer.algorithm.eps_clip_low=$CLIP_RATIO_LOW \
  trainer.algorithm.eps_clip_high=$CLIP_RATIO_HIGH \
  trainer.eval_batch_size=1024 \
  trainer.eval_before_train=false \
  trainer.eval_interval=5 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=$TRAIN_BATCH_SIZE \
  trainer.policy_mini_batch_size=$MINI_BATCH_SIZE \
  trainer.micro_forward_batch_size_per_gpu=1 \
  trainer.micro_train_batch_size_per_gpu=1 \
  trainer.ckpt_interval=-1 \
  trainer.max_prompt_length=$MAX_PROMPT_LENGTH \
  generator.sampling_params.max_generate_length=$MAX_RESPONSE_LENGTH \
  trainer.policy.optimizer_config.lr=$LR \
  trainer.policy.optimizer_config.num_warmup_steps=5 \
  trainer.policy.optimizer_config.weight_decay=0.1 \
  trainer.policy.optimizer_config.max_grad_norm=1.0 \
  generator.inference_engine.backend=vllm \
  generator.inference_engine.run_engines_locally=true \
  generator.inference_engine.weight_sync_backend=sharded_rdt \
  generator.inference_engine.use_expandable_segments=false \
  generator.inference_engine.async_engine=true \
  generator.inference_engine.language_model_only=true \
  generator.inference_engine.served_model_name=GLM-5.3 \
  generator.inference_engine.max_num_seqs=128 \
  generator.inference_engine.max_num_batched_tokens=8192 \
  generator.inference_engine.nvfp4_online_quant.rounding=nearest \
  generator.inference_engine.nvfp4_online_quant.block_scale=four_over_six \
  "generator.inference_engine.router_init_kwargs={\"policy\": \"sticky_least_loaded\", \"request_timeout_secs\": 5700}" \
  generator.inference_engine.engine_init_kwargs="$ENGINE_INIT_KWARGS" \
  generator.inference_engine.speculative_config="$SPECULATIVE_CONFIG" \
  generator.batched=false \
  environment.env_class=aime \
  generator.n_samples_per_prompt=$N_SAMPLES_PER_PROMPT \
  generator.eval_n_samples_per_prompt=$EVAL_N_SAMPLES_PER_PROMPT \
  generator.inference_engine.gpu_memory_utilization=0.77 \
  trainer.mtp.enabled=$MTP_ENABLED \
  trainer.mtp.num_speculative_tokens=$MTP_NUM_SPECULATIVE_TOKENS \
  trainer.mtp.loss_weight=$MTP_LOSS_WEIGHT \
  trainer.logger="$LOGGER" \
  trainer.project_name="glm5p3_dapo" \
  trainer.run_name="sd_dapo_glm5p3_megatron_tp${MEGATRON_TP}_pp${MEGATRON_PP}_cp${MEGATRON_CP}_ep${MEGATRON_EP}_mtp${MTP_ENABLED}" \
  trainer.export_path="$HOME/exports/sd_dapo_glm5p3_megatron_tp${MEGATRON_TP}_pp${MEGATRON_PP}_cp${MEGATRON_CP}_ep${MEGATRON_EP}_mtp${MTP_ENABLED}" \
  trainer.hf_save_interval=300 \
  trainer.resume_mode=latest \
  trainer.max_ckpts_to_keep=3 \
  trainer.ckpt_path="$HOME/ckpts/sd_dapo_glm5p3_megatron_tp${MEGATRON_TP}_pp${MEGATRON_PP}_cp${MEGATRON_CP}_ep${MEGATRON_EP}_mtp${MTP_ENABLED}" \
  "$@"
