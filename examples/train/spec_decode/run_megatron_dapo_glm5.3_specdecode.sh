set -x

# Sync GRPO for GLM-5.3 (753B glm_moe_dsa, full fine-tuning) on dapo-math-17k with Multi-Token Prediction (MTP)
# speculative decoding for faster rollout. 8 nodes x 8xB200: 6 trainer nodes (Megatron TP1/CP16/PP3/EP16,
# DeepEP, fused LM head, host AdamW) + 2 vLLM engine nodes (one DP8 x EP8 engine per node, NVFP4 routed experts,
# NVFP4 KV cache, NVLink one-sided expert all-to-all), weight sync over sharded_rdt.
# The MTP and baseline runs differ only in MTP_ENABLED.
#
#   bash examples/train/algorithms/dapo/prepare_dapo_data.sh
#   export WANDB_API_KEY=<key>
#   bash examples/train/spec_decode/run_megatron_dapo_glm5.3_specdecode.sh                    # MTP
#   MTP_ENABLED=false bash examples/train/spec_decode/run_megatron_dapo_glm5.3_specdecode.sh  # baseline

MODEL_PATH="${MODEL_PATH:-zai-org/GLM-5.3-BF16}"
DATA_DIR="${DATA_DIR:-$HOME/data/dapo}"
TRAIN_FILE="$DATA_DIR/dapo-math-17k-cleaned.parquet"
TEST_FILE="$DATA_DIR/aime-2024-cleaned.parquet"
LOGGER="${LOGGER:-wandb}"
MAX_TRAINING_STEPS="${MAX_TRAINING_STEPS:-20}"

# Multi-Token Prediction (MTP) speculative decoding.
# GLM-5.3 ships one MTP layer (layer 78); draft depths > 1 reuse it autoregressively. vLLM re-quantizes the
# drafter's experts to NVFP4 like the target's.
MTP_ENABLED="${MTP_ENABLED:-true}"
MTP_NUM_SPECULATIVE_TOKENS="${MTP_NUM_SPECULATIVE_TOKENS:-4}"
# Sample each draft token from the head's distribution instead of its argmax. Verification is still vLLM's
# exact rejection sampler, so this is lossless; it raises acceptance at temperature 1.
MTP_DRAFT_SAMPLE_METHOD="${MTP_DRAFT_SAMPLE_METHOD:-probabilistic}"
# hard_label (cross-entropy on the sampled tokens) is the draft loss that runs with CP, PP and the fused LM head.
# The head trains at the policy's lr, so over a short run it stays the pretrained drafter. A larger trainer.mtp.lr
# tracks a drifting policy, but without a warmup its first updates cut acceptance for a while.
MTP_LOSS_WEIGHT=0.1

MAX_RESPONSE_LENGTH=8192
MAX_SEQ_LEN=10240                 # prompt + response; also vLLM's max_model_len
TRAIN_BATCH_SIZE=16
N_SAMPLES_PER_PROMPT=16           # 256 sequences per step = 16 per engine DP rank

# Trainer: 6 nodes, CP16 and EP16; PP3 at 26/28/24 layers, every stage starting on a layer that computes its own
# DSA top-k (index_topk_freq 4). With 4 trainer nodes (EP8) the bf16 weights and grads alone are ~93 GB/GPU and
# model init runs out of memory.
TRAINER_NODES=6
MEGATRON_TP=1
MEGATRON_CP=16
MEGATRON_PP=3
MEGATRON_EP=16
HYBRID_LAYER_PATTERN="D-D-D-$(printf 'DE%.0s' {1..23})|$(printf 'DE%.0s' {1..28})|$(printf 'DE%.0s' {1..24})"

# Rollouts: 2 nodes, one vLLM engine per node with 8 data-parallel attention ranks sharing 8-way expert parallelism.
NUM_INFERENCE_ENGINES=2
if [ "$MTP_ENABLED" = "true" ]; then
  # MTP verifies k+1 tokens per sequence: capture graphs up to max_num_seqs * (k+1) tokens.
  CUDAGRAPH_CAPTURE_SIZES="[1, 2, 4, 8, 16, 32, 64, 128, 160, 320, 640]"; MAX_CUDAGRAPH_CAPTURE_SIZE=640
else
  CUDAGRAPH_CAPTURE_SIZES="[1, 2, 4, 8, 16, 32, 64, 128]"; MAX_CUDAGRAPH_CAPTURE_SIZE=128
fi
# NVFP4 routed experts, re-quantized online from the bf16 weights on every sync (W4A4, per-token activation
# scales, FlashInfer TRT-LLM); other layers stay bf16. NVFP4 KV cache on FlashMLA sparse attention. The engines
# boot from the real checkpoint, so the MTP drafter has its weights before the first sync.
ENGINE_INIT_KWARGS='{
  "max_model_len": '"$MAX_SEQ_LEN"', "trust_remote_code": true, "logprobs_mode": "processed_logprobs", "load_format": "auto",
  "quantization": "nvfp4_per_token", "kv_cache_dtype": "nvfp4_ds_mla", "block_size": 64,
  "async_scheduling": false, "prefill_schedule_interval": 4,
  "all2all_backend": "flashinfer_nvlink_one_sided", "enable_eplb": false, "enable_ep_weight_filter": true,
  "enable_auto_tool_choice": true, "tool_call_parser": "glm47", "reasoning_parser": "glm45",
  "kv_offloading_backend": "native", "kv_offloading_size": 96,
  "aux_output_config": {"max_bytes": 4294967296},
  "attention_config": {"backend": "FLASHMLA_SPARSE", "sparse_mla_force_mqa": true, "mla_prefill_backend": null,
                       "sparse_mla_prefill_workspace_factor": 2},
  "kernel_config": {"moe_backend": "flashinfer_trtllm",
                    "ir_op_priority": {"rms_norm": ["vllm_c"], "fused_add_rms_norm": ["vllm_c"]}},
  "compilation_config": {"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": '"$CUDAGRAPH_CAPTURE_SIZES"',
                         "max_cudagraph_capture_size": '"$MAX_CUDAGRAPH_CAPTURE_SIZE"', "cudagraph_num_of_warmups": 1, "pass_config": {"fuse_allreduce_rms": false}}}'

TRANSFORMER_CONFIG_KWARGS='{
  "recompute_granularity": "full", "recompute_method": "uniform", "recompute_num_layers": 1, "recompute_modules": null,
  "gradient_accumulation_fusion": false, "mlp_chunks_for_training": 1, "deallocate_pipeline_outputs": true,
  "moe_aux_loss_coeff": 0.0, "moe_permute_fusion": true, "moe_router_fusion": false, "moe_router_bias_update_rate": 0.0,
  "moe_router_dtype": "fp32", "moe_router_pre_softmax": true, "moe_router_topk_scaling_factor": 2.5,
  "moe_flex_dispatcher_backend": "deepep", "moe_deepep_num_sms": 20, "moe_shared_expert_overlap": false,
  "dsa_kernel_backend": "cudnn", "dsa_indexer_loss_coeff": 0.0, "dsa_indexer_topk_freq": 4, "dsa_indexer_skip_topk_offset": 3,
  "apply_rope_fusion": false, "bias_activation_fusion": true, "defer_fp32_logits": true,
  "mtp_num_layers": '"$([ "$MTP_ENABLED" = "true" ] && echo 1 || echo 0)"',
  "pipeline_model_parallel_layout": null, "hybrid_layer_pattern": "'"$HYBRID_LAYER_PATTERN"'"}'

MTP_OVERRIDES=(trainer.mtp.enabled=$MTP_ENABLED)
if [ "$MTP_ENABLED" = "true" ]; then
  MTP_OVERRIDES+=(
    trainer.mtp.num_speculative_tokens=$MTP_NUM_SPECULATIVE_TOKENS
    trainer.mtp.loss_type=hard_label
    trainer.mtp.loss_weight=$MTP_LOSS_WEIGHT
    trainer.policy.megatron_config.mtp_num_layers=1
    "trainer.policy.megatron_config.model_config_kwargs={\"model_config\": {\"num_nextn_predict_layers\": 1}}"
    "generator.inference_engine.speculative_config={\"method\": \"mtp\", \"num_speculative_tokens\": $MTP_NUM_SPECULATIVE_TOKENS, \"draft_sample_method\": \"$MTP_DRAFT_SAMPLE_METHOD\"}"
  )
else
  MTP_OVERRIDES+=(
    trainer.policy.megatron_config.mtp_num_layers=0
    "trainer.policy.megatron_config.model_config_kwargs={\"model_config\": {\"num_nextn_predict_layers\": 0}}"
  )
fi

# DSA kernels on the trainer: TileLang indexer top-k, cuDNN sparse attention; reuse the forward top-k in the
# recompute (exact: the indexer is frozen), parallel TileLang logits writes, FP8 indexer scoring as in vLLM.
export SKYRL_DSA_INDEXER_BACKEND=tilelang SKYRL_DSA_TOPK_REUSE=1 SKYRL_DSA_INDEXER_PARALLEL_WRITE=1 SKYRL_DSA_INDEXER_FP8=1
export SKYRL_DSA_CP_LAYOUT_CACHE=1 SKYRL_DISABLE_FA4=1 NVTE_FUSED_ATTN=1
# Full-recompute checkpoint inputs wait in pinned host memory; attention gets its own checkpoint inside MoE layers.
export SKYRL_OFFLOAD_CHECKPOINT_INPUTS=1 SKYRL_OFFLOAD_CHECKPOINT_INPUTS_PINNED=1 SKYRL_MOE_ATTENTION_RECOMPUTE=1
# Frozen weights (the DSA indexer) offload to file-backed pages in a writable node-local directory.
export SKYRL_FROZEN_OFFLOAD_DIR="${SKYRL_FROZEN_OFFLOAD_DIR:-/mnt/local_storage/skyrl_frozen_offload}"
# Online NVFP4 expert quantizer: round to nearest, 4/6 block scales.
export SKYRL_NVFP4_ROUNDING=nearest SKYRL_NVFP4_BLOCK_SCALE=four_over_six SKYRL_NVFP4_STOCHASTIC_STRENGTH=1.0
# sharded_rdt: tile pulls of tensors above 1.125 GiB (embedding / LM head) instead of failing planning.
export SKYRL_RDT_SPLIT_OVERSIZED_ROWS=1 SKYRL_RDT_MAX_PULL_GB=1.125
# Host KV through OffloadingConnector; DSA indexer prefill logits buffer.
export VLLM_USE_SIMPLE_KV_OFFLOAD=0 VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=2048
export VLLM_ALLOW_INSECURE_SERIALIZATION=1 VLLM_ENABLE_SCALE_OUT_ENDPOINTS=1 VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=7200
export OMP_NUM_THREADS=12 CUDA_DEVICE_MAX_CONNECTIONS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
# vLLM servers above Ray's default worker-port range (10002-19999).
export SKYRL_VLLM_START_PORT="${SKYRL_VLLM_START_PORT:-20000}"
export SKYRL_WORKER_NCCL_TIMEOUT_IN_S=7200 SKYRL_WAIT_UNTIL_INFERENCE_SERVER_HEALTHY_TIMEOUT_S=10800
# DeepEP internode (EP16 spans 2 trainer nodes) over EFA.
export NVSHMEM_REMOTE_TRANSPORT=libfabric NVSHMEM_LIBFABRIC_PROVIDER=efa NVIDIA_GDRCOPY=enabled

RUN_NAME="${RUN_NAME:-glm5p3_dapo_specdecode_mtp${MTP_ENABLED}_k${MTP_NUM_SPECULATIVE_TOKENS}}"

${UV_RUN:-uv run --isolated} --extra megatron -m skyrl.train.entrypoints.main_base \
  data.train_data="['$TRAIN_FILE']" \
  data.val_data="['$TEST_FILE']" \
  environment.env_class=aime \
  trainer.strategy=megatron \
  trainer.policy.model.path="$MODEL_PATH" \
  trainer.policy.language_model_only=true \
  trainer.placement.colocate_all=false \
  trainer.placement.policy_num_nodes=$TRAINER_NODES \
  trainer.placement.policy_num_gpus_per_node=8 \
  trainer.placement.colocate_policy_ref=false \
  trainer.placement.ref_num_nodes=0 \
  trainer.placement.ref_num_gpus_per_node=0 \
  trainer.placement.critic_num_nodes=0 \
  trainer.algorithm.advantage_estimator=grpo \
  trainer.algorithm.policy_loss_type=rollout_is \
  trainer.algorithm.loss_reduction=token_mean \
  trainer.algorithm.eps_clip_low=0.2 \
  trainer.algorithm.eps_clip_high=0.28 \
  trainer.algorithm.use_kl_loss=false \
  trainer.algorithm.entropy_loss_coef=0.0 \
  trainer.algorithm.grpo_norm_by_std=true \
  trainer.algorithm.zero_variance_filter=true \
  trainer.algorithm.max_seq_len=$MAX_SEQ_LEN \
  trainer.algorithm.enable_sample_support_replay=true \
  trainer.algorithm.enable_score_centering=true \
  trainer.algorithm.off_policy_correction.tis_ratio_type=null \
  trainer.algorithm.temperature=null \
  trainer.policy.optimizer_config.name=cpu_adamw_bf16 \
  "trainer.policy.optimizer_config.kwargs={\"num_threads\": 20}" \
  trainer.policy.optimizer_config.lr=1e-6 \
  trainer.policy.optimizer_config.min_lr=1e-6 \
  trainer.policy.optimizer_config.scheduler=constant_with_warmup \
  trainer.policy.optimizer_config.num_warmup_steps=0 \
  trainer.policy.optimizer_config.weight_decay=0.0 \
  trainer.policy.optimizer_config.max_grad_norm=1.0 \
  trainer.policy.optimizer_config.offload_after_step=true \
  trainer.policy.megatron_config.tensor_model_parallel_size=$MEGATRON_TP \
  trainer.policy.megatron_config.context_parallel_size=$MEGATRON_CP \
  trainer.policy.megatron_config.pipeline_model_parallel_size=$MEGATRON_PP \
  trainer.policy.megatron_config.expert_model_parallel_size=$MEGATRON_EP \
  trainer.policy.megatron_config.expert_tensor_parallel_size=1 \
  trainer.policy.megatron_config.moe_token_dispatcher_type=flex \
  trainer.policy.megatron_config.moe_router_load_balancing_type=none \
  trainer.policy.megatron_config.moe_router_score_function=sigmoid \
  trainer.policy.megatron_config.moe_router_enable_expert_bias=true \
  trainer.policy.megatron_config.moe_enable_routing_replay=true \
  trainer.policy.megatron_config.moe_fused_routing_replay=true \
  trainer.policy.megatron_config.moe_grouped_gemm=true \
  trainer.policy.megatron_config.freeze_moe_router=false \
  trainer.policy.megatron_config.freeze_dsa_indexer=true \
  trainer.policy.megatron_config.empty_cuda_cache=false \
  trainer.policy.megatron_config.ddp_config.average_in_collective=true \
  trainer.policy.megatron_config.ddp_config.overlap_grad_reduce=false \
  trainer.policy.megatron_config.ddp_config.overlap_param_gather=false \
  trainer.policy.megatron_config.ddp_config.grad_reduce_in_fp32=false \
  "trainer.policy.megatron_config.optimizer_config_kwargs={\"optimizer_cpu_offload\": true, \"optimizer_offload_fraction\": 1.0, \"overlap_cpu_optimizer_d2h_h2d\": true, \"pin_cpu_grads\": true, \"pin_cpu_params\": true, \"use_precision_aware_optimizer\": true, \"exp_avg_dtype\": \"bf16\", \"exp_avg_sq_dtype\": \"bf16\", \"main_params_dtype\": \"fp32\", \"store_param_remainders\": true}" \
  trainer.policy.megatron_config.transformer_config_kwargs="$TRANSFORMER_CONFIG_KWARGS" \
  trainer.fused_lm_head_logprob=true \
  trainer.fused_lm_head_logprob_backend=triton \
  trainer.flash_attn=false \
  trainer.remove_microbatch_padding=true \
  trainer.max_tokens_per_microbatch=65536 \
  trainer.micro_train_batch_size_per_gpu=1 \
  trainer.micro_forward_batch_size_per_gpu=1 \
  trainer.train_batch_size=$TRAIN_BATCH_SIZE \
  trainer.policy_mini_batch_size=$TRAIN_BATCH_SIZE \
  trainer.update_epochs_per_batch=1 \
  trainer.max_prompt_length=$MAX_SEQ_LEN \
  trainer.epochs=100 \
  trainer.max_training_steps=$MAX_TRAINING_STEPS \
  trainer.eval_before_train=false \
  trainer.eval_interval=-1 \
  trainer.ckpt_interval=0 \
  trainer.hf_save_interval=0 \
  trainer.resume_mode=null \
  generator.inference_engine.backend=vllm \
  generator.inference_engine.run_engines_locally=true \
  generator.inference_engine.num_engines=$NUM_INFERENCE_ENGINES \
  generator.inference_engine.tensor_parallel_size=1 \
  generator.inference_engine.data_parallel_size=8 \
  generator.inference_engine.expert_parallel_size=8 \
  generator.inference_engine.distributed_executor_backend=ray \
  generator.inference_engine.weight_sync_backend=sharded_rdt \
  generator.inference_engine.use_expandable_segments=false \
  generator.inference_engine.gpu_memory_utilization=0.77 \
  generator.inference_engine.max_num_batched_tokens=8192 \
  generator.inference_engine.max_num_seqs=128 \
  generator.inference_engine.enable_prefix_caching=true \
  generator.inference_engine.enable_chunked_prefill=true \
  generator.inference_engine.enforce_eager=false \
  generator.inference_engine.language_model_only=true \
  generator.inference_engine.served_model_name=GLM-5.3 \
  generator.inference_engine.enable_return_routed_experts=true \
  generator.inference_engine.enable_return_sample_support_set=true \
  generator.inference_engine.nvfp4_online_quant.rounding=nearest \
  generator.inference_engine.nvfp4_online_quant.block_scale=four_over_six \
  "generator.inference_engine.router_init_kwargs={\"policy\": \"sticky_least_loaded\", \"request_timeout_secs\": 5700}" \
  generator.inference_engine.engine_init_kwargs="$ENGINE_INIT_KWARGS" \
  generator.batched=false \
  generator.n_samples_per_prompt=$N_SAMPLES_PER_PROMPT \
  generator.max_input_length=$MAX_SEQ_LEN \
  generator.sampling_params.max_generate_length=$MAX_RESPONSE_LENGTH \
  generator.sampling_params.temperature=1.0 \
  generator.sampling_params.top_k=128 \
  generator.sampling_params.top_p=0.99 \
  generator.sampling_params.logprobs=1 \
  generator.eval_sampling_params.max_generate_length=$MAX_RESPONSE_LENGTH \
  "${MTP_OVERRIDES[@]}" \
  trainer.logger="$LOGGER" \
  trainer.project_name="glm5p3_dapo_specdecode" \
  trainer.run_name="$RUN_NAME" \
  trainer.ckpt_path="$HOME/ckpts/$RUN_NAME" \
  "$@"
