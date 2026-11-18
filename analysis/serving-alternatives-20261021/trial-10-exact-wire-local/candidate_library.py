import os,sys,json,math,shutil,subprocess,time
from pathlib import Path
FINE_PREFIX_CACHE_PATCH_IDENTITY = {'artifact': 'arc3_vllm_main_e975732_arc3_overlay.tar.blob', 'applier': 'apply_vllm_main_e975732_arc3.py', 'overlay_sha256': 'e3a6fe0d9f010bc1fb66e53e7dc6fde43272824d5200da5a49b5c4456065e0cb', 'vllm_commit': 'e9757321527ca1ecd514c07c1418dd2c53da3d19'}
FINE_PREFIX_CACHE_REQUIRED_OVERLAYS = {'vllm/v1/core/kv_cache_coordinator.py': {'stock_sha256': '7fa4065d19021e77b3424d24e1b088c530ed70b90c5ed9bc52cf4b7581d5fc48', 'sha256': '7494e9f76d4a6b0bb469e80816e4052fb52171bc63b34799c3b9588092bc243e'}, 'vllm/v1/core/kv_cache_utils.py': {'stock_sha256': 'd359221ef91a94f7e570220f083979297c297227c1ad1b8ccdbec76cc24565b7', 'sha256': '816aea33eaa28a15b8d739be602a877aa0be1a8ec83853cdc92412dcdd137b4e'}, 'vllm/v1/core/single_type_kv_cache_manager.py': {'stock_sha256': '6ebd4ddb5210d500b52b300dfaf462e2e92f58a4ee0e15691a860549dc086d68', 'sha256': '97a0a4518c757cbe48e0d13d7a1890f48bc0db777004a3d66a8eb9ea73ba2405'}, 'vllm/v1/worker/gpu/model_states/mamba_hybrid.py': {'stock_sha256': '6ea89adcb39762b533f0c5b765a66e6f942cbde90208a5d13a758ef86a4ac6f1', 'sha256': '8cd9eac0da31add7685ba7f23dfc2dcd7d6d422470fa98281aa73f90d5a8408d'}, 'vllm/v1/worker/gpu_model_runner.py': {'stock_sha256': '95dbd301feae580913938cafe4151da7af2cbd5d3f37fa645fab62a8cd1b54ef', 'sha256': 'a2d3709963d463256b8b62ecb5264bce10e1b133d552ab10afdd31f68b42d1b2'}, 'vllm/v1/worker/mamba_utils.py': {'stock_sha256': '28c8a9bb07a56e3612e245ccc48cd7f29d85181dbe5e4cf8940c6bcd086df856', 'sha256': 'b5e30c8ab2343340f478c9c8ea94b9369649c74044cd42ea11b246dc84e11243'}}
FINE_PREFIX_CACHE_RUNTIME_IDENTITY = {'image': 'vllm/vllm-openai:nightly-e9757321527ca1ecd514c07c1418dd2c53da3d19', 'amd64_manifest_digest': 'sha256:e0eee5c5506bea9bfe350f7d99b07dc49e37d42647a128c2a57ff184551fba10', 'vllm_version': '0.29.1rc1.dev573+ge97573215'}
RUNTIME_MANIFEST_FILE_NAME = 'runtime-manifest.json'
SERVED_MODEL_NAME = 'flashnext'
VLLM_CUDAGRAPH_CAPTURE_STEP = 8
VLLM_CUDA_LAUNCH_BLOCKING = False
VLLM_DRAFT_VOCAB = 'configs/draft_vocab_32k.json'
VLLM_DTYPE = 'bfloat16'
VLLM_ENABLE_PREFIX_CACHING = True
VLLM_GDN_DECODE_KERNEL = 'triton'
VLLM_GPU_MEMORY_UTILIZATION = '0.96'
VLLM_HOST = '127.0.0.1'
VLLM_IMAGE_MAX_PIXELS = 409600
VLLM_INDEXER_KV_DTYPE = 'fp8'
VLLM_KV_CACHE_DTYPE = 'fp8_e4m3'
VLLM_KV_CACHE_MEMORY_BYTES = 0
VLLM_MAMBA_SSM_CACHE_DTYPE = 'bfloat16'
VLLM_MAX_MODEL_LEN = 139264
VLLM_MAX_NUM_BATCHED_TOKENS = 2048
VLLM_MAX_NUM_SEQS = 10
VLLM_MOE_BACKEND = ''
VLLM_MTP_TOKENS = 3
VLLM_PORT = 8001
VLLM_PREFIX_CACHE_RETENTION_INTERVAL = 0
VLLM_PREFIX_MATCH_UNIT = 128
VLLM_PROFILE_REQUESTED = False
VLLM_QUANTIZATION = 'compressed-tensors'
VLLM_SERVER_READY_TIMEOUT_SECONDS = 3600
VLLM_TENSOR_PARALLEL_SIZE = 1
def prepend_path_entries(existing: str, entries: list[Path]) -> str:
    joined = os.pathsep.join([str(entry) for entry in entries] + ([existing] if existing else []))
    return joined

def read_runtime_manifest() -> dict:
    manifest_path = RUNTIME_DATASET / RUNTIME_MANIFEST_FILE_NAME
    if not manifest_path.exists():
        raise FileNotFoundError(f'Missing image runtime manifest: {manifest_path}')
    return json.loads(manifest_path.read_text(encoding='utf-8'))

def build_runtime_python_paths(manifest: dict) -> list[Path]:
    """PYTHONPATH entries; the cutlass DSL .pth of the image is not processed on PYTHONPATH, so add its target."""
    dist_packages = IMAGE_RUNTIME_ROOT / manifest['dist_packages']
    return [dist_packages / 'nvidia_cutlass_dsl' / 'dsl_packages', dist_packages]

def resolve_runtime_cuda_home(manifest: dict) -> Path:
    cuda_home = IMAGE_RUNTIME_ROOT / Path(manifest['nvcc']).parent.parent
    return cuda_home

def build_runtime_library_paths(manifest: dict) -> list[Path]:
    dist_packages = IMAGE_RUNTIME_ROOT / manifest['dist_packages']
    nvidia_libs = sorted(dist_packages.glob('nvidia/*/lib')) + sorted(dist_packages.glob('nvidia/cu13/*/lib'))
    return [resolve_runtime_cuda_home(manifest) / 'targets' / 'x86_64-linux' / 'lib', dist_packages / 'torch' / 'lib', *nvidia_libs, Path('/usr/local/nvidia/lib64')]

def build_image_runtime_env(manifest: dict) -> dict[str, str]:
    """Build the vLLM server environment for the extracted image runtime.

    With the b12x MoE backend the b12x micro path is disabled, so every batch runs through
    the dynamic kernel. vLLM marks padding rows with expert id -1, and the b12x 1.3.0 micro
    kernel (up to 6 tokens at top-10) indexes weights with that id, which crashes with an
    illegal memory access. The dynamic kernel skips such rows.

    A configured draft vocabulary is read in place from the mounted bundle and handed to the
    e975732 rc3 runtime through VLLM_QWEN4_EXP_DRAFT_VOCAB; without one the variable is removed.

    VLLM_ARC3_GDN_RECOVERSSM=1 makes the GDN layers verify MTP drafts without per-draft state
    blocks and commit the accepted tokens after sampling, which frees 9 KV blocks per request.
    It is only a default, so an A/B launcher that sets the variable to 0 keeps the stock path.
    """
    env = os.environ.copy()
    cuda_home = resolve_runtime_cuda_home(manifest)
    cache_paths = {'VLLM_CACHE_ROOT': IMAGE_RUNTIME_CACHE_ROOT / 'vllm', 'TRITON_CACHE_DIR': IMAGE_RUNTIME_CACHE_ROOT / 'triton', 'FLASHINFER_WORKSPACE_BASE': IMAGE_RUNTIME_CACHE_ROOT / 'flashinfer', 'B12X_COMPILE_CACHE_DIR': IMAGE_RUNTIME_CACHE_ROOT / 'b12x' / 'compile', 'TMPDIR': IMAGE_RUNTIME_CACHE_ROOT / 'tmp'}
    for path in cache_paths.values():
        path.mkdir(parents=True, exist_ok=True)
    env.pop('PYTORCH_ALLOC_CONF', None)
    env.pop('VLLM_PLE_CPU_OFFLOAD', None)
    env.pop('VLLM_PLE_OFFLOAD_READY_TIMEOUT', None)
    env.pop('VLLM_PREFIX_CACHE_RETENTION_INTERVAL', None)
    env['PYTHONPATH'] = prepend_path_entries(env.get('PYTHONPATH', ''), build_runtime_python_paths(manifest))
    env['PATH'] = prepend_path_entries(env.get('PATH', ''), [cuda_home / 'bin', IMAGE_RUNTIME_ROOT / 'usr' / 'local' / 'bin'])
    env['LD_LIBRARY_PATH'] = prepend_path_entries(env.get('LD_LIBRARY_PATH', ''), [path for path in build_runtime_library_paths(manifest) if path.is_dir()])
    env.update({key: str(path) for key, path in cache_paths.items()})
    env.update({'USE_TF': '0', 'TRANSFORMERS_NO_TF': '1', 'TRANSFORMERS_NO_TORCHVISION': '1', 'VLLM_NO_USAGE_STATS': '1', 'CUDA_HOME': str(cuda_home), 'CUDACXX': str(cuda_home / 'bin' / 'nvcc'), 'TORCH_CUDA_ARCH_LIST': '12.0', 'VLLM_ENABLE_CUDA_COMPATIBILITY': '0', 'VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS': '1', 'VLLM_WORKER_MULTIPROC_METHOD': 'spawn', 'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:False', 'HF_HUB_OFFLINE': '1', 'HF_DATASETS_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'TOKENIZERS_PARALLELISM': 'false'})
    env.setdefault('VLLM_ARC3_GDN_RECOVERSSM', '1')
    if VLLM_GDN_DECODE_KERNEL:
        env['VLLM_GDN_DECODE_KERNEL'] = VLLM_GDN_DECODE_KERNEL
    if VLLM_MOE_BACKEND == 'b12x':
        env['B12X_MICRO_DYNAMIC_CUTOVER_PAIRS'] = '0'
    if VLLM_CUDA_LAUNCH_BLOCKING:
        env['CUDA_LAUNCH_BLOCKING'] = '1'
    env.pop('VLLM_QWEN4_EXP_DRAFT_VOCAB', None)
    if VLLM_DRAFT_VOCAB_FILE is not None:
        env['VLLM_QWEN4_EXP_DRAFT_VOCAB'] = str(VLLM_DRAFT_VOCAB_FILE)
    return env

def verify_layer_blob(layer: dict) -> Path:
    """Kaggle already guarantees dataset integrity, so only the recorded size is checked here."""
    blob = RUNTIME_DATASET / layer['file']
    if not blob.exists():
        raise FileNotFoundError(f'Missing runtime layer blob: {blob}')
    if blob.stat().st_size != int(layer['size']):
        raise RuntimeError(f"Runtime layer {layer['index']} does not match its manifest size: {blob.name}")
    return blob

def clear_opaque_whiteout_targets(layer: dict) -> None:
    for whiteout in layer.get('whiteouts', []):
        if whiteout['kind'] == 'opaque' and whiteout['target'] not in ('', '.'):
            shutil.rmtree(IMAGE_RUNTIME_ROOT / whiteout['target'], ignore_errors=True)

def remove_whiteout_markers_and_targets(layer: dict) -> None:
    for whiteout in layer.get('whiteouts', []):
        (IMAGE_RUNTIME_ROOT / whiteout['marker']).unlink(missing_ok=True)
        if whiteout['kind'] == 'file':
            target = IMAGE_RUNTIME_ROOT / whiteout['target']
            if target.is_dir() and (not target.is_symlink()):
                shutil.rmtree(target)
            else:
                target.unlink(missing_ok=True)

def extract_layer(blob: Path) -> None:
    """pigz, when the image ships it, moves reading, checksumming and writing off the single inflate thread
    (roughly 1.2-1.5x over gzip); the .pyc files stay in so the server does not recompile vLLM and torch."""
    decompressor = ['--use-compress-program=pigz'] if shutil.which('pigz') else ['-z']
    subprocess.run(['tar', '-xf', str(blob), '-C', str(IMAGE_RUNTIME_ROOT), '--no-same-owner', *decompressor], check=True)

def apply_runtime_patch(manifest: dict) -> None:
    patch = manifest.get('patch')
    if not patch:
        print('Image runtime ships no patch', flush=True)
        return
    overlay_files = patch.get('overlay_files')
    if not isinstance(overlay_files, list) or not overlay_files:
        raise RuntimeError('Runtime patch requires a non-empty overlay_files manifest')
    dist_packages = IMAGE_RUNTIME_ROOT / manifest['dist_packages']
    result = subprocess.run([sys.executable, str(RUNTIME_DATASET / patch['applier']), str(dist_packages)], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f'Runtime patch failed:\n{result.stderr.strip()}')
    print(f"Runtime patch {patch['artifact']}: {result.stdout.strip()}", flush=True)

def install_image_runtime() -> dict:
    """Unpack the image layers in index order, replay whiteouts and apply the runtime overlay."""
    manifest = read_runtime_manifest()
    validate_prefix_cache_runtime(manifest)
    shutil.rmtree(IMAGE_RUNTIME_ROOT, ignore_errors=True)
    IMAGE_RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
    layers = sorted(manifest['selected_layers'], key=lambda layer: int(layer['index']))
    started = time.monotonic()
    for layer in layers:
        blob = verify_layer_blob(layer)
        clear_opaque_whiteout_targets(layer)
        extract_layer(blob)
        remove_whiteout_markers_and_targets(layer)
        print(f"Extracted runtime layer {layer['index']} ({blob.name}) after {time.monotonic() - started:.0f}s", flush=True)
    apply_runtime_patch(manifest)
    print(f"Image runtime ready at {IMAGE_RUNTIME_ROOT}: vLLM {manifest['vllm_version']}, torch {manifest['torch_version']}", flush=True)
    return manifest

def validate_prefix_cache_runtime(manifest: dict) -> None:
    """Fail closed when fine-grained Qwen3.8 caching lacks its Mamba/MTP safety overlays."""
    if not VLLM_ENABLE_PREFIX_CACHING or VLLM_PREFIX_MATCH_UNIT <= 0:
        return
    identity_mismatches = sorted((key for key, expected_value in FINE_PREFIX_CACHE_RUNTIME_IDENTITY.items() if manifest.get(key) != expected_value))
    if identity_mismatches:
        raise RuntimeError('Fine-grained Qwen3.8 prefix caching requires the pinned vLLM runtime identity; mismatched fields: ' + ', '.join(identity_mismatches))
    patch = manifest.get('patch') or {}
    patch_identity_mismatches = sorted((key for key, expected_value in FINE_PREFIX_CACHE_PATCH_IDENTITY.items() if patch.get(key) != expected_value))
    if patch_identity_mismatches:
        raise RuntimeError('Fine-grained Qwen3.8 prefix caching requires the pinned runtime patch identity; mismatched fields: ' + ', '.join(patch_identity_mismatches))
    overlays = patch.get('overlay_files', [])
    overlay_targets = [overlay.get('target') for overlay in overlays]
    overlays_by_target = {overlay.get('target'): overlay for overlay in overlays}
    missing_targets = sorted(set(FINE_PREFIX_CACHE_REQUIRED_OVERLAYS) - set(overlay_targets))
    duplicate_targets = sorted((target for target in FINE_PREFIX_CACHE_REQUIRED_OVERLAYS if overlay_targets.count(target) > 1))
    mismatched_targets = sorted((target for target, expected_hashes in FINE_PREFIX_CACHE_REQUIRED_OVERLAYS.items() if target in overlays_by_target and any((overlays_by_target[target].get(hash_name) != expected_hash for hash_name, expected_hash in expected_hashes.items()))))
    problems = []
    if missing_targets:
        problems.append('missing targets: ' + ', '.join(missing_targets))
    if duplicate_targets:
        problems.append('duplicate targets: ' + ', '.join(duplicate_targets))
    if mismatched_targets:
        problems.append('mismatched overlay hashes: ' + ', '.join(mismatched_targets))
    if problems:
        raise RuntimeError('Fine-grained Qwen3.8 prefix caching requires the validated Mamba/MTP runtime overlays; ' + '; '.join(problems))

def build_vllm_server_command() -> list[str]:
    """The Qwen3.8-Flash-Next launch on the packed image runtime, as verified on Kaggle in v187.

    Prefix caching follows the config. With it enabled, the hybrid model uses
    mamba_cache_mode "align" and the runtime aligns Mamba checkpoints to the resolved
    attention block size. The patched runtime is required for correct scheduler and
    worker alignment; saved checkpoints share the KV pool with active requests.

    The quantization method, the sequence cap and the KV cache dtype come from the config, so all
    three reach this command. The method is passed explicitly rather than left to vLLM's detection,
    so a checkpoint whose config.json disagrees fails at startup instead of loading under a scheme
    nobody picked. The cap has to follow the KV pool rather than the worker count, because a cap
    above what the pool holds only queues the surplus in the scheduler. The experimental runtime
    adds FP8 KV support to QSA attention. Auto is passed as no flag at all; FP8 is enabled only
    when explicitly selected in the config. The MTP head is the model's own drafter, so no
    draft model rides along. The pinned KV cache size is optional; zero lets vLLM size the pool
    from gpu_memory_utilization after profiling the activation peak.

    A decode step carries one token per sequence plus its MTP drafts, and vLLM captures CUDA
    graphs at 1, 2 and 4 tokens and then at every multiple of 8 below 256. A capture size off that
    ladder is truncated down to the last size it captured, so a full batch whose step lands above
    it decodes with no graph at all. That is what nine sequences did in v216, a 36-token step
    against a largest graph of 32. The size is rounded up to the next multiple of 8 instead, and a
    step that falls between two sizes pads into the graph above it.

    With a draft vocabulary the MTP drafter picks its greedy proposals through
    use_local_argmax_reduction, which the e975732 rc3 runtime serves from the listed LM-head rows
    only. The target still verifies against the full head, so the list changes draft acceptance
    and never the generated text.

    embed_tokens in --cpu-offload-params keeps the 1.18 GiB input embedding table in pinned host
    memory, next to the PLE table, and the KV pool gets those bytes. The e975732 rc3 runtime reads
    only the requested rows over PCIe, so embeddings stay bit-identical, and the MTP drafter shares
    the table. The table sits outside the --cpu-offload-gb budget, which stays 0, so vLLM's generic
    weight offloader never starts.

    vLLM profiles the vision encoder on the largest image the preprocessor accepts, 4096x4096
    or 16384 tokens, and in v257 that one dummy set the whole 2.19 GiB activation peak, which the
    KV pool gives up. The agent only ever sends its grid image, so max_pixels caps the
    preprocessor at that image's area. The profile then encodes as many grid images as one
    max_num_batched_tokens chunk holds, the worst case a real step can meet, while a grid image is
    processed exactly as before. Video is disabled, otherwise its 12288-token dummy becomes the profiled item.
    """
    max_num_seqs = VLLM_MAX_NUM_SEQS
    decode_step_tokens = max_num_seqs * (1 + VLLM_MTP_TOKENS)
    cudagraph_capture_size = math.ceil(decode_step_tokens / VLLM_CUDAGRAPH_CAPTURE_STEP) * VLLM_CUDAGRAPH_CAPTURE_STEP
    cmd = [sys.executable, '-m', 'vllm.entrypoints.cli.main', 'serve', str(MODEL_PATH), '--served-model-name', SERVED_MODEL_NAME, '--host', VLLM_HOST, '--port', str(VLLM_PORT), '--load-format', 'safetensors', '--dtype', VLLM_DTYPE, '--tensor-parallel-size', str(VLLM_TENSOR_PARALLEL_SIZE), '--distributed-executor-backend', 'mp', '--max-model-len', str(VLLM_MAX_MODEL_LEN), '--max-num-seqs', str(max_num_seqs), '--max-num-batched-tokens', str(VLLM_MAX_NUM_BATCHED_TOKENS), '--async-scheduling', '--enable-chunked-prefill', '--max-cudagraph-capture-size', str(cudagraph_capture_size), '--enable-prefix-caching' if VLLM_ENABLE_PREFIX_CACHING else '--no-enable-prefix-caching', '--enable-auto-tool-choice', '--tool-call-parser', 'qwen3_coder', '--reasoning-parser', 'qwen3', '--generation-config', 'vllm', '--engram-config', json.dumps({'cpu_offload': True}), '--cpu-offload-params', 'embed_tokens', '--mm-processor-kwargs', json.dumps({'max_pixels': VLLM_IMAGE_MAX_PIXELS}), '--limit-mm-per-prompt', json.dumps({'video': 0}), '--default-chat-template-kwargs', '{"preserve_thinking": true, "reasoning_effort": "xhigh"}']
    if VLLM_QUANTIZATION:
        cmd += ['--quantization', VLLM_QUANTIZATION]
    if VLLM_MOE_BACKEND:
        cmd += ['--moe-backend', VLLM_MOE_BACKEND]
    if VLLM_ENABLE_PREFIX_CACHING and VLLM_PREFIX_MATCH_UNIT > 0:
        cmd += ['--prefix-match-unit', str(VLLM_PREFIX_MATCH_UNIT)]
    if VLLM_ENABLE_PREFIX_CACHING and VLLM_PREFIX_CACHE_RETENTION_INTERVAL >= 0:
        cmd += ['--prefix-cache-retention-interval', str(VLLM_PREFIX_CACHE_RETENTION_INTERVAL)]
    if VLLM_MTP_TOKENS > 0:
        speculative_config = {'method': 'mtp', 'num_speculative_tokens': VLLM_MTP_TOKENS}
        if VLLM_DRAFT_VOCAB:
            speculative_config['use_local_argmax_reduction'] = True
        cmd += ['--speculative-config', json.dumps(speculative_config)]
    if VLLM_GPU_MEMORY_UTILIZATION:
        cmd += ['--gpu-memory-utilization', VLLM_GPU_MEMORY_UTILIZATION]
    if VLLM_KV_CACHE_MEMORY_BYTES > 0:
        cmd += ['--kv-cache-memory-bytes', str(VLLM_KV_CACHE_MEMORY_BYTES)]
    if VLLM_KV_CACHE_DTYPE and VLLM_KV_CACHE_DTYPE != 'auto':
        cmd += ['--kv-cache-dtype', VLLM_KV_CACHE_DTYPE]
    if VLLM_INDEXER_KV_DTYPE:
        cmd += ['--attention-config', json.dumps({'indexer_kv_dtype': VLLM_INDEXER_KV_DTYPE})]
    if VLLM_MAMBA_SSM_CACHE_DTYPE and VLLM_MAMBA_SSM_CACHE_DTYPE != 'auto':
        cmd += ['--mamba-ssm-cache-dtype', VLLM_MAMBA_SSM_CACHE_DTYPE]
    chat_template = MODEL_PATH / 'chat_template.jinja'
    if chat_template.exists():
        cmd += ['--chat-template', str(chat_template)]
    return cmd
