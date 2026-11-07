"""
LAF AI Studios - ACE-Step 1.5 serverless worker (RunPod).

No Gradio UI, no API server, no always-on pod. RunPod starts this when a
request arrives and the endpoint scales back to zero when idle.

Request  (POST /run or /runsync, body {"input": {...}}):
  caption        str   style/genre/vocal description (required)
  lyrics         str   lyrics, or "[Instrumental]"           (default "")
  duration       float seconds, 10-600; <=0 lets model pick   (default 120)
  bpm            int   optional
  keyscale       str   optional, e.g. "A minor"
  seed           int   -1 = random                            (default -1)
  batch_size     int   1-4                                    (default 1)
  steps          int   inference steps; turbo 8, base 32-64   (default per model)
  thinking       bool  use the small LM planner               (default env)
  rewrite_caption bool LM may rewrite caption ("CaptionRewrite") (default False)
  format         str   "mp3" | "wav" | "flac"                 (default "mp3")

Response:
  {"songs": [{"seed": 123, "format": "mp3", "url": "..."} or {"audio_b64": "..."}],
   "seconds": 41.2, "model": "acestep-v15-turbo"}
"""

import base64
import os
import shutil
import time
import uuid

import runpod

# ---------------------------------------------------------------- settings
PROJECT_ROOT = os.environ.get("ACESTEP_PROJECT_ROOT", "/opt/ace-step")
DIT_MODEL = os.environ.get("DIT_MODEL", "acestep-v15-turbo")        # or acestep-v15-xl-turbo
LM_MODEL = os.environ.get("LM_MODEL", "acestep-5Hz-lm-0.6B")        # smallest planner
LM_BACKEND = os.environ.get("LM_BACKEND", "pt")                     # "pt" avoids vLLM grabbing most of the VRAM
DEFAULT_THINKING = os.environ.get("DEFAULT_THINKING", "true").lower() == "true"
MAX_BATCH = int(os.environ.get("MAX_BATCH", "4"))
MAX_DURATION = float(os.environ.get("MAX_DURATION", "300"))

# Optional Cloudflare R2 upload (keeps responses small). If unset, audio is returned as base64.
R2_ENDPOINT = os.environ.get("R2_ENDPOINT")          # https://<account>.r2.cloudflarestorage.com
R2_BUCKET = os.environ.get("R2_BUCKET")
R2_KEY_ID = os.environ.get("R2_ACCESS_KEY_ID")
R2_SECRET = os.environ.get("R2_SECRET_ACCESS_KEY")
R2_PUBLIC_BASE = os.environ.get("R2_PUBLIC_BASE")    # e.g. https://files.lafaistudios.com

# Model weights: use a network volume if one is attached, so cold starts don't re-download.
VOLUME = "/runpod-volume"
CKPT_LINK = os.path.join(PROJECT_ROOT, "checkpoints")
OUT_DIR = "/tmp/laf_out"

# Repos for models that are NOT in the main unified repo (ACE-Step/Ace-Step1.5).
SEPARATE_REPOS = {
    "acestep-5Hz-lm-0.6B": "ACE-Step/acestep-5Hz-lm-0.6B",
    "acestep-5Hz-lm-4B": "ACE-Step/acestep-5Hz-lm-4B",
    "acestep-v15-base": "ACE-Step/acestep-v15-base",
    "acestep-v15-sft": "ACE-Step/acestep-v15-sft",
    "acestep-v15-xl-base": "ACE-Step/acestep-v15-xl-base",
    "acestep-v15-xl-sft": "ACE-Step/acestep-v15-xl-sft",
    "acestep-v15-xl-turbo": "ACE-Step/acestep-v15-xl-turbo",
}


def _prepare_checkpoints():
    """Point PROJECT_ROOT/checkpoints at the network volume when one exists."""
    if os.path.isdir(VOLUME):
        target = os.path.join(VOLUME, "acestep-checkpoints")
        os.makedirs(target, exist_ok=True)
        if os.path.islink(CKPT_LINK):
            return target
        if os.path.isdir(CKPT_LINK):
            # Move anything baked into the image onto the volume once.
            for name in os.listdir(CKPT_LINK):
                src, dst = os.path.join(CKPT_LINK, name), os.path.join(target, name)
                if not os.path.exists(dst):
                    shutil.move(src, dst)
            shutil.rmtree(CKPT_LINK, ignore_errors=True)
        os.symlink(target, CKPT_LINK)
        return target
    os.makedirs(CKPT_LINK, exist_ok=True)
    return CKPT_LINK


def _ensure_model(ckpt_dir, name):
    """Download a separately hosted model if it isn't on disk yet."""
    if name not in SEPARATE_REPOS:
        return  # the main repo is fetched by ACE-Step itself
    local = os.path.join(ckpt_dir, name)
    if os.path.isdir(local) and os.listdir(local):
        return
    from huggingface_hub import snapshot_download
    print(f"[LAF] downloading {name} ...", flush=True)
    snapshot_download(repo_id=SEPARATE_REPOS[name], local_dir=local)


# ---------------------------------------------------------------- load once per worker
t0 = time.time()
CKPT_DIR = _prepare_checkpoints()

from acestep.handler import AceStepHandler            # noqa: E402
from acestep.llm_inference import LLMHandler          # noqa: E402
from acestep.inference import GenerationParams, GenerationConfig, generate_music  # noqa: E402

try:  # make sure the main unified repo (vae, text encoder, turbo) is present
    from acestep.model_downloader import ensure_main_model
    ensure_main_model()
except Exception as e:  # older/newer code may differ; initialize_service also downloads
    print(f"[LAF] ensure_main_model skipped: {e}", flush=True)

_ensure_model(CKPT_DIR, DIT_MODEL)
_ensure_model(CKPT_DIR, LM_MODEL)

DIT = AceStepHandler()
status, ok = DIT.initialize_service(project_root=PROJECT_ROOT, config_path=DIT_MODEL, device="cuda")
if not ok:
    raise RuntimeError(f"DiT failed to load: {status}")

LLM = LLMHandler()
LLM.initialize(checkpoint_dir=CKPT_DIR, lm_model_path=LM_MODEL, backend=LM_BACKEND, device="cuda")

IS_TURBO = "turbo" in DIT_MODEL
print(f"[LAF] models ready in {time.time() - t0:.1f}s ({DIT_MODEL} + {LM_MODEL})", flush=True)


# ---------------------------------------------------------------- helpers
def _r2_client():
    if not (R2_ENDPOINT and R2_BUCKET and R2_KEY_ID and R2_SECRET):
        return None
    import boto3
    return boto3.client("s3", endpoint_url=R2_ENDPOINT,
                        aws_access_key_id=R2_KEY_ID, aws_secret_access_key=R2_SECRET,
                        region_name="auto")


R2 = _r2_client()
MIME = {"mp3": "audio/mpeg", "wav": "audio/wav", "flac": "audio/flac"}


def _deliver(path, fmt):
    if R2:
        key = f"songs/{uuid.uuid4().hex}.{fmt}"
        R2.upload_file(path, R2_BUCKET, key, ExtraArgs={"ContentType": MIME[fmt]})
        return {"key": key, "url": f"{R2_PUBLIC_BASE.rstrip('/')}/{key}" if R2_PUBLIC_BASE else None}
    with open(path, "rb") as f:
        return {"audio_b64": base64.b64encode(f.read()).decode()}


# ---------------------------------------------------------------- handler
def handler(job):
    inp = job.get("input") or {}
    caption = (inp.get("caption") or "").strip()
    if not caption:
        return {"error": "caption is required"}

    fmt = inp.get("format", "mp3")
    if fmt not in MIME:
        return {"error": "format must be mp3, wav or flac"}

    duration = float(inp.get("duration", 120))
    duration = min(duration, MAX_DURATION) if duration > 0 else -1.0
    batch = max(1, min(int(inp.get("batch_size", 1)), MAX_BATCH))
    thinking = bool(inp.get("thinking", DEFAULT_THINKING))
    rewrite = bool(inp.get("rewrite_caption", False))
    steps = int(inp.get("steps", 8 if IS_TURBO else 48))
    seed = int(inp.get("seed", -1))

    params = GenerationParams(
        caption=caption,
        lyrics=inp.get("lyrics", ""),
        duration=duration,
        bpm=inp.get("bpm"),
        keyscale=inp.get("keyscale", ""),
        seed=seed,
        inference_steps=steps,
        thinking=thinking,
        use_cot_caption=rewrite,   # off by default: keep the customer's style text as written
    )
    config = GenerationConfig(
        batch_size=batch,
        audio_format=fmt,
        use_random_seed=(seed == -1),
        seeds=None if seed == -1 else seed,
    )

    job_dir = os.path.join(OUT_DIR, job.get("id", uuid.uuid4().hex))
    os.makedirs(job_dir, exist_ok=True)
    start = time.time()
    try:
        result = generate_music(DIT, LLM, params, config, save_dir=job_dir)
        if not result.success:
            return {"error": result.error or "generation failed"}
        songs = []
        for a in result.audios:
            item = {"seed": (a.get("params") or {}).get("seed"), "format": fmt}
            item.update(_deliver(a["path"], fmt))
            songs.append(item)
        return {"songs": songs, "seconds": round(time.time() - start, 2), "model": DIT_MODEL}
    finally:
        shutil.rmtree(job_dir, ignore_errors=True)


runpod.serverless.start({"handler": handler})
