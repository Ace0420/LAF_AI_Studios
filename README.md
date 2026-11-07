# LAF AI Studios: ACE-Step 1.5 Serverless Worker

Replaces the ValyrianTech one-click image (`valyriantech/ace-step-1.5`).

## Why it uses less compute
| ValyrianTech pod | This worker |
|---|---|
| Always-on pod, billed every second it's up | Serverless, scales to zero, billed only while making a song |
| Runs the Gradio UI and the API server | Runs only the generator; your site is the UI |
| 32 GB GPU recommended | 2B turbo on a 24 GB card (16 GB likely works) |
| Large LM planner on vLLM, which reserves most of the VRAM | 0.6B planner on plain PyTorch (`LM_BACKEND=pt`) |
| Base model, 32-64 steps | Turbo model, 8 steps |

## Files
- `Dockerfile`: the container
- `handler.py`: takes a request, makes the song, returns it
- `test_input.json`: sample request for the RunPod test tab

## Deploy from your phone (you don't need Docker)
1. Repo: `Ace0420/LAF_AI_Studios` (Dockerfile at the root).
2. RunPod > Settings > Connections > GitHub > Connect. Allow access to that repo only.
3. RunPod > Serverless > New Endpoint > GitHub repo. Pick the repo, branch `main`, Dockerfile path `Dockerfile`.
4. GPU: a 24 GB tier. **Don't use MIG cards.** Max workers 1 for now. Active workers 0. Idle timeout 5 s.
5. **Network volume (recommended):** create one (about 30 GB) in the same region and attach it. On the first run the models download to it. After that, cold starts load from the volume instead.
6. Environment variables (all optional):
   - `DIT_MODEL`: `acestep-v15-turbo` (default) or `acestep-v15-xl-turbo` (needs 24 GB+)
   - `LM_MODEL`: `acestep-5Hz-lm-0.6B` (default)
   - `DEFAULT_THINKING`: `true` / `false`
   - `R2_ENDPOINT`, `R2_BUCKET`, `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`, `R2_PUBLIC_BASE`: when set, songs upload to Cloudflare R2 and the response returns a link. When unset, the response carries the audio as base64. That works for one short song, but RunPod caps response size.
7. Deploy. Watch the Builds tab. Then go to Requests, paste `test_input.json`, and run it.
8. Record `seconds` from the response for the first (cold) run and a second (warm) run. Those numbers replace the padded $0.10 per song.

## How the site talks to it
Browser (your HTML) → Cloudflare Worker (holds the RunPod API key, checks the Supabase user and quota) → RunPod `/run` → poll `/status/{id}` → R2 link back to the page.
Never put the RunPod key in the HTML.

## If the build fails
- **CPU-only torch in the log:** add a line before `uv pip install -e .` that installs torch from the CUDA index the ACE-Step README names.
- **Build timeout:** keep the model-bake line commented out and use the network volume.
- **Loader error on an XL model name:** set `ACESTEP_REF` to a newer tag. XL support arrived in the April 2026 code.
- Once a build works, pin `ACESTEP_REF` to that tag.
