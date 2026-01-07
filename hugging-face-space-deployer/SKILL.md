---
name: hugging-face-space-deployer
description: Create, configure, and deploy Hugging Face Spaces for ML models. Auto-detects model type, chooses optimal deployment strategy, monitors build status, and provides remediation for common errors. Supports 15+ pipeline types including chat, image generation, ASR, TTS, VQA, and more.
---

# Hugging Face Space Deployer

A comprehensive skill for deploying ML models to interactive Hugging Face Spaces with minimal configuration.

## Overview

This skill enables you to:
- Deploy any HuggingFace model to an interactive Space
- Auto-detect model type (LoRA adapter vs full model)
- Auto-select deployment strategy (Inference API vs ZeroGPU)
- Monitor build status and troubleshoot errors
- Support 15+ pipeline types with optimized templates

## When to Use This Skill

Use this skill when:
- A user wants to create a demo/Space for a HuggingFace model
- A user has fine-tuned a model and wants to showcase it
- A user wants to deploy a LoRA adapter with a base model
- A user needs help troubleshooting a failing Space
- A user wants to understand hardware options for deployment

## Key Directives

**CRITICAL: Follow this workflow for every deployment:**

1. **Run pre-flight checks** - Validate token, subscription, model access
2. **Analyze the model** - Detect LoRA vs full model, check Inference API
3. **Select strategy automatically** - Only ask user when truly ambiguous
4. **Deploy with smart defaults** - Hardware, requirements, templates
5. **Monitor and remediate** - Check build status, suggest fixes

**ONLY prompt the user when:**
- LoRA adapter detected but `base_model` cannot be found in adapter_config.json
- ZeroGPU recommended but user doesn't have PRO subscription

## Prerequisites Checklist

### Account Requirements
- [ ] HuggingFace account
- [ ] HF token with write permissions (`huggingface-cli login`)
- [ ] PRO subscription (only required for ZeroGPU hosting)

### Model Requirements
- [ ] Model is accessible (not gated without access, not private)
- [ ] Model has `pipeline_tag` set in model card (for auto-detection)
- [ ] For LoRA adapters: base model must be identified

## Quick Start Examples

### Example 1: Deploy Popular Model with Inference API
```bash
# Uses Inference API automatically (free, cpu-basic)
python scripts/deploy_model.py meta-llama/Llama-3.1-8B-Instruct --type chat
```

### Example 2: Deploy Personal Model with ZeroGPU
```bash
# Uses ZeroGPU automatically (free with PRO, requires GPU)
python scripts/deploy_model.py GhostScientist/my-finetuned-model --type chat
```

### Example 3: Deploy LoRA Adapter
```bash
# Auto-detects base model from adapter_config.json
python scripts/deploy_model.py GhostScientist/my-lora-adapter --type chat

# Or specify base model explicitly
python scripts/deploy_model.py GhostScientist/my-lora-adapter --type chat --base-model Qwen/Qwen2.5-Coder-1.5B-Instruct
```

### Example 4: Dry Run (Preview Without Deploying)
```bash
python scripts/deploy_model.py meta-llama/Llama-3.1-8B-Instruct --type chat --dry-run
```

## Deployment Decision Tree

```
START: Analyze Model
│
├── Check files in model repo
│   │
│   ├── Has adapter_config.json + adapter_model.safetensors?
│   │   └── YES → It's a LoRA/PEFT adapter
│   │       │
│   │       ├── base_model_name_or_path found?
│   │       │   ├── YES → Use base model + adapter with PEFT
│   │       │   └── NO → **PROMPT USER for base model**
│   │       │
│   │       └── Strategy: ZeroGPU LoRA Template
│   │
│   ├── Has model.safetensors or pytorch_model.bin?
│   │   └── YES → It's a full model
│   │       │
│   │       ├── Is from known Inference API provider?
│   │       │   │  (meta-llama, mistralai, google, stabilityai, Qwen, etc.)
│   │       │   ├── YES → Strategy: Inference API (free, cpu-basic)
│   │       │   └── NO → Strategy: ZeroGPU Full Model
│   │       │
│   │       └── Check ZeroGPU eligibility
│   │           ├── User has PRO? → Use ZeroGPU (free)
│   │           └── User is Free? → **PROMPT: Upgrade to PRO or use paid GPU**
│   │
│   └── Neither found?
│       └── ERROR: Model appears incomplete
│
└── Deploy with selected strategy
```

## Supported Pipeline Types

| Pipeline Tag | Template | Strategy | Hardware |
|-------------|----------|----------|----------|
| `text-generation` | Chat/Text Gen | Inference API or ZeroGPU | cpu-basic / zero-a10g |
| `text2text-generation` | Chat | Inference API or ZeroGPU | cpu-basic / zero-a10g |
| `conversational` | Chat | Inference API or ZeroGPU | cpu-basic / zero-a10g |
| `text-to-image` | Image Gen | Inference API | cpu-basic |
| `image-to-image` | Img2Img | Inference API | cpu-basic |
| `image-classification` | Image Class | Local transformers | cpu-upgrade |
| `object-detection` | Detection | Local transformers | cpu-upgrade |
| `image-segmentation` | Segmentation | Local transformers | cpu-upgrade |
| `automatic-speech-recognition` | ASR | Inference API or ZeroGPU | cpu-basic / zero-a10g |
| `text-to-speech` | TTS | Inference API | cpu-basic |
| `audio-classification` | Audio Class | Local transformers | cpu-upgrade |
| `visual-question-answering` | VQA | ZeroGPU | zero-a10g |
| `zero-shot-classification` | Zero-Shot | Local transformers | cpu-upgrade |
| `depth-estimation` | Depth | Local transformers | cpu-upgrade |
| `feature-extraction` | Embedding | Local sentence-transformers | cpu-upgrade |

## Hardware Selection Guide

### Hardware Tiers (2025 Pricing)

| Hardware | VRAM | Cost | Best For |
|----------|------|------|----------|
| `cpu-basic` | - | Free | Inference API apps, small models |
| `cpu-upgrade` | - | $0.03/hr | Classification, embeddings |
| `zero-a10g` | 70GB (H200) | Free* | Most GPU models <7B |
| `t4-small` | 16GB | $0.40/hr | Entry-level GPU |
| `l4` | 24GB | $0.80/hr | 3-7B models, great value |
| `l40s` | 48GB | $1.80/hr | 7-14B models |
| `a10g-small` | 24GB | $1.00/hr | Alternative to L4 |
| `a100-large` | 80GB | $2.50/hr | 14-30B+ models |

*ZeroGPU requires PRO subscription to host, but is free to use as visitor.

### ZeroGPU Requirements

**CRITICAL: ZeroGPU Hosting Requirements**

| Account Type | Can HOST ZeroGPU? | Can USE ZeroGPU? | Daily Quota |
|--------------|-------------------|------------------|-------------|
| Free | No | Yes | 3.5 min |
| PRO | Yes | Yes | 25 min |
| Team/Enterprise | Yes | Yes | 25-45 min |

**ZeroGPU Technical Specs:**
- GPU: NVIDIA H200 slice
- VRAM: 70GB available per workload
- SDK: Gradio only (no Streamlit/Docker)
- Python: 3.10.13
- PyTorch: 2.1.0 - 2.8.0
- No `torch.compile` support (use ahead-of-time compilation)

### Hardware Recommendation by Model Size

| Model Size | Recommended Hardware |
|------------|---------------------|
| < 0.5B params | cpu-upgrade |
| 0.5B - 3B params | zero-a10g (PRO) or t4-small |
| 3B - 7B params | zero-a10g (PRO) or l4 |
| 7B - 14B params | l40s |
| 14B - 30B params | a100-large |
| > 30B params | a100-large + quantization |

## Pre-Deployment Checklist

### 1. Check Model Type

**Use the Hub API or hf-skills MCP tool:**
```
hf-skills - Hub Repo Details (repo_ids: ["username/model"], repo_type: "model")
```

**Look for these indicators:**

| Files Present | Model Type | Template |
|---------------|------------|----------|
| `model.safetensors` | Full model | ZeroGPU Full or Inference API |
| `adapter_config.json` + `adapter_model.safetensors` | LoRA adapter | ZeroGPU LoRA |
| Only config files | Incomplete | Ask user to verify |

### 2. Check Inference API Availability

**Models with Inference API:**
- From known providers: `meta-llama`, `mistralai`, `google`, `stabilityai`, `Qwen`, `deepseek-ai`, etc.
- Have inference widget on model page
- High download count with standard architecture

**Models WITHOUT Inference API:**
- Personal namespace (e.g., `username/my-model`)
- LoRA/PEFT adapters (never have direct API)
- Missing `pipeline_tag` metadata

### 3. Check ZeroGPU Eligibility

Before recommending ZeroGPU:
```bash
python scripts/preflight.py check-subscription
```

### 4. Determine Hardware

```bash
python scripts/preflight.py estimate-size username/model-id
```

## Deployment Templates

### Template 1: Inference API (Recommended for Supported Models)

**Use when:** Model has Inference API support (from major provider)

**Dependencies:**
```
gradio>=5.0.0
huggingface_hub>=0.26.0
```

**Key Code Pattern:**
```python
from huggingface_hub import InferenceClient

client = InferenceClient(MODEL_ID)

def respond(message, history):
    response = ""
    for token in client.chat_completion(messages, stream=True):
        response += token.choices[0].delta.content or ""
        yield response
```

### Template 2: ZeroGPU Full Model

**Use when:** Full model without Inference API, user has PRO subscription

**Dependencies:**
```
gradio>=5.0.0
torch
transformers
accelerate
spaces
```

**Key Code Pattern:**
```python
import spaces
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16)

@spaces.GPU(duration=120)  # Allocates GPU for up to 120 seconds
def generate(message, history):
    # GPU available only inside this function
    outputs = model.generate(...)
    return response
```

### Template 3: ZeroGPU LoRA Adapter

**Use when:** Model is a LoRA/PEFT adapter

**Dependencies:**
```
gradio>=5.0.0
torch
transformers
accelerate
spaces
peft
```

**Key Code Pattern:**
```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID)

# Apply adapter
model = PeftModel.from_pretrained(base_model, ADAPTER_ID)
model = model.merge_and_unload()  # Merge for faster inference
```

## Post-Deployment Steps

### 1. Set Hardware (Required for GPU Models)

After Space is created, go to Settings and select hardware:
- URL: `https://huggingface.co/spaces/USERNAME/SPACE_NAME/settings`
- Select: ZeroGPU (for free GPU) or paid tier

### 2. Monitor Build Status

```bash
# Watch until running or failed
python scripts/monitor_space.py watch username/space-name

# Check current status
python scripts/monitor_space.py status username/space-name
```

### 3. Analyze Errors if Build Fails

```bash
python scripts/monitor_space.py analyze-errors username/space-name
```

### 4. Auto-Fix Common Issues

```bash
python scripts/remediate.py auto-fix username/space-name
```

## Error Detection & Remediation

### Common Errors and Fixes

| Error Pattern | Cause | Fix |
|---------------|-------|-----|
| `ModuleNotFoundError: No module named 'X'` | Missing package | Add to requirements.txt |
| `CUDA out of memory` | Model too large | Upgrade hardware or use quantization |
| `CUDA is not available` | No GPU configured | Set hardware to GPU tier in Settings |
| `401/403 token` | Token missing/invalid | Add HF_TOKEN secret |
| `cannot import name 'HfFolder'` | Version mismatch | Use gradio>=5.0.0, huggingface_hub>=0.26.0 |
| `does not appear to have...safetensors` | LoRA loaded as full model | Use PEFT to load adapter |
| `Cannot use chat template` | Model missing template | Use text-gen instead or apply template |
| `GPU allocation timed out` | ZeroGPU queue | Try again later or use paid GPU |
| `examples must be nested list` | Gradio 5.x format | Use `[["ex1"], ["ex2"]]` not `["ex1", "ex2"]` |

### Auto-Remediation

The skill can automatically fix:
- Missing packages in requirements.txt
- Gradio/huggingface_hub version conflicts
- Hardware mismatches

```bash
# Dry run to see what would be fixed
python scripts/remediate.py auto-fix username/space-name --dry-run

# Apply fixes
python scripts/remediate.py auto-fix username/space-name
```

## Scripts Reference

### preflight.py - Pre-Deployment Validation

```bash
# Run all checks
python scripts/preflight.py check-all username/model-id

# Individual checks
python scripts/preflight.py check-token
python scripts/preflight.py check-subscription
python scripts/preflight.py check-model username/model-id
python scripts/preflight.py estimate-size username/model-id
```

### deploy_model.py - Main Deployment

```bash
# Basic deployment
python scripts/deploy_model.py MODEL_ID --type TYPE

# With options
python scripts/deploy_model.py MODEL_ID --type chat \
    --name my-space \
    --hardware l4 \
    --private \
    --org my-org

# Dry run
python scripts/deploy_model.py MODEL_ID --type chat --dry-run

# Skip pre-flight checks
python scripts/deploy_model.py MODEL_ID --type chat --skip-preflight
```

**Model Types:** `chat`, `text-generation`, `image-classification`, `text-to-image`, `embedding`

### monitor_space.py - Build/Runtime Monitoring

```bash
# Get current status
python scripts/monitor_space.py status username/space-name

# Watch build progress
python scripts/monitor_space.py watch username/space-name

# Get logs
python scripts/monitor_space.py logs username/space-name --type build
python scripts/monitor_space.py logs username/space-name --type runtime

# Analyze for errors
python scripts/monitor_space.py analyze-errors username/space-name

# Full health check
python scripts/monitor_space.py health-check username/space-name
```

### remediate.py - Auto-Fix Issues

```bash
# Auto-detect and fix issues
python scripts/remediate.py auto-fix username/space-name

# Add specific packages
python scripts/remediate.py fix-requirements username/space-name --add torch transformers

# Change hardware
python scripts/remediate.py fix-hardware username/space-name --tier zero-a10g

# Add secret
python scripts/remediate.py add-secret username/space-name --key HF_TOKEN --value hf_xxx

# Restart Space
python scripts/remediate.py restart username/space-name
```

### manage_space.py - Space Management

```bash
# Get status
python scripts/manage_space.py status username/space-name

# Change hardware
python scripts/manage_space.py hardware username/space-name --tier l4

# Manage secrets
python scripts/manage_space.py secret username/space-name --key KEY --value VALUE
python scripts/manage_space.py rm-secret username/space-name --key KEY

# Pause/restart
python scripts/manage_space.py pause username/space-name
python scripts/manage_space.py restart username/space-name
```

## Templates Reference

### Available Templates

| File | Pipeline | Description |
|------|----------|-------------|
| `gradio_chat.py` | text-generation | Inference API chat interface |
| `gradio_zerogpu_chat.py` | text-generation | ZeroGPU full model chat |
| `gradio_lora_chat.py` | text-generation | ZeroGPU LoRA adapter chat |
| `gradio_image_gen.py` | text-to-image | Image generation |
| `gradio_img2img.py` | image-to-image | Image-to-image transformation |
| `gradio_asr.py` | automatic-speech-recognition | Speech transcription |
| `gradio_tts.py` | text-to-speech | Text to speech |
| `gradio_vqa.py` | visual-question-answering | Visual QA with ZeroGPU |
| `gradio_object_detection.py` | object-detection | Object detection |
| `gradio_segmentation.py` | image-segmentation | Image segmentation |
| `gradio_zero_shot.py` | zero-shot-classification | Zero-shot text classification |
| `gradio_audio_class.py` | audio-classification | Audio classification |
| `gradio_depth.py` | depth-estimation | Monocular depth estimation |
| `streamlit_app.py` | text-generation | Streamlit chat interface |

### Template Variables

All templates use these placeholders:
- `{model_id}` - Full model ID (e.g., `username/model-name`)
- `{title}` - Human-readable title derived from model name
- `{base_model}` - Base model ID (for LoRA templates only)

## Gradio 5.x Requirements

**CRITICAL: Gradio 5.x has breaking changes**

### Examples Format
```python
# CORRECT (Gradio 5.x):
examples=[
    ["Example 1"],
    ["Example 2"],
]

# WRONG (causes ValueError):
examples=[
    "Example 1",
    "Example 2",
]
```

### Version Requirements
```
gradio>=5.0.0
huggingface_hub>=0.26.0
```

Do NOT use `gradio==4.44.0` - causes `ImportError: cannot import name 'HfFolder'`

## Troubleshooting Guide

### "No API found" Error
**Cause:** Gradio app isn't exposing API, often due to hardware mismatch
**Fix:** Go to Space Settings and set runtime to "ZeroGPU" or appropriate GPU tier

### "OSError: does not appear to have a file named..."
**Cause:** Trying to load a LoRA adapter as a full model
**Fix:** Check for `adapter_config.json`. If present, use PEFT:
```python
from peft import PeftModel
base_model = AutoModelForCausalLM.from_pretrained("base-model")
model = PeftModel.from_pretrained(base_model, "adapter-id")
```

### Inference API Not Available
**Cause:** Model doesn't have Inference API support
**Fix:** Either:
1. Add `pipeline_tag: text-generation` to model's README.md
2. Or use ZeroGPU with transformers instead of InferenceClient

### `ImportError: cannot import name 'HfFolder'`
**Cause:** gradio/huggingface_hub version mismatch
**Fix:** Use `gradio>=5.0.0` and `huggingface_hub>=0.26.0`

### `ValueError: examples must be nested list`
**Cause:** Gradio 5.x format change
**Fix:** Use `[["ex1"], ["ex2"]]` not `["ex1", "ex2"]`

### Space builds but model doesn't load
**Cause:** Missing `peft` for adapters, or wrong base model
**Fix:** Check adapter_config.json for correct `base_model_name_or_path`

### ZeroGPU Quota Exceeded
**Cause:** Daily GPU quota used up
**Fix:** Wait for quota reset (daily) or upgrade to PRO for more quota

### CUDA Out of Memory
**Cause:** Model too large for selected hardware
**Fix:** Upgrade to larger GPU tier (l40s, a100) or use quantization

## Testing Checklist

Before submitting PR, test these scenarios:

### Inference API Path
- [ ] Deploy popular model (e.g., `meta-llama/Llama-3.1-8B-Instruct`)
- [ ] Verify Space uses cpu-basic hardware
- [ ] Verify chat interface works

### ZeroGPU Full Model Path
- [ ] Deploy personal namespace model
- [ ] Verify ZeroGPU hardware recommended
- [ ] Verify model loads and generates

### ZeroGPU LoRA Path
- [ ] Deploy LoRA adapter with auto-detected base model
- [ ] Deploy LoRA adapter with explicit `--base-model`
- [ ] Verify PEFT loads correctly

### Error Detection
- [ ] Test with model that causes common errors
- [ ] Verify error patterns detected
- [ ] Verify fix suggestions are accurate

### Dry Run
- [ ] Verify `--dry-run` shows plan without changes
- [ ] Verify pre-flight checks run correctly

## CLI Quick Reference

```bash
# Deploy
python scripts/deploy_model.py MODEL --type TYPE [--dry-run]

# Monitor
python scripts/monitor_space.py watch SPACE
python scripts/monitor_space.py health-check SPACE

# Fix
python scripts/remediate.py auto-fix SPACE [--dry-run]

# Manage
python scripts/manage_space.py status SPACE
python scripts/manage_space.py hardware SPACE --tier TIER
```

## Workflow Summary

1. **Analyze model** (check for adapter_config.json, model files, inference widget)
2. **Run pre-flight checks** (token, subscription, model access)
3. **Determine strategy** (Inference API vs ZeroGPU, full model vs LoRA)
4. **Ask user ONLY if:**
   - LoRA adapter without detectable base model
   - ZeroGPU recommended but user lacks PRO
5. **Generate correct template** based on analysis
6. **Create Space** with correct requirements and README
7. **Upload files** using HF Hub API
8. **Set hardware** in Space Settings (ZeroGPU for free GPU access)
9. **Monitor build logs** for any issues
10. **Auto-remediate** common errors if detected

## Resources

- [HF Spaces Overview](https://huggingface.co/docs/hub/spaces-overview)
- [Spaces GPU Guide](https://huggingface.co/docs/hub/spaces-gpus)
- [ZeroGPU Documentation](https://huggingface.co/docs/hub/spaces-zerogpu)
- [Gradio Documentation](https://gradio.app/docs/)
- [PEFT Documentation](https://huggingface.co/docs/peft/)
