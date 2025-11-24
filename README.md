# Using Hugging Face skills with Claude Code

## Quests

- Week 1 - [Evaluate a Hub model](quests/evaluate-hub-model.md)
- Week 2 - [Supervised fine-tuning on the Hub](quests/sft-finetune-hub.md)
- Week 3 - [Publish a Hub dataset](quests/publish-hub-dataset.md)

## What are Claude skills?

Claude skills are self-contained folders that package instructions, scripts, and resources. Each folder includes a `SKILL.md` file with YAML frontmatter (name and description) followed by the guidance Claude follows while the skill is active. Skills can target creative work, technical automation, or enterprise workflows—this repository focuses on Hugging Face tasks such as data set preparation, evaluation, and model training.

## Repository layout

- `hf_dataset_creator/` – prompts, templates, and scripts for creating structured training data sets. Includes reusable templates in `templates/`, sample prompts in `examples/`, and orchestration code in `scripts/dataset_manager.py`.
- `hf_model_evaluation/` – instructions plus utilities for orchestrating evaluation jobs, generating reports, and mapping metrics. The `scripts/` folder hosts entry points such as `run_eval_job.py`, `evaluation_manager.py`, and supporting tools. Requirements live in `requirements.txt`.
- `hf-llm-trainer/` – a comprehensive training skill with `SKILL.md` guidance, helper scripts (e.g., `train_sft_example.py`, `convert_to_gguf.py`, cost estimators), and deep reference docs under `references/`.
- `hf-paper-publisher/` – tools for publishing and managing research papers on Hugging Face Hub. Index papers from arXiv, link papers to models/datasets, generate professional research articles from templates, and manage paper authorship. Includes paper templates (`standard`, `modern`, `arxiv`, `ml-report`) and citation generation utilities.
- `pyproject.toml` defines minimal tooling metadata for the repository, and `LICENSE` covers usage.

## Install these skills in Claude Code

1. Register the repository as a plugin marketplace:
   ```
   /plugin marketplace add huggingface/skills
   ```
2. In Claude Code, open **Browse and install plugins** → **huggingface-skills** (the marketplace you just added).
3. Choose the folder you want (`hf-llm-trainer`, `hf_model_evaluation`, `hf_dataset_creator`, or `hf-paper-publisher`) and select **Install now**.
4. Prefer commands? After registering the marketplace, run `/plugin install <skill-folder>@huggingface-skills` (for example, `/plugin install hf-llm-trainer@huggingface-skills`).

## Use an installed skill

Once a skill is installed, mention it directly while giving Claude Code instructions:

- "Use the HF LLM trainer skill to estimate the GPU memory needed for a 70B model run."
- "Use the HF model evaluation skill to launch `run_eval_job.py` on the latest checkpoint."
- "Use the HF dataset creator skill to draft new few-shot classification templates."
- "Use the HF paper publisher skill to index my arXiv paper and link it to my model."

Claude automatically loads the corresponding `SKILL.md` instructions and helper scripts while it completes the task.

## Contribute or customize a skill

1. Copy one of the existing skill folders (for example, `hf_dataset_creator/`) and rename it.
2. Update the new folder’s `SKILL.md` frontmatter:
   ```markdown
   ---
   name: my-skill-name
   description: Describe what the skill does and when to use it
   ---

   # Skill Title
   Guidance + examples + guardrails
   ```
3. Add or edit supporting scripts, templates, and documents referenced by your instructions.
4. Reinstall or reload the skill bundle in Claude Code so the updated folder is available.

## Additional references

- Browse the latest instructions, scripts, and templates directly at [huggingface/skills](https://github.com/huggingface/skills).
- Review Hugging Face documentation for the specific libraries or workflows you reference inside each skill.