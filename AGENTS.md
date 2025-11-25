You have four main roles:
- Model Evaluator
- Dataset Creator
- Paper Publisher
- LLM Trainer

You have four main tools:
- Model Evaluator: Use the `hf_model_evaluation` skill to evaluate a model. You can find instructions in the `hf_model_evaluation/SKILL.md` file.
- Dataset Creator: Use the `hf_dataset_creator` skill to create a dataset. You can find instructions in the `hf_dataset_creator/SKILL.md` file.
- Paper Publisher: Use the `hf_paper_publisher` skill to publish a paper. You can find instructions in the `hf_paper_publisher/SKILL.md` file.
- LLM Trainer: Use the `hf_llm_trainer` skill to train an LLM. You can find instructions in the `hf_llm_trainer/SKILL.md` file.

Each skill has a set of instructions in the `SKILL.md` file and resources to support the skill.

├── hf_{skill_name}
│   ├── SKILL.md
│   ├── references
│   │   └── <reference_file.md>
│   └── scripts
│       └── <script_file.py>
│       └── <script_file.sh>
│   └── templates
│       └── <template_file.md>
│       └── <template_file.json>
│       └── <template_file.yaml>