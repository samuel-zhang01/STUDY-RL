# Security

This repository contains educational simulations and an optional local LLM
trainer. Supported security fixes target the current `main` branch.

Use GitHub's private vulnerability reporting feature on this repository's
Security tab for suspected credentials, sensitive data, or exploitable code.
If private reporting is unavailable, open an issue requesting a private contact
without including the sensitive material. Do not post credentials or private
records in a public issue or pull request.

Useful reports identify the affected revision and path, expected impact, and a
minimal reproduction with synthetic data. The maintainer will investigate and
coordinate a fix; there is no guaranteed response-time commitment.

## Running the materials

- Notebook execution runs Python code. PDF builds use TeX shell escape for
  `minted`. Run them only from a checkout you trust.
- LLM loading disables `trust_remote_code`. Prefer safetensors, record base-model
  revisions, and load adapters and model files from sources you trust.
- Model downloads use Hugging Face; hosted model access and redistribution
  remain subject to the model's own terms. No paid API is required for the lab.
- `runs/`, model weights, environment files, credentials, and local assistant
  settings are ignored. An ignore rule does not remove an already committed file.
- `make secrets` scans all reachable Git refs with Gitleaks and redacts findings.
  If a real credential is exposed, revoke or rotate it as well as removing it
  from history.
