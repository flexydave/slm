# African Folktales Small Language Model

This project generates stories for a small, synthetic African-folktale benchmark. It implements a TF-IDF retrieval baseline and LoRA adaptation of Gemma 2B IT; both condition on prompt, theme, and region. The benchmark scores mean character-level Levenshtein distance.

## 1. Dataset

`data/documents.csv` is a 24-document synthetic corpus with `document_id`, `title`, `text`, `theme`, `culture_region`, `origin`, `source_url`, and `license`. Its six themes have four documents each; regions are West (6), East (6), Southern (5), Central (5), and diaspora (2). Every row is `synthetic` and marked `CC0-1.0`.

`data/train_prompts.csv` contains 38 prompt/reference-story pairs with `PromptId`, `prompt`, `theme`, `culture_region`, `document_id`, and `reference_story`; all 24 linked IDs exist in the corpus. `data/test_prompts.csv` has 10 prompt-only rows; hidden references are absent. Dataset metadata declares CC-BY-4.0, unlike the per-document CC0-1.0 values; reconcile this before release.

The retrieval notebook lowercases text, replaces non-alphanumeric characters with spaces, and collapses whitespace. No separate cleaning, deduplication, or train/validation split is implemented for LoRA training.

## 2. Training Pipeline

### Retrieval baseline

`scripts/document-retrieval.ipynb` builds each corpus representation from `title`, `text`, `theme`, and `culture_region`; each test representation combines `prompt`, `theme`, and `culture_region`. A `TfidfVectorizer` uses word unigrams/bigrams (`min_df=1`, `sublinear_tf=True`), then cosine similarity selects the top document with `argmax`. The selected document's original `text` becomes `Story` in `submission_retrieval.csv`; there is no fallback or generative step.

### LoRA generation

`scripts/final-submission.ipynb` uses Kaggle-mounted `google/gemma/transformers/2b-it/2`. It formats 38 training pairs plus 24 corpus documents as `### Prompt`, `### Theme`, `### Region`, and `### Story` (62 sequences). The tokenizer uses EOS as pad when needed and truncates to 256 tokens; `DataCollatorForLanguageModeling(mlm=False)` supplies causal-LM labels.

Gemma is loaded in FP16 when CUDA is available (otherwise FP32) and adapted with PEFT LoRA: rank 16, alpha 32, dropout 0.05, no bias, and targets `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, and `down_proj`. Training runs for 8 epochs with per-device batch size 2, gradient accumulation 4, learning rate `2e-4`, epoch checkpointing, and logging every five steps. The adapter and tokenizer are saved to `/kaggle/working/folktale-lora`.

For each test row, it samples 10 candidates (`max_new_tokens=120`, temperature 0.7, top-p 0.9, repetition penalty 1.1), then removes echoed prompt sections. No systematic hyperparameter search is implemented: the notebook has one explicit configuration. LoRA confines updates to adapter layers.

## 3. Evaluation

**Mean Levenshtein Distance - lower is better.** The benchmark documentation specifies character-level distance against hidden test references. There is no held-out validation split in the current notebooks.

The retrieval notebook prints top-three matches and writes 10 predictions. The final notebook scores each LoRA candidate against the matching retrieved `Story`, keeps the lowest-distance candidate, and writes the two-column submission. This selects against retrieval output, not hidden references. The data card separately reports 38/38 associated training-document retrieval and a 112.42 mean source-document/reference-story distance; it is not a Kaggle score. No leaderboard score is stored.

## 4. Reproduction

The notebooks target Kaggle; this repository has no requirements file or pinned environment. The fine-tuning workflow explicitly runs:

```bash
pip install --upgrade torchao
pip install -q python-Levenshtein
```

It also imports `numpy`, `pandas`, `scikit-learn`, `torch`, `datasets`, `transformers`, and `peft`.

1. Clone the repository and retain the CSV files under `data/`:
   ```bash
   git clone https://github.com/flexydave/slm.git
   cd slm
   ```
2. Attach the competition input at `/kaggle/input/competitions/african-folktales-slm-challenge`. The unmodified retrieval notebook also reads `sample_submission.csv` and `baseline_submission.csv`; neither is versioned here.
3. Run `scripts/document-retrieval.ipynb` end to end. It creates `/kaggle/working/submission_retrieval.csv` with `PromptId` and `Story`.
4. Publish/attach that notebook output so the file is mounted at `/kaggle/input/notebooks/davidattah/document-retrieval/submission_retrieval.csv`, and attach the Gemma model at `/kaggle/input/models/google/gemma/transformers/2b-it/2`.
5. Run `scripts/final-submission.ipynb` end to end. It trains the adapter, generates/selects candidates, and creates `/kaggle/working/submission.csv` with the required `PromptId,Story` columns.

## 5. Repository Structure

```text
.
├── data/
│   ├── documents.csv
│   ├── train_prompts.csv
│   ├── test_prompts.csv
│   └── dataset-metadata.json
├── docs/                       # data, impact, problem, and stakeholder cards
├── scripts/
│   ├── document-retrieval.ipynb
│   └── final-submission.ipynb
└── README.md
```

## 6. Appendix

### Contributors

* David Attah - sole Git author; prepared `docs/stakeholder_engagement.pdf`.
* Orangun Folaranmi
* Bright Francis

### Mentors

Mr Patrick Owor

## References

* [Project data card](docs/data_card.pdf)
* [Gemma 2B IT model card](https://huggingface.co/google/gemma-2b-it)
* [Hugging Face PEFT documentation](https://huggingface.co/docs/peft/index)
* [Hugging Face Transformers documentation](https://huggingface.co/docs/transformers/index)
