# Who Changed Their Tone?

Rachel Sherpa : NLP Homework 2, Hanyang University

This project tests whether Qwen2.5-3B-Instruct can detect when a speaker changes between polite Korean (Haeyo-che) and informal Korean (Hae-che). A difference between two speakers does not count as a shift: the same speaker must change their own speech level.

The dataset contains 224 six-turn dialogues based on eight scenarios: 32 without a shift (H1) and 192 with a shift (H2). The notebook compares direct model predictions with two pipelines that classify each utterance and then track speech levels separately for each speaker.

## Files

Keep these files together. If a downloaded filename ends in `(1)` or `(3)`, rename it to match this list without changing its contents.

| File | Contents |
| --- | --- |
| `HW2_final_experiments.ipynb` | Experiments and analysis |
| `final_dataset_v1.json` | Dataset and main prompt |
| `final_direct_results.json` | Direct model predictions |
| `final_recognition_results.json` | Classifications for the 96 distinct utterances |
| `final_pipeline_results.json` | Predictions from the two tracking pipelines |
| `README.md` | Instructions |

## Reproduce the results

You can reproduce the tables from the saved JSON files without loading the model or using a GPU.

### In Google Colab

1. Open the notebook and upload the four JSON files using the filenames above.
2. Run the code cell below **Quantitative analysis**.
3. Run the next code cell for error analysis and sampled examples.

You do not need to run the earlier inference cells for this step.

### Locally

Install the analysis packages:

```bash
python -m pip install numpy pandas
```

From the folder containing the notebook and JSON files, run:

```bash
python - <<'PY'
import json
from pathlib import Path

notebook = json.loads(
    Path('HW2_final_experiments.ipynb').read_text(encoding='utf-8')
)
namespace = {}
for index in (6, 7):
    source = ''.join(notebook['cells'][index]['source'])
    source = source.replace('from google.colab import files', '')
    exec(compile(source, f'cell_{index}', 'exec'), namespace)
PY
```

This runs the two analysis cells from the submitted notebook. All four JSON files must already be in the folder; the command removes the Colab import and does not support interactive uploads. The cell indices refer to this notebook version, counting code and Markdown cells from zero.

The command prints the main results, H1 results by condition, H2 results by gap, bootstrap contrasts, error categories, coverage and sampled examples. It also saves:

- `main_results.csv`
- `bootstrap_contrasts.csv`
- `error_categories.csv`
- `sampled_examples.json`

The analysis was checked in a separate temporary folder using the saved results and reproduced the reported scores.

## Run the model again

The inference code is intended for Google Colab with a T4 GPU. It installs `transformers`, `accelerate` and `bitsandbytes`, then loads `Qwen/Qwen2.5-3B-Instruct` in 4-bit NF4 with double quantization and float16 computation. Downloading the model requires internet access.

To generate new predictions, start a fresh GPU runtime, upload `final_dataset_v1.json`, and run the notebook cells in order. Do not upload the result JSON files if you want a completely new run: existing results are used to resume completed work. Download the output files before closing the runtime.

The saved direct run records torch `2.14.1+cu130` and transformers `5.22.1`. Other package versions were not recorded, and the notebook does not pin package versions or a model revision. A complete fresh GPU run has not been checked during preparation of this README, so identical inference outputs across environments are not guaranteed.

### Settings

| Setting | Value |
| --- | --- |
| Generation | Greedy decoding (`do_sample=False`) |
| Maximum new tokens | 32 for direct prediction; 16 for utterance classification |
| Model seed | 42 |
| Bootstrap | 10,000 paired resamples of the eight scenarios; NumPy seed 42 |
| Example sampling | Python random seed 42; grouped by experiment and correctness |

The notebook and saved JSON files contain the prompts. Bootstrap sampling keeps related variants together. Utterance classifications are cached and reused across dialogues.

The notebook checks the dataset against this SHA-256 hash:

```text
bbf21bff6c99c2a059752d6fa6ef4d1052886e744944a552d0b15ccee8a41699
```

## Main results

| Method | H1 accuracy | H2 exact localization |
| --- | --- | --- |
| Direct prediction | 20/32 (62.50%) | 9/192 (4.69%) |
| Model classification + tracking | 0/32 (0%) | 0/192 (0%) |
| Rule classification + tracking | 22/32 (68.75%) | 132/192 (68.75%) |

The model classified 52 of 96 distinct utterances correctly (54.17%). The rule pipeline answered 154 dialogues correctly and abstained on 70; abstentions count as errors in the table.

For H1, the main contrast is the false-positive rate in contrasting-level dialogues minus that in same-level dialogues. For H2, it is exact localization accuracy at gap 2 minus accuracy at gap 0. Direct H2 scores are 0/64, 5/64 and 4/64 at gaps 0, 1 and 2.

## Limitations

These results concern one quantized model and a small synthetic dataset. The Korean examples were created with AI assistance and have not been independently reviewed by a proficient Korean speaker. The rules cover only a limited set of endings. In H2, changing the gap also changes the position of the preceding target utterance, so distance is not fully separated from positional effects.
