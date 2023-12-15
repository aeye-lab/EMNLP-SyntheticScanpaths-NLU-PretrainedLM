# Pre-Trained Language Models Augmented with Synthetic Scanpaths for Natural Language Understanding
[![paper](https://img.shields.io/static/v1?label=paper&message=download%20link&color=brightgreen)]([https://arxiv.org/abs/2310.14676](https://aclanthology.org/2023.emnlp-main.400/))

In this paper, we develop a model that integrates synthetic scanpath generation with a scanpath-augmented language model, eliminating the need for human gaze data. Since the model’s error gradient can be propagated throughout all parts of the model, the scanpath generator can be fine-tuned to downstream tasks. We find that the proposed model not only outperforms the underlying language model, but achieves a performance that is comparable to a language model augmented with real human gaze data.

## Setup
Clone repository:

```
git clone https://github.com/aeye-lab/EMNLP-SyntheticScanpaths-NLU-PretrainedLM
```

Install dependencies:

```
pip install -r requirements.txt
```

Download precomputed sn_word_len mean and std (from CELER dataset) for Eyettention model feature normalization:

```
wget https://github.com/aeye-lab/EMNLP-SyntheticScanpaths-NLU-PretrainedLM/releases/download/v1.0/feature_norm_celer.pickle
```

## Run Experiments
To reproduce the results in Section 3.2:
```
python run_ETSA_ours.py
python run_ETSA_bert.py
python run_ETSA_PLM_AS.py
```

To reproduce the results in Section 3.3:
```
python run_glue_ours_high_resource.py
python run_glue_ours_low_resource.py
python run_glue_bert_high_resource.py
python run_glue_bert_low_resource.py
```

To pre-train the Eyettention model (For more details see https://github.com/aeye-lab/Eyettention):
```
run_pretrain_eyettention_celer_position_prediction.py
```


## Cite our work
If you use our code for your research, please consider citing our paper:

```bibtex
@inproceedings{deng2023-augmented,
    author = "Deng, Shuwen and Prasse, Paul and Reich, David R. and Scheffer, Tobias and J{\"a}ger, Lena A.",
    title = "{P}re-Trained Language Models Augmented with Synthetic Scanpaths for Natural Language Understanding",
    booktitle = "Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing (EMNLP 2023)",
    year = "2023",
    month = dec,
    address = "Singapore, Singapore",
    publisher = "Association for Computational Linguistics",
    url = "",
    doi = "",
    pages = ""
}
```

