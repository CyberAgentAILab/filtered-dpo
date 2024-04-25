# Filtered Direct Preference Optimization

### tl;dr
 Introducing Filtered Direct Preference Optimization (fDPO) that enhances language model alignment with human preferences by discarding lower-quality samples compared to those generated by the learning model

## Prerequisites
- [Python 3.10.x][python]  
- [Poetry 1.7.x][poetry]  
- [direnv][direnv] 

[python]: https://www.python.org/downloads/release/python-31012/
[poetry]: https://python-poetry.org/
[direnv]: https://direnv.net/

## Get Started

To set up your local environment, start by copying the example environment file:

```shell
cp .env.example .env
```

Next, you need to edit the .env file to include your Hugging Face API token. Replace the placeholder value with your actual token:

```
HF_HUB_TOKEN="your_hugging_face_token_here"
```

If you do not already have a Hugging Face account or API token, you will need to create an account on Hugging Face and then generate an API token from your account settings.

Once your .env file is set up, apply the configuration to your environment using direnv:

```shell
direnv allow .
```

### Installation
```shell
poetry install
```

### Obtain Access to Datasets and Models

To use the datasets and models listed below, you must apply for access privileges on their respective Hugging Face repository pages. Please follow the links provided, and on each page, click the “Apply” button to submit your access request. This process is necessary to ensure compliance with the data usage policies and intellectual property rights associated with each resource.


- [Dataset][dataset] - Follow this link to apply for access to the dataset.
- [Model][model] - Follow this link to apply for access to the model.


[dataset]: https://huggingface.co/datasets/Mitsuki-Sakamoto/fdpo-preference-dataset
[model]: https://huggingface.co/Mitsuki-Sakamoto/fdpo-models

## Usage

### Test training
Execution time of about an hour in the notebook.
```
bash scripts/test.sh 
```

### Train 160m model
Execution time of several hours using A100 80G
```
# $seed in {1, 2, 3}
seed=1
bash scripts/160m/fdpo_mix.sh ${seed}
```


### Train 1.4b model
Execution time of about a day using A100 80G
```
# $seed in {1, 2, 3}
seed=1
bash scripts/1.4b/fdpo_mix.sh ${seed}
```

## Checking Experimental Results
The verification of experiment logs and creation of reports follow the standard of [Transformers](https://huggingface.co/docs/transformers/v4.39.3/en/main_classes/callback#callbacks) .


## Reference

[Morimura, T., Sakamoto, M., Jinnai, Y., Abe, K., and Ariu, K., Filtered Direct Preference Optimization. arXiv preprint arXiv:2404.13846, 2024.](https://arxiv.org/abs/2404.13846)

Bibtex:
```
@misc{morimura2024filtered,
      title={Filtered Direct Preference Optimization}, 
      author={Tetsuro Morimura and Mitsuki Sakamoto and Yuu Jinnai and Kenshi Abe and Kaito Ariu},
      year={2024},
      eprint={2404.13846},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```