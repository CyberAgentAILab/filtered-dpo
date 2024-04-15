import json
import logging
from typing import List, Literal, Optional

import numpy as np
from datasets import concatenate_datasets, load_dataset
from .reward_model import GPTNeoXRewardModel
from transformers import (AutoModelForCausalLM,
                          AutoModelForSequenceClassification, AutoTokenizer,
                          pipeline)

logger = logging.getLogger(__name__)


def load_format_dict_from_json(json_path):
    with open(json_path, "r") as f:
        format_dict = json.load(f)
    return format_dict


def alpaca_farm_format_prompt(example, prompt_dict):
    formatted_expample = ""
    if example["input"] is None or len(example["input"]) == 0:
        formatted_expample = prompt_dict["prompt_noinputs"].format_map(example)
    else:
        formatted_expample = prompt_dict["prompt_inputs"].format_map(example)
    return formatted_expample


def alpaca_farm_pre_preprocess_function(json_path):
    prompt_dict = load_format_dict_from_json(json_path)

    def alpaca_preprocess_function(examples):
        if "input" not in examples:
            examples["input"] = ["" for _ in range(len(examples["instruction"]))]
        prompts = []
        for instruction, input_ in zip(examples["instruction"], examples["input"]):
            example = {"instruction": instruction, "input": input_}
            prompt = alpaca_farm_format_prompt(example, prompt_dict)
            prompts.append(prompt)
        new_examples = {
            "unformatted_prompt": prompts,
        }
        new_examples.update(examples)
        return new_examples

    return alpaca_preprocess_function


def load_and_process_dataset(
    dataset_path,
    dataset_name,
    split,
    hub_token=None,
    preprocess_function=None,
    filter_function=None,
    seed=42,
    percentage=100,
    batched=False,
    name="",
    num_logged_samples=10,
):
    dataset = load_dataset(dataset_path, name=dataset_name, split=split, token=hub_token)
    logger.info(f"Loading {name} dataset from {dataset_path}/{dataset_name}")
    dataset = dataset.shuffle(seed=seed)
    dataset = dataset.select(range(int(len(dataset) * percentage / 100)))
    logger.info(f"{name} dataset size: {len(dataset)}, percentage: {percentage}%")

    if preprocess_function is not None:
        dataset = dataset.map(preprocess_function, batched=batched)
    if filter_function is not None:
        dataset = dataset.filter(filter_function)

    logger.info(f"Preprocessed {name} dataset: {dataset}")
    num_logged_samples = min(num_logged_samples, len(dataset))
    for i_e, example in enumerate(dataset):
        if i_e >= num_logged_samples:
            break
        logger.info(f"Preprocessed {name} Dataset example {i_e}: {example}")
    return dataset


def train_test_split(dataset, test_size=0.1, seed=42):
    train_test_split_dataset = dataset.train_test_split(test_size=test_size, seed=seed)
    train_dataset, eval_dataset = (
        train_test_split_dataset["train"],
        train_test_split_dataset["test"],
    )
    logger.info(f"Processed train dataset: {train_dataset}")
    logger.info(f"Processed eval dataset: {eval_dataset}")
    return train_dataset, eval_dataset


def load_model(
    model_loader,
    model_path,
    **kwargs,
):
    # Want to implmet a fuction that loads a model from a given path of cloud storage
    # if it not exists on the local machine
    model = model_loader(model_path, **kwargs)
    if hasattr(model, "config"):
        logger.info(f"Loaded model config {model.config}")
    else:
        logger.info(f"Loaded model {model}")
    return model


def load_tokenizer(
    tokenizer_loader,
    tokenizer_path,
    **kwargs,
):
    # Want to implmet a fuction that loads a tokenizer from a given path of cloud storage
    # if it not exists on the local machine
    tokenizer = tokenizer_loader(tokenizer_path, **kwargs)
    logger.info(f"Loaded tokenizer {tokenizer}")
    return tokenizer


def check_and_add_pad_token(model, tokenizer):
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token
        if model.config.pad_token_id is None:
            model.config.pad_token_id = tokenizer.eos_token_id
    logger.info(f"Added tokenizer: {tokenizer}")


def create_reward_pipeline(
    reward_model_path: str,
    device: str,
    reward_model_type: str = "sentiment-analysis",
    reward_model_subfolder: str = "",
    hub_token: Optional[str] = None,
    context_manager=None,
    **kwargs,
):
    reward_model = load_model(
        AutoModelForSequenceClassification.from_pretrained,
        reward_model_path,
        token=hub_token,
        subfolder=reward_model_subfolder,
        **kwargs,
    )
    reward_model_tokenizer = load_tokenizer(
        AutoTokenizer.from_pretrained,
        reward_model_path,
        token=hub_token,
        subfolder=reward_model_subfolder,
        **kwargs,
    )
    if getattr(reward_model_tokenizer, "pad_token", None) is None:
        reward_model_tokenizer.pad_token = reward_model_tokenizer.eos_token
        if reward_model.config.pad_token_id is None:
            reward_model.config.pad_token_id = reward_model_tokenizer.eos_token_id
    if context_manager is not None:
        raise NotImplementedError("Context manager not implemented")
        with context_manager:
            reward_pipe = pipeline(
                reward_model_type,
                model=reward_model,
                tokenizer=reward_model_tokenizer,
                device=device,
            )
    else:
        reward_pipe = pipeline(
            reward_model_type,
            model=reward_model,
            tokenizer=reward_model_tokenizer,
            device=device,
        )
    return reward_pipe


def create_generate_pipeline(
    policy_model_path: str,
    device: str,
    **kwargs,
):
    policy_model = load_model(AutoModelForCausalLM.from_pretrained, policy_model_path, **kwargs)
    policy_model_tokenizer = load_tokenizer(
        AutoTokenizer.from_pretrained, policy_model_path, padding_side="left", **kwargs
    )
    check_and_add_pad_token(policy_model, policy_model_tokenizer)

    return pipeline(
        "text-generation",
        model=policy_model,
        tokenizer=policy_model_tokenizer,
        device=device,
    )
