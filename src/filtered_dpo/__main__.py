import logging
import os
from dataclasses import dataclass, field
from typing import List, Literal, Optional

import torch
import transformers
from accelerate import Accelerator
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
    pipeline,
)
from trl import AutoModelForCausalLMWithValueHead, DPOTrainer, PPOConfig, PPOTrainer
from trl.core import logprobs_from_logits
from trl.trainer.utils import pad_to_length

from . import utils
from .callbacks import EvalGoldRewardModelCallback, FilterTrainDatasetCallback
from .reward_model import GPTNeoXRewardModel


# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    The arguments for the fDPO training script.
    """

    policy_model_path: str = field(
        default="Mitsuki-Sakamoto/pythia_160m_alpaca_farm_instructions_sft_constant_pa_seed_1",
        metadata={"help": "the path to the policy model"},
    )
    policy_model_subfolder: str = field(default="", metadata={"help": "the subpath to the policy model"})
    dataset_train_path: str = field(
        default="Mitsuki-Sakamoto/alpaca_farm-reward-model-deberta-v3-large-v2-re-preference-64-nsample-2-16_mix_random_seed_1",
        metadata={"help": "the dataset path for train"},
    )
    dataset_train_name: str = field(
        default="alpaca_instructions-pythia_160m_alpaca_farm_instructions_sft_constant_pa_seed_1",
        metadata={"help": "the dataset name for train"},
    )
    dataset_train_split: str = field(
        default="preference",
        metadata={"help": "the split of the dataset to use for training"},
    )
    dataset_train_percentage: float = field(default=100, metadata={"help": "the percentage of the dataset to use"})
    test_size: float = field(default=0.05, metadata={"help": "the test size for the train test split"})

    dataset_eval_prompt_path: Optional[str] = field(
        default="tatsu-lab/alpaca_farm", metadata={"help": "the path to the prompt dataset"}
    )
    dataset_eval_prompt_name: Optional[str] = field(
        default="alpaca_instructions", metadata={"help": "the name of the prompt dataset"}
    )
    dataset_eval_prompt_percentage: float = field(
        default=100, metadata={"help": "the number of examples of the dataset to use"}
    )
    dataset_eval_prompt_split: str = field(
        default="val",
        metadata={"help": "the split of the dataset to use for evaluation"},
    )
    prompt_response_format_json: str = field(
        default="instruction_format_json/prompt_response_format/prompter_assistant.json",
        metadata={"help": "the json file to format the prompt and response"},
    )
    prompt_format_json: str = field(
        default="instruction_format_json/alpaca_farm_format_prompt/ln.json",
        metadata={"help": "the json file to format the prompt using alpaca_farm_format_prompt"},
    )
    proxy_reward_model_type: str = field(
        default="sentiment-analysis",
        metadata={"help": "the type of the proxy reward model"},
    )
    proxy_reward_model_path: Optional[str] = field(
        default=None, metadata={"help": "the path to the proxy reward model"}
    )
    proxy_reward_model_subfolder: str = field(default="", metadata={"help": "the subpath to the proxy reward model"})
    proxy_reward_model_format_json: Optional[str] = field(
        default=None, metadata={"help": "the json file to format for the reward model"}
    )
    gold_reward_model_type: Optional[str] = field(
        default="sentiment-analysis",
        metadata={"help": "the type of the gold reward model"},
    )
    gold_reward_model_path: Optional[str] = field(default=None, metadata={"help": "the path to the gold reward model"})
    gold_reward_model_subfolder: str = field(default="", metadata={"help": "the subpath to the gold reward model"})
    gold_reward_model_format_json: Optional[str] = field(
        default=None, metadata={"help": "the json file to format for the reward model"}
    )
    init_eval: bool = field(default=False, metadata={"help": "whether to evaluate the gold reward model"})
    # generation parameters for policy model when evaluating reward models
    reward_model_bath_size: int = field(default=8, metadata={"help": "the batch size for the reward model"})
    top_k: int = field(default=0, metadata={"help": "the top k tokens to sample from"})
    top_p: float = field(default=1.0, metadata={"help": "the top p tokens to sample from"})
    temperature: float = field(default=1.0, metadata={"help": "the temperature for sampling"})
    policy_generate_max_length: int = field(default=64, metadata={"help": "the maximum length of the generated text"})
    beta: Optional[float] = field(default=0.01, metadata={"help": "the beta parameter for DPO loss"})
    # training parameters
    max_length: Optional[int] = field(default=520 + 256, metadata={"help": "max length of each sample"})
    max_prompt_length: Optional[int] = field(default=520, metadata={"help": "max length of each sample's prompt"})
    max_target_length: Optional[int] = field(
        default=256,
        metadata={"help": "Only used for encoder decoder model. Max target of each sample's prompt"},
    )
    sanity_check: Optional[bool] = field(default=True, metadata={"help": "only train on 1000 samples"})
    filtered_train_dataset: bool = field(default=False, metadata={"help": "whether to filter the train dataset"})
    filtered_threshold: Optional[float] = field(
        default=0.0, metadata={"help": "the threshold for the filtered reward model"}
    )
    filtered_start_epoch: int = field(default=0, metadata={"help": "the start epoch for the filtered dataset"})
    filtered_top_k: int = field(default=0, metadata={"help": "the top k tokens to sample from"})
    filtered_top_p: float = field(default=1.0, metadata={"help": "the top p tokens to sample from"})
    filtered_temperature: float = field(default=1.0, metadata={"help": "the temperature for sampling"})


@dataclass
class TrainingArguments(transformers.TrainingArguments, ScriptArguments):
    per_device_train_batch_size: Optional[int] = field(default=4, metadata={"help": "batch size per device"})
    max_steps: Optional[int] = field(default=-1, metadata={"help": "max number of training steps"})
    remove_unused_columns: bool = field(default=False, metadata={"help": "whether to remove unused columns"})
    learning_rate: Optional[float] = field(default=1e-6, metadata={"help": "optimizer learning rate"})
    gradient_accumulation_steps: Optional[int] = field(
        default=1, metadata={"help": "the number of gradient accumulation steps"}
    )
    report_to: Optional[str] = field(
        default=None,
        metadata={
            "help": 'The list of integrations to report the results and logs to. Supported platforms are `"azure_ml"`,'
            '`"comet_ml"`, `"mlflow"`, `"neptune"`, `"tensorboard"`,`"clearml"` and `"wandb"`. '
            'Use `"all"` to report to all integrations installed, `"none"` for no integrations.'
        },
    )
    logging_first_step: bool = field(default=True, metadata={"help": "Whether to log the first training step"})
    logging_strategy: str = field(
        default="steps",
        metadata={"choices": ["steps", "epoch"], "help": "the logging strategy"},
    )
    run_name: Optional[str] = field(default=None, metadata={"help": "the run name for mlflow and wandb"})
    output_dir: str = field(default="output", metadata={"help": "the output directory"})
    optim: str = field(default="rmsprop", metadata={"help": "the optimizer type"})
    lr_scheduler_type: str = field(default="constant", metadata={"help": "the learning rate scheduler type"})
    evaluation_strategy: str = field(
        default="epoch",
        metadata={"choices": ["steps", "epoch"], "help": "the evaluation strategy"},
    )
    save_strategy: str = field(
        default="epoch",
        metadata={"choices": ["steps", "epoch"], "help": "the save model strategy"},
    )
    gradient_checkpointing: Optional[bool] = field(
        default=False, metadata={"help": "Whether to use gradient checkpointing or no"}
    )
    log_level: str = field(default="info", metadata={"help": "The logging level"})


def main(args: Optional[List[str]] = None) -> None:
    parser = HfArgumentParser(TrainingArguments)
    training_args = parser.parse_args_into_dataclasses()[0]
    logging.basicConfig(level=getattr(logging, training_args.log_level.upper()))
    script_args = training_args
    hub_token = training_args.hub_token if training_args.hub_token is not None else os.environ.get("HF_TOKEN")
    del training_args.hub_token
    logger.info(f"Parsed Training Args {training_args}")
    logger.info(f"Parsed Script Arg {script_args}")

    # 1. load a pretrained model
    model = utils.load_model(
        AutoModelForCausalLM.from_pretrained,
        script_args.policy_model_path,
        subfolder=script_args.policy_model_subfolder,
        token=hub_token,
        device_map=None,
    )
    model_ref = utils.load_model(
        AutoModelForCausalLM.from_pretrained,
        script_args.policy_model_path,
        subfolder=script_args.policy_model_subfolder,
        token=hub_token,
        device_map=None,
    )
    tokenizer = utils.load_tokenizer(
        AutoTokenizer.from_pretrained,
        script_args.policy_model_path,
        subfolder=script_args.policy_model_subfolder,
        token=hub_token,
    )
    utils.check_and_add_pad_token(model, tokenizer)

    # preprocessing dataset
    pre_preprocess_function = utils.alpaca_farm_pre_preprocess_function(script_args.prompt_format_json)
    prompt_response_dict = utils.load_format_dict_from_json(script_args.prompt_response_format_json)

    def preprocess_function(examples):
        pre_preprocessed_examples = (
            pre_preprocess_function(examples) if pre_preprocess_function is not None else examples
        )
        prompts = []
        chosens = []
        rejecteds = []
        for unformatted_prompt, output_1, output_2, preference in zip(
            pre_preprocessed_examples["unformatted_prompt"],
            pre_preprocessed_examples["output_1"],
            pre_preprocessed_examples["output_2"],
            pre_preprocessed_examples["preference"],
        ):
            chosen, rejected = (output_1, output_2) if preference == 1 else (output_2, output_1)
            prompt_formatted = prompt_response_dict["prompt"].format_map({"prompt": unformatted_prompt})
            chosen = prompt_response_dict["response"].format_map({"response": chosen})
            rejected = prompt_response_dict["response"].format_map({"response": rejected})
            prompts.append(prompt_formatted)
            chosens.append(chosen)
            rejecteds.append(rejected)
        new_examples = {
            "unformatted_prompt": pre_preprocessed_examples["unformatted_prompt"],
            "prompt": prompts,
            "chosen": chosens,
            "rejected": rejecteds,
        }
        return new_examples

    def filter_func(x):
        return (
            len(x["prompt"]) + len(x["chosen"]) <= script_args.max_length
            and len(x["prompt"]) + len(x["rejected"]) <= script_args.max_length
            and len(x["prompt"]) <= script_args.max_prompt_length
            and len(x["chosen"]) <= script_args.max_target_length
            and len(x["rejected"]) <= script_args.max_target_length
        )

    train_dataset = utils.load_and_process_dataset(
        script_args.dataset_train_path,
        script_args.dataset_train_name,
        script_args.dataset_train_split,
        hub_token=hub_token,
        preprocess_function=preprocess_function,
        filter_function=filter_func,
        seed=training_args.data_seed,
        percentage=script_args.dataset_train_percentage,
        batched=True,
        name="train",
    )

    if script_args.test_size > 0:
        train_dataset, eval_dataset = utils.train_test_split(train_dataset, test_size=script_args.test_size)
    else:
        eval_dataset = None
        logger.info("No eval dataset provided")

    eval_prompt_dataset = None
    if script_args.dataset_eval_prompt_path is not None:

        def eval_alpaca_preprocess_function(samples):
            pre_preprocessed_samples = (
                pre_preprocess_function(samples) if pre_preprocess_function is not None else samples
            )
            prompt_input_ids_list = []
            prompts_list = []
            for unformatted_prompt in pre_preprocessed_samples["unformatted_prompt"]:
                formatted_prompt = prompt_response_dict["prompt"].format_map({"prompt": unformatted_prompt})
                prompt_input_ids = tokenizer.encode(formatted_prompt)
                prompt_input_ids_list.append(prompt_input_ids)
                prompts_list.append(tokenizer.decode(prompt_input_ids))
            new_samples = {
                "unformatted_prompt": pre_preprocessed_samples["unformatted_prompt"],
                "prompt_input_ids": prompt_input_ids_list,
                "prompt": prompts_list,
            }
            return new_samples

        def eval_filter_func(x):
            return len(x["prompt"]) <= script_args.max_prompt_length

        eval_prompt_dataset = utils.load_and_process_dataset(
            script_args.dataset_eval_prompt_path,
            script_args.dataset_eval_prompt_name,
            script_args.dataset_eval_prompt_split,
            hub_token=hub_token,
            preprocess_function=eval_alpaca_preprocess_function,
            filter_function=eval_filter_func,
            seed=training_args.data_seed,
            percentage=script_args.dataset_eval_prompt_percentage,
            batched=True,
            name="eval_prompt",
        )

    dpo_trainer = DPOTrainer(
        model,
        model_ref,
        args=training_args,
        beta=script_args.beta,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        max_length=script_args.max_length,
        max_target_length=script_args.max_target_length,
        max_prompt_length=script_args.max_prompt_length,
        generate_during_eval=False,
    )

    proxy_reward_pipe = None
    proxy_reward_format_dict = None
    if script_args.proxy_reward_model_path is not None:
        proxy_reward_pipe = utils.create_reward_pipeline(
            script_args.proxy_reward_model_path,
            dpo_trainer.accelerator.device,
            reward_model_type=script_args.proxy_reward_model_type,
            reward_model_subfolder=script_args.proxy_reward_model_subfolder,
            hub_token=hub_token,
        )
        assert script_args.proxy_reward_model_format_json is not None, "reward_model_format_json must be specified"
        proxy_reward_format_dict = utils.load_format_dict_from_json(script_args.proxy_reward_model_format_json)
        logger.info(f"proxy_reward_format_dict: {proxy_reward_format_dict}")

    if script_args.gold_reward_model_path is not None:
        gold_reward_pipe = utils.create_reward_pipeline(
            script_args.gold_reward_model_path,
            dpo_trainer.accelerator.device,
            reward_model_type=script_args.gold_reward_model_type,
            reward_model_subfolder=script_args.gold_reward_model_subfolder,
            hub_token=hub_token,
        )
        assert script_args.gold_reward_model_format_json is not None, "reward_model_format_json must be specified"
        gold_reward_format_dict = utils.load_format_dict_from_json(script_args.gold_reward_model_format_json)
        logger.info(f"gold_reward_format_dict: {gold_reward_format_dict}")

        def collator(data):
            return dict((key, [d[key] for d in data]) for key in data[0])

        eval_gold_reward_model_callback = EvalGoldRewardModelCallback(
            dpo_trainer=dpo_trainer,
            eval_prompt_dataset=eval_prompt_dataset,
            script_args=script_args,
            gold_reward_pipe=gold_reward_pipe,
            gold_reward_format_dict=gold_reward_format_dict,
            collator=collator,
        )

        dpo_trainer.add_callback(eval_gold_reward_model_callback)

    if script_args.filtered_train_dataset:
        assert proxy_reward_pipe is not None, "proxy_reward_pipe must be specified"
        reward_model_pipe = proxy_reward_pipe
        reward_format_dict = proxy_reward_format_dict
        filtered_train_dataset_callback = FilterTrainDatasetCallback(
            dpo_trainer=dpo_trainer,
            script_args=script_args,
            reward_model_pipe=reward_model_pipe,
            reward_format_dict=reward_format_dict,
            threshold=script_args.filtered_threshold,
            filtered_start_epoch=script_args.filtered_start_epoch,
        )
        dpo_trainer.add_callback(filtered_train_dataset_callback)

    if script_args.init_eval:
        eval_gold_reward_model_callback.evauate()

    if script_args.filtered_train_dataset:
        num_train_epochs = dpo_trainer.args.num_train_epochs
        dpo_trainer.args.num_train_epochs = 1
        for epoch in range(int(num_train_epochs)):
            dpo_trainer.train()
    else:
        dpo_trainer.train()


if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    tqdm.pandas()
    main()
