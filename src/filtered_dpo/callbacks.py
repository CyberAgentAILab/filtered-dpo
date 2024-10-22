import copy
import logging
import os

import numpy as np
import torch
from tqdm import tqdm
from transformers import TrainerCallback, pipeline
from transformers.integrations import MLflowCallback
from transformers.trainer_callback import TrainerControl, TrainerState
from transformers.training_args import TrainingArguments
from trl import AutoModelForCausalLMWithValueHead, DPOTrainer, PPOConfig, PPOTrainer
from trl.core import logprobs_from_logits
from trl.trainer.utils import pad_to_length

from . import evaluation, utils

logger = logging.getLogger(__name__)


class EvalGoldRewardModelCallback(TrainerCallback):

    def __init__(
        self,
        dpo_trainer,
        eval_prompt_dataset,
        script_args,
        gold_reward_pipe=None,
        gold_reward_format_dict=None,
        collator=None,
    ):
        super().__init__()
        self.dpo_trainer = dpo_trainer
        self.eval_prompt_dataset = eval_prompt_dataset
        self.script_args = script_args
        self.gold_reward_pipe = gold_reward_pipe
        self.gold_reward_format_dict = gold_reward_format_dict
        self.collator = collator

    def update_logs_dict_extended(self, logs_dict, new_logs_dict):
        for key, value_list in new_logs_dict.items():
            if key not in logs_dict:
                logs_dict[key] = value_list
            else:
                logs_dict[key].extend(value_list)
        return logs_dict

    def evauate(self):
        tokenizer = self.dpo_trainer.tokenizer
        script_args = self.script_args
        eval_prompt_dataloader = torch.utils.data.DataLoader(
            self.eval_prompt_dataset,
            batch_size=script_args.per_device_eval_batch_size,
            collate_fn=self.collator,
            shuffle=True,
            drop_last=True,
        )
        eval_prompt_dataloader = self.dpo_trainer.accelerator.prepare(eval_prompt_dataloader)

        gen_kwargs = {
            "top_k": script_args.top_k,
            "top_p": script_args.top_p,
            "temperature": script_args.temperature,
            "do_sample": True,
            "pad_token_id": tokenizer.eos_token_id,
            "max_new_tokens": script_args.policy_generate_max_length,
            "batch_size": script_args.per_device_eval_batch_size,
        }
        sent_kwargs = {
            "return_all_scores": True,
            "function_to_apply": "none",
            "batch_size": script_args.reward_model_bath_size,
        }
        # check why wrappeing is needed
        model = self.dpo_trainer._wrap_model(self.dpo_trainer.model, training=False, dataloader=None).eval()
        ref_model = self.dpo_trainer.ref_model
        generate_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device=self.dpo_trainer.accelerator.device,
        )
        with torch.no_grad():
            logs_dict = {}
            for step, inputs in enumerate(tqdm(eval_prompt_dataloader)):
                logger.info("generate responses...")
                responses = evaluation.generate_responses(
                    generate_pipeline,
                    inputs["prompt"],
                    gen_kwargs,
                )
                logger.debug(inputs["prompt"][0], responses[0])
                response_lengths = [len(response) for response in responses]
                self.update_logs_dict_extended(logs_dict, {"response_length": response_lengths})
                logger.info("calculate kl...")
                kl_logs_dict = evaluation.calc_kl_for_ref_model(
                    model=model,
                    ref_model=ref_model,
                    tokenizer=tokenizer,
                    prompts=inputs["prompt"],
                    responses=responses,
                    batch_size=script_args.per_device_eval_batch_size,
                    pad_value=self.dpo_trainer.padding_value,
                    label_pad_token_id=self.dpo_trainer.label_pad_token_id,
                )
                self.update_logs_dict_extended(logs_dict, kl_logs_dict)
                if self.gold_reward_pipe is not None:
                    logger.info("eval gold reward model...")
                    gold_rewards = evaluation.eval_reward_model(
                        reward_pipe=self.gold_reward_pipe,
                        prompt_response_dict=self.gold_reward_format_dict,
                        prompts=inputs["unformatted_prompt"],
                        responses=responses,
                        sent_kwargs=sent_kwargs,
                    )
                    self.update_logs_dict_extended(logs_dict, {"gold_reward": gold_rewards})
            logs = {}
            logs["num_samples"] = len(logs_dict["kl"])
            for name, value_list in logs_dict.items():
                met_array = np.array(value_list)
                logs[f"eval/{name}_mean"] = met_array.mean().item()
                logs[f"eval/{name}_std"] = met_array.std().item()
                logs[f"eval/{name}_max"] = met_array.max().item()
                logs[f"eval/{name}_min"] = met_array.min().item()
                logger.info(
                    f"{name}: {logs[f'eval/{name}_mean']}, std: {logs[f'eval/{name}_std']},"
                    f"max: {logs[f'eval/{name}_max']}, min: {logs[f'eval/{name}_min']}"
                )
            if len(logs) > 0:
                self.dpo_trainer.log(logs)
                self.dpo_trainer.state.log_history.pop()

    def on_evaluate(self, args, state, control, **kwargs):
        self.evauate()
        return super().on_evaluate(args, state, control, **kwargs)


class FilterTrainDatasetCallback(TrainerCallback):

    def __init__(
        self,
        dpo_trainer,
        script_args,
        reward_model_pipe,
        reward_format_dict,
        threshold,
        filtered_start_epoch,
    ):
        super().__init__()
        self.epoch = 0
        self.dpo_trainer = dpo_trainer
        self.script_args = script_args
        self.reward_model_pipe = reward_model_pipe
        self.reward_format_dict = reward_format_dict
        self.threshold = threshold
        self.filtered_start_epoch = filtered_start_epoch
        self.state = None
        if "index" not in self.dpo_trainer.train_dataset.column_names:
            self.dpo_trainer.train_dataset = self.dpo_trainer.train_dataset.add_column(
                "index", range(len(self.dpo_trainer.train_dataset))
            )
        self.filtered_dataset = copy.deepcopy(dpo_trainer.train_dataset)
        self.filtered_dataset = self.filtered_dataset.add_column("filtered_epoch", [-1] * len(self.filtered_dataset))
        self.base_run_name = copy.deepcopy(dpo_trainer.args.run_name)
        dpo_trainer.args.run_name = f"{self.base_run_name}_epoch_{self.epoch}"
        if filtered_start_epoch == 0:
            logger.info(f"filtering dataset on epoch {self.epoch}")
            self.filter_dataset()
            logger.info(f"save filtered dataset on epoch {self.epoch} to {dpo_trainer.args.output_dir}")
            self.filtered_dataset.save_to_disk(f"{dpo_trainer.args.output_dir}/filtered_dataset_epoch_{self.epoch}") 
            
    def filter_dataset(self):
        logger.info("filtering dataset...")
        logger.info(f"train_dataset size: {len(self.dpo_trainer.train_dataset)}")
        # Generate responses
        model = self.dpo_trainer._wrap_model(self.dpo_trainer.model, training=False, dataloader=None).eval()
        generate_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=self.dpo_trainer.tokenizer,
            device=self.dpo_trainer.accelerator.device,
        )
        gen_kwargs = {
            "top_k": self.script_args.filtered_top_k,
            "top_p": self.script_args.filtered_top_p,
            "temperature": self.script_args.filtered_temperature,
            "do_sample": True,
            "pad_token_id": self.dpo_trainer.tokenizer.eos_token_id,
            "max_new_tokens": self.script_args.policy_generate_max_length,
            "batch_size": self.script_args.per_device_eval_batch_size,
        }
        sent_kwargs = {
            "return_all_scores": True,
            "function_to_apply": "none",
            "batch_size": self.script_args.reward_model_bath_size,
        }
        responses = evaluation.generate_responses(
            generate_pipeline,
            self.dpo_trainer.train_dataset["prompt"],
            gen_kwargs,
        )
        # Evaluate reward model for dataset and responses
        dataset_reward_list = evaluation.eval_reward_model(
            reward_pipe=self.reward_model_pipe,
            prompt_response_dict=self.reward_format_dict,
            prompts=self.dpo_trainer.train_dataset["unformatted_prompt"],
            responses=self.dpo_trainer.train_dataset["chosen"],
            sent_kwargs=sent_kwargs,
        )
        response_reward_list = evaluation.eval_reward_model(
            reward_pipe=self.reward_model_pipe,
            prompt_response_dict=self.reward_format_dict,
            prompts=self.dpo_trainer.train_dataset["unformatted_prompt"],
            responses=responses,
            sent_kwargs=sent_kwargs,
        )

        diff_reward_dict_for_train_dataset = {
            example["index"]: response_reward - dataset_reward
            for example, dataset_reward, response_reward in zip(
                self.dpo_trainer.train_dataset,
                dataset_reward_list,
                response_reward_list,
            )
        }
        respoce_dict_for_train_dataset = {
            example["index"]: response for example, response in zip(self.dpo_trainer.train_dataset, responses)
        }

        def filter_for_reward(example):
            if example["index"] in diff_reward_dict_for_train_dataset:
                return diff_reward_dict_for_train_dataset[example["index"]] < self.threshold
            return False

        def processing_dataset_for_reward(example):
            if example["index"] in diff_reward_dict_for_train_dataset:
                filtered = filter_for_reward(example)
                example["filtered_epoch"] = self.epoch if not filtered else example["filtered_epoch"]
                example["gen_reward"] = diff_reward_dict_for_train_dataset[example["index"]]
                example["gen_response"] = respoce_dict_for_train_dataset[example["index"]]
            return example

        logger.info(f"Before filtering dataset {self.dpo_trainer.train_dataset}")
        self.dpo_trainer.train_dataset = self.dpo_trainer.train_dataset.filter(filter_for_reward)
        logger.info(f"After filtering dataset {self.dpo_trainer.train_dataset}")
        self.filtered_dataset = self.filtered_dataset.map(processing_dataset_for_reward)

    def _load_state(self, state):
        if self.state is None:
            return
        for name, item in vars(state).items():
            if name == "max_steps":
                sum_max_steps = getattr(self.state, name) + item
                setattr(self.state, name, sum_max_steps)
            setattr(state, name, getattr(self.state, name))

    def _save_state(self, state):
        self.state = copy.deepcopy(state)

    def on_epoch_begin(self, args, state, control, model=None, **kwargs):
        self._load_state(state)
        if self.epoch > 0:
            control.should_training_stop = False
            control.should_epoch_stop = False
        return super().on_epoch_begin(args, state, control, **kwargs)

    def on_epoch_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        self.epoch += 1
        self._save_state(state)
        if self.epoch >= self.filtered_start_epoch:
            logger.info(f"filtering dataset on epoch {self.epoch}")
            self.filter_dataset()
            logger.info(f"save filtered dataset on epoch {self.epoch} to {args.output_dir}")
            self.filtered_dataset.save_to_disk(f"{args.output_dir}/filtered_dataset_epoch_{self.epoch}")
        return super().on_epoch_end(args, state, control, **kwargs)

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        state.epoch += self.epoch
        return super().on_step_end(args, state, control, **kwargs)
