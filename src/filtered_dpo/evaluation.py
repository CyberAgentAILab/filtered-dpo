import logging

import torch
from trl.trainer.utils import pad_to_length

logger = logging.getLogger(__name__)


def generate_responses(generate_pipeline, prompts, gen_kwargs):
    padding_side_default = generate_pipeline.tokenizer.padding_side
    generate_pipeline.tokenizer.padding_side = "left"
    outputs = generate_pipeline(prompts, **gen_kwargs)
    generate_pipeline.tokenizer.padding_side = padding_side_default
    assert all([prompt in output[0]["generated_text"] for prompt, output in zip(prompts, outputs)])
    responses = [output[0]["generated_text"].replace(prompt, "", 1) for prompt, output in zip(prompts, outputs)]
    return responses


def eval_reward_model(reward_pipe, prompt_response_dict, prompts, responses, sent_kwargs):
    text_list = [
        prompt_response_dict["prompt_response"].format_map({"prompt": prompt, "response": response})
        for prompt, response in zip(prompts, responses)
    ]
    logger.debug(f"eval_policy_model text: {text_list[0]}")

    def text_loader():
        for text in text_list:
            yield text

    eval_rewards_list = [output[0]["score"] for output in reward_pipe(text_loader(), **sent_kwargs)]
    return eval_rewards_list


def calc_kl_for_ref_model(
    model,
    ref_model,
    prompts,
    responses,
    tokenizer,
    batch_size,
    pad_value,
    label_pad_token_id,
):
    if len(prompts) % batch_size != 0:
        logger.warning("The number of prompts is not divisible by the batch size. The last batch will be smaller.")
    for i in range(0, len(prompts), batch_size):
        prompts_batch = prompts[i : i + batch_size]
        responses_batch = responses[i : i + batch_size]
        kl_list, poliocy_logps_list, ref_logps_list = _calc_kl_for_ref_model_batch(
            model,
            ref_model,
            tokenizer,
            prompts_batch,
            responses_batch,
            pad_value,
            label_pad_token_id,
        )
        if i == 0:
            kl_list_all = kl_list
            poliocy_logps_list_all = poliocy_logps_list
            ref_logps_list_all = ref_logps_list
        else:
            kl_list_all.extend(kl_list)
            poliocy_logps_list_all.extend(poliocy_logps_list)
            ref_logps_list_all.extend(ref_logps_list)
    logs_dict = {
        "kl": kl_list_all,
        "policy_logps": poliocy_logps_list_all,
        "ref_logps": ref_logps_list_all,
    }
    return logs_dict


def _calc_kl_for_ref_model_batch(
    model,
    ref_model,
    tokenizer,
    prompts_batch,
    responses_batch,
    pad_value,
    label_pad_token_id,
):
    # Convert the prompt response to input_ids and attention_mask, create labels for calculating logprob. The label is the input_ids masked at the prompt part.
    (input_ids_padded, attention_mask_padded, labels) = __prepare_inputs_padding(
        prompts_batch, responses_batch, tokenizer, pad_value, label_pad_token_id
    )

    with torch.no_grad():
        policy_logits = model(
            input_ids_padded.to(model.device),
            attention_mask=attention_mask_padded.to(model.device),
        ).logits.to(torch.float32)

        policy_logps = _get_batch_logps(
            policy_logits,
            labels.to(model.device),
            average_log_prob=False,
            label_pad_token_id=label_pad_token_id,
        )

        ref_logits = ref_model(
            input_ids_padded.to(ref_model.device),
            attention_mask=attention_mask_padded.to(ref_model.device),
        ).logits.to(torch.float32)

        ref_logps = _get_batch_logps(
            ref_logits,
            labels.to(ref_model.device),
            average_log_prob=False,
            label_pad_token_id=label_pad_token_id,
        )

        kls = (policy_logps - ref_logps).detach()

    kl_list = kls.cpu().numpy().tolist()
    poliocy_logps_list = policy_logps.cpu().numpy().tolist()
    ref_logps_list = ref_logps.cpu().numpy().tolist()
    return kl_list, poliocy_logps_list, ref_logps_list


# WARN: this function is used only in calc_kl_for_ref_model
def __prepare_inputs_padding(prompts, responses, tokenizer, pad_value, label_pad_token_id):
    prompt_input_ids_list = [
        tokenizer(prompt, add_special_tokens=False, return_tensors="pt")["input_ids"].squeeze(0) for prompt in prompts
    ]
    response_input_ids_list = [
        tokenizer(response, add_special_tokens=False, return_tensors="pt")["input_ids"].squeeze(0)
        for response in responses
    ]
    input_ids_list = [
        torch.cat([prompt_input_ids, response_input_ids])
        for prompt_input_ids, response_input_ids in zip(prompt_input_ids_list, response_input_ids_list)
    ]
    attention_masks_list = [torch.ones_like(input_ids) for input_ids in input_ids_list]
    labels_list = [input_ids.clone() for input_ids in input_ids_list]
    # Create input_ids that maskthe prompt part
    for i_l, prompt_input_ids in enumerate(prompt_input_ids_list):
        labels_list[i_l][: prompt_input_ids.shape[-1]] = label_pad_token_id

    # padding
    max_input_length = max([input_ids.shape[-1] for input_ids in input_ids_list])
    input_ids_padded = torch.stack(
        [pad_to_length(input_id, max_input_length, pad_value=pad_value) for input_id in input_ids_list],
        dim=0,
    ).to(torch.long)

    attention_mask_padded = torch.stack(
        [
            pad_to_length(attention_mask, max_input_length, pad_value=pad_value)
            for attention_mask in attention_masks_list
        ],
        dim=0,
    ).to(torch.long)
    labels_padded = torch.stack(
        [pad_to_length(label, max_input_length, pad_value=label_pad_token_id) for label in labels_list],
        dim=0,
    ).to(torch.long)
    return input_ids_padded, attention_mask_padded, labels_padded


# https://github.com/huggingface/trl/blob/a78a05d7b76eb1b4dade9c5c1d4704cceaf8a5b1/trl/trainer/dpo_trainer.py#L459
def _get_batch_logps(
    logits: torch.FloatTensor,
    labels: torch.LongTensor,
    label_pad_token_id: int,
    average_log_prob: bool = False,
) -> torch.FloatTensor:
    """Compute the log probabilities of the given labels under the given logits.

    Args:
        logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
        labels: Labels for which to compute the log probabilities. Label tokens with a value of label_pad_token_id are
        ignored. Shape: (batch_size, sequence_length)
        average_log_prob: If True, return the average log probability per (non-masked) token. Otherwise, return the
        sum of the log probabilities of the (non-masked) tokens.

    Returns:
        A tensor of shape (batch_size,) containing the average/sum log probabilities of the given labels under the
        given logits.
    """
    if logits.shape[:-1] != labels.shape:
        raise ValueError("Logits (batch and sequence length dim) and labels must have the same shape.")

    labels = labels[:, 1:].clone()
    logits = logits[:, :-1, :]
    loss_mask = labels != label_pad_token_id
    # dummy token; we'll ignore the losses on these tokens later
    labels[labels == label_pad_token_id] = 0

    per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

    if average_log_prob:
        return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
    else:
        return (per_token_logps * loss_mask).sum(-1)
