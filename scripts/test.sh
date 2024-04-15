
set -eu

DATASET_PATHS=(\
 "Mitsuki-Sakamoto/alpaca_farm-reward-model-deberta-v3-large-v2-re-preference-64-nsample-2-16_mix_random_seed_1"\
 "Mitsuki-Sakamoto/alpaca_farm-reward-model-deberta-v3-large-v2-re-preference-64-nsample-2-16_mix_random_seed_2"\
 "Mitsuki-Sakamoto/alpaca_farm-reward-model-deberta-v3-large-v2-re-preference-64-nsample-2-16_mix_random_seed_3"\
)
# please change the seed
seed=1
# different from 1.4b
k=$(($seed - 1))
DATASET_PATH=${DATASET_PATHS[k]}
DATASET_NAME="alpaca_instructions-pythia_160m_alpaca_farm_instructions_sft_constant_pa_seed_1"
POLICY_MODEL="Mitsuki-Sakamoto/pythia_160m_alpaca_farm_instructions_sft_constant_pa_seed_1"
PROXY_REWARD_MODEL="Mitsuki-Sakamoto/self_160m_bo16_2_mix_50_rm_pythia-70m_seed_${seed}_fixed"
GOLD_REWARD_MODEL="OpenAssistant/reward-model-deberta-v3-large-v2"
kl_coef=0.1
TRAIN_BATCH_SIZE=1
EVAL_BATCH_SIZE=1
GRADIENT_ACCUMILATION_STEPS=1

## different from dpo
EPOCHS=4
OUTPUT_DIR="output/fdpo_test_seed_${seed}"
FDPO_OPTIONS="\
--filtered_train_dataset True \
--filtered_threshold 0.0 \
--proxy_reward_model_path ${PROXY_REWARD_MODEL} \
--proxy_reward_model_format_json instruction_format_json/prompt_response_format/prompter_assistant.json \
"


COMMAND="\
poetry run python -m filtered_dpo \
--policy_model_path ${POLICY_MODEL} \
--prompt_response_format_json instruction_format_json/prompt_response_format/prompter_assistant.json \
--prompt_format_json instruction_format_json/alpaca_farm_prompt_format/ln.json \
--dataset_train_path ${DATASET_PATH} \
--dataset_train_name ${DATASET_NAME} \
--dataset_train_split preference \
--test_size 0.05 \
--dataset_eval_prompt_path tatsu-lab/alpaca_farm \
--dataset_eval_prompt_name alpaca_instructions \
--dataset_eval_prompt_split val \
--dataset_eval_prompt_percentage 1 \
--dataset_train_percentage 1 \
--gold_reward_model_path ${GOLD_REWARD_MODEL} \
--gold_reward_model_format_json instruction_format_json/prompt_response_format/deberta_sep.json \
--max_prompt_length 512 \
--max_target_length 2048 \
--max_length 2048 \
--beta ${kl_coef} \
--learning_rate 0.000001 \
--logging_steps 1 \
--eval_steps 1 \
--save_total_limit 1 \
--init_eval True \
--per_device_train_batch_size ${TRAIN_BATCH_SIZE} \
--per_device_eval_batch_size ${EVAL_BATCH_SIZE} \
--gradient_accumulation_steps ${GRADIENT_ACCUMILATION_STEPS} \
--num_train_epochs ${EPOCHS} \
--output_dir ${OUTPUT_DIR} \
--run_name test_run \
--policy_generate_max_length 64 \
--seed $seed \
--data_seed $seed \
--report_to tensorboard \
${FDPO_OPTIONS}
"
echo $COMMAND