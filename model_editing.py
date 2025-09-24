import torch, transformers
import json
import pyreft
from peft import LoraConfig, get_peft_model
from tqdm import tqdm
from typing import Dict
import copy
import datasets
import io, contextlib, re
import time
import sys

train_domain = sys.argv[1]
# test_domain = sys.argv[2]

IGNORE_INDEX = -100
# layers_to_transform = range(32)
fs_layer = int(sys.argv[2])
fn_layer = int(sys.argv[3])
is_weight_editing = sys.argv[4].lower() == 'true'
learning_rate = 2e-5
layers_to_transform = range(fs_layer, fn_layer)
aspect_suffix_index = 2  # This is based on the position of aspect in the prompt
intervention_size = len(layers_to_transform)

parameters_dict = {}


def get_parameters(input_text):
    pattern = re.compile(
        r"trainable intervention params: ([\d,]+) \|\| trainable model params: ([\d,]+)\nmodel params: ([\d,]+) \|\| trainable%: ([\d.]+)")
    match = pattern.search(input_text)

    if match:
        trainable_intervention_parameters = int(match.group(1).replace(',', ''))
        trainable_model_parameters = int(match.group(2).replace(',', ''))
        all_model_parameters = int(match.group(3).replace(',', ''))
        trainable_percentage = float(match.group(4))

        # Print the extracted variables
        return {
            "trainable_intervention_parameters": trainable_intervention_parameters,
            "trainable_model_parameters": trainable_model_parameters,
            "all_model_parameters": all_model_parameters,
            "trainable_percentage": trainable_percentage
        }
    else:
        return None


def make_last_position_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, model, inputs, outputs,
                                              nonstop=False) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""

    all_base_input_ids, all_intervention_locations, all_output_ids = [], [], []
    for i in range(len(inputs)):
        _input = inputs[i]
        _output = outputs[i]

        base_prompt = _input
        base_input = base_prompt + _output
        if not nonstop:
            base_input += tokenizer.eos_token

        # tokenize
        base_prompt_ids = tokenizer(
            base_prompt, max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")["input_ids"][0]
        base_prompt_length = len(base_prompt_ids)
        base_input_ids = tokenizer(
            base_input, max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")["input_ids"][0]
        output_ids = copy.deepcopy(base_input_ids)
        output_ids[:base_prompt_length] = IGNORE_INDEX

        all_base_input_ids.append(base_input_ids)
        all_intervention_locations.append([[round(
            base_prompt_length) - aspect_suffix_index]] * intervention_size)  # intervention location based on aspect suffix index
        all_output_ids.append(output_ids)

    train_dataset = datasets.Dataset.from_dict({
        "input_ids": all_base_input_ids,
        "intervention_locations": all_intervention_locations,
        "labels": all_output_ids,
    })

    data_collator_fn = transformers.DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
        padding="longest"
    )

    data_collator = pyreft.ReftDataCollator(data_collator=data_collator_fn)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)


def read_data(train_data_path_in):
    train_data_sentence = []
    labels = []
    with open(train_data_path_in, 'r') as data_read:
        for item in data_read.readlines():
            data = json.loads(item)
            sentence_format = "Given the sentence: '{}', what is the sentiment polarity of the aspect term {} ?".format(
                data['sentence'], data['aspect'])
            label = polarity_dic[data['label']]
            train_data_sentence.append(sentence_format)
            labels.append(label)
    return train_data_sentence, labels


def format_floats(d):
    for key, value in d.items():
        if isinstance(value, float):
            d[key] = round(value, 3)
    return d


train_domains = ['device', 'laptop', 'rest', 'service']
test_domains = ['device', 'laptop', 'rest', 'service']

model_name_or_path = " "
model = transformers.AutoModelForCausalLM.from_pretrained(
    model_name_or_path, torch_dtype=torch.bfloat16, device_map="cuda")

# get tokenizer
tokenizer = transformers.AutoTokenizer.from_pretrained(
    model_name_or_path, model_max_length=512,
    padding_side="right", use_fast=False)
tokenizer.pad_token = tokenizer.unk_token

include_peft = True

if is_weight_editing:
    peft_config = LoraConfig(
        r=4,
        lora_alpha=8,
        target_modules=["o_proj"],
        layers_to_transform=layers_to_transform,
        use_rslora=True,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, peft_config)

reft_config = pyreft.ReftConfig(representations=[{
    # string component access is enforced for customized model such as a peft model!
    "layer": l,
    "component": f"base_model.model.model.layers[{l}].output" if include_peft else "block_output",
    "low_rank_dimension": 4,
    "intervention": pyreft.LoreftIntervention(embed_dim=model.config.hidden_size,
                                              low_rank_dimension=4)} for l in layers_to_transform])

reft_model = pyreft.get_reft_model(model, reft_config)

# you need to call this to re-enable lora grads!
if include_peft:
    reft_model.model.enable_adapter_layers()

reft_model.print_trainable_parameters()
# Capture output
output_buffer = io.StringIO()
with contextlib.redirect_stdout(output_buffer):
    reft_model.print_trainable_parameters()
# Get the captured output
output = output_buffer.getvalue()
parameters_dict.update(get_parameters(output))

train_data_path = f'data/{train_domain}_train.json'
polarity_dic = {'Pos': 'positive', 'Neg': 'negative', 'Neu': "neutral"}

train_data, labels = read_data(train_data_path)

# data_module = pyreft.make_last_position_supervised_data_module(tokenizer, model, train_data, labels)
data_module = make_last_position_supervised_data_module(tokenizer, model, train_data, labels)

# train
training_start_time = time.time()
training_args = transformers.TrainingArguments(
    num_train_epochs=1, output_dir=" ", per_device_train_batch_size=2,
    learning_rate=learning_rate, logging_steps=20, report_to=[])
trainer = pyreft.ReftTrainerForCausalLM(
    model=reft_model, tokenizer=tokenizer, args=training_args, **data_module)
_ = trainer.train()
training_end_time = time.time()
training_time = training_end_time - training_start_time
parameters_dict['train_time'] = training_time

for test_domain in test_domains:
    print(f"\ntrain_domain:{train_domain},test_domain:{test_domain}\n")
    test_data_path = f'data/{test_domain}_test.json'
    # ensure everything is in eval mode
    eval_start_time = time.time()
    reft_model.model.eval()

    for k, v in reft_model.interventions.items():
        if hasattr(v, 'eval'):
            _ = v.eval()  # New version: directly call eval
        elif isinstance(v, (list, tuple)) and len(v) > 0:
            _ = v[0].eval()  # Old version: access through index

    acc = 0
    with open(test_data_path, 'r') as read_file:
        sentences = read_file.readlines()
        for item in tqdm(sentences):
            data = json.loads(item)
            sentence_format = "Given the sentence: '{}', what is the sentiment polarity of the aspect term {} ?".format(
                data['sentence'], data['aspect'])
            label = polarity_dic[data['label']]

            # tokenize and prepare the input
            # prompt = prompt_no_input_template % instruction

            prompt = tokenizer(sentence_format, return_tensors="pt").to("cuda")

            base_unit_location = round(prompt["input_ids"].shape[
                                           -1] - aspect_suffix_index)  # intervention location based on aspect suffix index
            _, reft_response = reft_model.generate(
                prompt, unit_locations={"sources->base": (None, [[[base_unit_location]]] * intervention_size)},
                intervene_on_prompt=True, max_new_tokens=10, do_sample=True,
                eos_token_id=tokenizer.eos_token_id, early_stopping=True
            )
            output_text = tokenizer.decode(reft_response[0], skip_special_tokens=True)
            # print(output_text)
            new_generated_text = output_text[len(sentence_format):].strip()
            # print(new_generated_text)
            if new_generated_text == label:
                acc += 1
    eval_end_time = time.time()
    eval_time = eval_end_time - eval_start_time
    test_acc = acc / len(sentences)
    parameters_dict['inference_time'] = eval_time
    parameters_dict['test_acc'] = test_acc
    parameters_dict['transformer_layers'] = list(layers_to_transform)
    parameters_dict['train_data_path'] = train_data_path
    parameters_dict['test_data_path'] = test_data_path
    parameters_dict['lr'] = str(learning_rate)
    print(acc / len(sentences))

    parameters_dict = format_floats(parameters_dict)
    print(parameters_dict)
    with open(f'results/{train_domain}_{test_domain}_{fs_layer}_{fn_layer - 1}_*.json',
              'w') as wt_file:
        json.dump(parameters_dict, wt_file, indent=4)
