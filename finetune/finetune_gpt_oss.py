import wandb
import torch
from datasets import load_dataset

from transformers import AutoModelForCausalLM, Mxfp4Config, AutoTokenizer
from peft import LoraConfig, get_peft_model
from trl import SFTConfig, SFTTrainer

from openai_harmony import (
    load_harmony_encoding, HarmonyEncodingName,
    Conversation, Message, Role,
    SystemContent, DeveloperContent, ReasoningEffort
)

# Initialize wandb
# wandb.init(
#     project="LLM-Recipe",
#     entity="sayannath235",
#     name="GPT-OSS:20B-MedicalDataset",
# )

quantization_config = Mxfp4Config(dequantize=True)
model_kwargs = dict(
    attn_implementation="eager",
    dtype=torch.bfloat16,
    quantization_config=quantization_config,
    use_cache=False,
    device_map="auto",
)

model = AutoModelForCausalLM.from_pretrained("openai/gpt-oss-20b", **model_kwargs)
tokenizer = AutoTokenizer.from_pretrained("openai/gpt-oss-20b")

enc = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)

def render_pair_harmony(question, answer):
    convo = Conversation.from_messages([
        Message.from_role_and_content(
            Role.DEVELOPER,
            DeveloperContent.new().with_instructions(
                "You are a medical expert with advanced knowledge in clinical reasoning and diagnostics. "
                "Respond with ONLY the final diagnosis/cause in ≤5 words."
            )
        ),
        Message.from_role_and_content(Role.USER, question.strip()),
        Message.from_role_and_content(Role.ASSISTANT, answer.strip()),
    ])
    tokens = enc.render_conversation(convo)
    text = enc.decode(tokens)
    return text

def prompt_style_harmony(examples):
    qs = examples["Open-ended Verifiable Question"]
    ans = examples["Ground-True Answer"]
    outputs = {"text": []}
    for q, a in zip(qs, ans):
        rendered = render_pair_harmony(q, a)
        outputs["text"].append(rendered)
    return outputs

dataset = load_dataset(
    "FreedomIntelligence/medical-o1-verifiable-problem",
    split="train"
)
dataset = dataset.map(prompt_style_harmony, batched=True)

print("Column names", dataset.column_names)
print("Text Sample: ", dataset[0]["text"])

def render_inference_harmony(question):
    convo = Conversation.from_messages([
        Message.from_role_and_content(
            Role.DEVELOPER,
            DeveloperContent.new().with_instructions(
                "You are a medical expert with advanced knowledge in clinical reasoning and diagnostics. "
                "Respond with ONLY the final diagnosis/cause in ≤5 words."
            )
        ),
        Message.from_role_and_content(Role.USER, question.strip()),
    ])
    tokens = enc.render_conversation_for_completion(convo, Role.ASSISTANT)
    text = enc.decode(tokens)
    return text

question = dataset[0]["Open-ended Verifiable Question"]

text = render_inference_harmony(question)

inputs = tokenizer(
    [text + tokenizer.eos_token], return_tensors="pt"
).to("cuda")
outputs = model.generate(
    input_ids=inputs.input_ids,
    attention_mask=inputs.attention_mask,
    max_new_tokens=20,
    eos_token_id=tokenizer.eos_token_id,
    use_cache=True,
)
response = tokenizer.batch_decode(outputs)
print("Predicted Label: ", response[0])
print("Ground Truth Label: ", dataset[0]["Ground-True Answer"])

peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules="all-linear",
    target_parameters=[
        "7.mlp.experts.gate_up_proj",
        "7.mlp.experts.down_proj",
        "15.mlp.experts.gate_up_proj",
        "15.mlp.experts.down_proj",
        "23.mlp.experts.gate_up_proj",
        "23.mlp.experts.down_proj",
    ],
)
peft_model = get_peft_model(model, peft_config)
print(peft_model.print_trainable_parameters())

training_args = SFTConfig(
    learning_rate=2e-4,
    gradient_checkpointing=True,
    num_train_epochs=1,
    logging_steps=10,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    max_length=2048,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine_with_min_lr",
    lr_scheduler_kwargs={"min_lr_rate": 0.1},
    output_dir="gpt-oss-20b-medical-qa",
    report_to="wandb",
    push_to_hub=True,
)

trainer = SFTTrainer(
    model=peft_model,
    args=training_args,
    train_dataset=dataset,
    processing_class=tokenizer,
)
trainer.train()

trainer.save_model(training_args.output_dir)