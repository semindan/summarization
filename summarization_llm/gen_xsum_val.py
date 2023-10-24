# %%
from summarization_llm.llama import LlamaModule
from summarization_llm.modeling_llama import LlamaForCausalLM
from summarization_llm.llama_data import SumDataset
from transformers import AutoTokenizer
import transformers
from torch.utils.data import DataLoader
from peft import LoraConfig, PeftModel, PeftConfig, get_peft_model
import torch
from datasets import Dataset
from tqdm import tqdm
# %%
path = 'meta-llama/Llama-2-7b-hf'
peft_path = "/home/semindan/summarization_llm_pipeline/summarization_llm/checkpoints/vanilla7b/lora-ckpt-epoch=1-step=7293"

tokenizer = AutoTokenizer.from_pretrained(path)
tokenizer.pad_token_id = 0

bnb_4_bit_compute_type = "bfloat16"
compute_dtype = getattr(torch, bnb_4_bit_compute_type)
bnb_config = transformers.BitsAndBytesConfig(
    # load_in_4bit=True,
    load_in_8bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=False,
    bnb_4bit_compute_dtype=compute_dtype
)

model_config = transformers.AutoConfig.from_pretrained(
    path,
)
model = LlamaForCausalLM.from_pretrained(
    path,
    trust_remote_code=True,
    config=model_config,
    quantization_config=bnb_config,
    device_map="auto",
)

model = PeftModel.from_pretrained(model, peft_path)

model.config.use_cache = True  # silence the warnings. Please re-enable for inference!
model.eval()
# %%


model.eval()
def gen(batch):
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            batch_size = len(batch["input_ids"])
            original_length = batch["input_ids"].size()[1]
            outputs = model.generate(input_ids=batch['input_ids'].to("cuda"),
                                    attention_mask=batch['attention_mask'].to("cuda"),
                                    max_new_tokens = 128,
                                    min_length=55,
                                    no_repeat_ngram_size=3,
                                    repetition_penalty=1.5,
                                    num_beams=6,
                                    length_penalty=2.0,
                                    early_stopping=True)
            predictions = tokenizer.batch_decode(outputs[:, original_length:], skip_special_tokens=True)
            # candidates = [candidates[i*n_candidates:(i+1)*n_candidates] for i in range(batch_size)]
            # return {"candidates" : candidates}
            return [{"prediction" : entry, "label": label} for entry, label in zip(predictions, batch["labels"])]



def collate(data):
    text = list(map(lambda entry: entry["text"], data))
    labels = list(map(lambda entry: entry["labels"], data))
    tokenizer.padding_side = "left"
    # inputs = tokenizer(text, padding=True, max_length=1024, return_tensors="pt", truncation=True)
    inputs = tokenizer(text, max_length=2048, return_tensors="pt", truncation=True)
    inputs["labels"] = labels
    return inputs

data = SumDataset(batch_size=1, overwrite=False)
data.setup("validation")
validation_loader = DataLoader(data.validation.select(range(11302, len(data.validation))), collate_fn=collate, batch_size=1)
# ret = Dataset.from_list([el for batch in tqdm(validation_loader) for el in gen(batch)])
# ret.to_json("vanilla_validation_predictions.json", force_ascii=False)


ret = Dataset.from_list([])
for i, batch in enumerate(tqdm(validation_loader)):
    for el in gen(batch):
        ret = ret.add_item(el)
        print(el)
    if i % 100 == 0:
        ret.to_json("vanilla_validation_predictions_11302.json", force_ascii=False)

ret.to_json("vanilla_validation_predictions_11302.json", force_ascii=False)