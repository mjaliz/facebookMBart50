from transformers import pipeline
import torch

print(torch.cuda.is_available())

# checkpoint = "facebook/mbart-large-50-many-to-many-mmt"
# checkpoint = "/home/mrph/Desktop/LEARNit/facebookMBart50/saved_models/checkpoint-57465"
# checkpoint = "/home/mrph/Desktop/LEARNit/facebookMBart50/saved_models/checkpoint-114930"
# checkpoint = "/home/mrph/Desktop/LEARNit/facebookMBart50/saved_models/checkpoint-229860"
checkpoint = "/root/facebookMBart50/saved_model/checkpoint-884954"
translator = pipeline("translation",
                      model=checkpoint,
                      src_lang="en_XX", tgt_lang="fa_IR")
print(translator("trying not to laugh, but I didn't understand"))
