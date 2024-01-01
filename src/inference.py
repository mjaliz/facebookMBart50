from transformers import pipeline

# checkpoint = "facebook/mbart-large-50-many-to-many-mmt"
# checkpoint = "/home/mrph/Desktop/LEARNit/facebookMBart50/saved_models/checkpoint-57465"
# checkpoint = "/home/mrph/Desktop/LEARNit/facebookMBart50/saved_models/checkpoint-114930"
checkpoint = "/home/mrph/Desktop/LEARNit/facebookMBart50/saved_models/checkpoint-229860"
translator = pipeline("translation",
                      model=checkpoint,
                      src_lang="en_XX", tgt_lang="fa_IR")
print(translator("trying not to laugh, but I didn't understand"))
