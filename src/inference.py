from transformers import pipeline

checkpoint = "facebook/mbart-large-50-many-to-many-mmt"
# checkpoint = "/home/mrph/Desktop/LEARNit/facebookMBart50/facebook/mbart-large-50-many-to-many-mmt_finetuned/checkpoint-3"
translator = pipeline("translation",
                      model=checkpoint,
                      src_lang="en_XX", tgt_lang="fa_IR")
print(translator("Hello World!"))
