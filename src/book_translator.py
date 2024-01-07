from transformers import pipeline
import os
import pandas as pd

data_dir = "/home/mrph/Desktop/LEARNit/facebookMBart50/Muhammad_Ali-B_Smith_csvs"
csvs = [file for file in os.listdir(data_dir) if file.endswith(".csv")]
df = pd.concat(
    (pd.read_csv(os.path.join(data_dir, file)) for file in sorted(os.listdir(data_dir)) if file.endswith(".csv")),
    ignore_index=True)
print(df)
# checkpoint = "facebook/mbart-large-50-many-to-many-mmt"
# checkpoint = "/home/mrph/Desktop/LEARNit/facebookMBart50/saved_model/checkpoint-57465"
# checkpoint = "/home/mrph/Desktop/LEARNit/facebookMBart50/saved_model/checkpoint-114930"
# checkpoint = "/home/mrph/Desktop/LEARNit/facebookMBart50/saved_model/checkpoint-229860"
checkpoint = "/home/mrph/Desktop/LEARNit/facebookMBart50/saved_model/checkpoint-695321-e11/checkpoint-695321"
translator = pipeline("translation",
                      model=checkpoint,
                      src_lang="en_XX", tgt_lang="fa_IR")


def translate(txt):
    translation = translator(txt)[0]["translation_text"]
    print(f"Translation: {txt} => {translation}")
    return translation


df["translated_text"] = df["sentence"].apply(translate)
df.to_csv("muhammad.csv")
print(translator(
    'Tour through our photo gallery of Pacino’s 25 greatest films, ranked worst to best.'))
