from transformers import pipeline

# checkpoint = "facebook/mbart-large-50-many-to-many-mmt"
# checkpoint = "/home/mrph/Desktop/LEARNit/facebookMBart50/saved_model/checkpoint-57465"
# checkpoint = "/home/mrph/Desktop/LEARNit/facebookMBart50/saved_model/checkpoint-114930"
checkpoint_path = "/home/mrph/Desktop/LEARNit/facebookMBart50/saved_model/checkpoint-229860"


# checkpoint_path = "/root/facebookMBart50/saved_model/checkpoint-884954"


class Inference:
    def __init__(self, checkpoint):
        self.checkpoint = checkpoint

    def translate(self, txt):
        translator = pipeline("translation",
                              model=self.checkpoint,
                              src_lang="en_XX",
                              tgt_lang="fa_IR")
        return translator(txt)[0]["translation_text"]


if __name__ == "__main__":
    text = "trying not to laugh, but I didn't understand"
    model = Inference(checkpoint_path)
    print(model.translate(text))
