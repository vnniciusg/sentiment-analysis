if __name__ == "__main__":

    __import__("warnings").filterwarnings("ignore")

    from datasets import load_dataset
    from nlpprepkit import TextPreprocessor

    dataset = load_dataset("sentiment140", trust_remote_code=True)

    text_preprocessor = TextPreprocessor()

    def preprocess_batch(batch):
        return {"clean_text": text_preprocessor.process_text(batch["text"])}

    processed_dataset = dataset.map(preprocess_batch, batched=True, desc="preprocessing")

    print(processed_dataset["train"][0])
