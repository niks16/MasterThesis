import pandas as pd
from transformers import TextClassificationPipeline, AutoModelForSequenceClassification, AutoTokenizer
model_name = "ElKulako/cryptobert"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels = 3)
pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer, max_length=64, truncation=True, padding = 'max_length')
reddit_texts = ['bitcoin','bitcoinbeginners','bitcoinmarkets','bitcoinmining','btc']
preds = pd.DataFrame(pipe(reddit_texts))
preds.to_csv("crypto_bitcoin submission_sentiment.csv",index=False)
