import pandas as pd
from transformers import TextClassificationPipeline, AutoModelForSequenceClassification, AutoTokenizer
model_name = "ElKulako/cryptobert"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels = 3)
pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer, max_length=64, truncation=True, padding = 'max_length')
# list_subs = ['bitcoin','bitcoinbeginners','bitcoinmarkets','bitcoinmining','btc']
list_subs = ['ethereum','ethermining','ethfinance','eth','ethtrader']
for subreddit in list_subs:
    final_df = pd.read_csv(f"{subreddit}_comment.csv")
    reddit_texts = final_df['body'].tolist()
    preds = pipe(reddit_texts)
    for element in preds:
        label = element['label']
        if label == "Bearish":
            element['label']="Negative"
        elif label == 'Bullish':
            element['label']="Positive"
        else:
            element['label']="Neutral"
    new_data_df = pd.DataFrame(preds)
    print(f"{subreddit} scores min: ",new_data_df.score.min(),"max: ",new_data_df.score.max())
    labeled_df = pd.concat([final_df, new_data_df], axis=1)
    labeled_df.to_csv(f"{subreddit}_comment_sentiment.csv",index=False)