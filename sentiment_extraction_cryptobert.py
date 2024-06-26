import pandas as pd
from transformers import TextClassificationPipeline, AutoModelForSequenceClassification, AutoTokenizer
model_name = "ElKulako/cryptobert"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels = 3)
pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer, max_length=64, truncation=True, padding = 'max_length')
crypto = 'crypto_bitcoin'
list_subs = ['bitcoin','bitcoinbeginners','bitcoinmarkets','bitcoinmining','btc']
# crypto = 'crypto_ethereum'
# list_subs = ['ethereum','ethermining','ethfinance','eth','ethtrader']
for subreddit in list_subs:
    final_df = pd.read_csv(f"Data/Cleaned_Data/{crypto}/submission_and_comments/{subreddit}_submissions_19_22.csv")
    reddit_texts = final_df['selftext'].tolist()
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
    labeled_df.to_csv(f"Data/Sentiment/{crypto}/CRYPTOBERT/{subreddit}_submission_19_22.csv",index=False)