import json
import pandas as pd
import regex
import emoji
import sys

crypto = ['crypto_bitcoin', 'crypto_ethereum']
list_subs = ['bitcoin','bitcoinbeginners','bitcoinmarkets','bitcoinmining','btc','ethereum','ethermining','ethfinance','eth','ethtrader']


        
def clean_df(df):
    df = df.drop_duplicates()
    df.dropna(inplace=True)
    # df['selftext'] = df['selftext'].str.lower()
    df['selftext'] = df['selftext'].replace(r'http\S+', '', regex=True).replace(r'www\S+', '', regex=True) #removes all URLs
    df['selftext'] = df['selftext'].replace(r'[@][A-Za-z0-9_]+', '', regex=True).replace(r'[#][A-Za-z0-9_]+', '', regex=True) #removes various types of mentions, hashtags
    df['selftext'] = df['selftext'].replace(r'&amp;', 'and', regex=True).replace(r'&amp', 'and', regex=True) #replaces the "&" HTML entity with the word "and"
    df['selftext'] = df['selftext'].replace(r'â€™', '\'', regex=True).replace(r'&#39;', '\'', regex=True).replace(r'&#x200B;', '', regex=True).replace(r'&;', '\'', regex=True) #replaces HTML entities and special characters with corresponding characters or spaces.
    df['selftext'] = df['selftext'].replace(r'\s+', ' ', regex=True) #replaces multiple consecutive spaces with a single space
    df['selftext'] = df['selftext'].replace(r'&quot;', '', regex=True) #removes '&quot;'  
    df.dropna(inplace=True)
    # df['selftext'] = df['selftext'].apply(lambda x: remove_emoji(x))
    df = df.drop_duplicates()
    df.dropna(inplace=True)
    return df

for sub in list_subs:
    # The dataset path should to be modified
    df = pd.read_csv(f"reddit/subreddits23/{sub}_submissions_2023.csv")
    df.rename(columns={'id': 'submission','created_utc':'created','permalink':'shortlink'},inplace=True)
    df['posted_on'] = pd.to_datetime(df['created'], unit ='s')
    df = df[(df['posted_on'].dt.year >= 2019) & (df['posted_on'].dt.year <= 2022)]
    df.drop('created',axis=1, inplace=True)
    print(f"{sub}:",min(df.posted_on),max(df.posted_on))
    df = clean_df(df)
    final_df = df[~df['title'].str.lower().str.contains('daily discussion')]
    final_df.reset_index(drop=True, inplace=True)
    print(f"Final shape of {sub} df:{final_df.shape}")
    if sub.startswith('b'):
        crypto='crypto_bitcoin'
    else:
        crypto='crypto_ethereum'
    # The dataset path should to be modified
    final_df.to_csv(f"Data/Cleaned_Data/{crypto}/submission_and_comments/{sub}_submissions_19_22.csv",index=False)