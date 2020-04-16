import got3 as got
import numpy as np
import pandas as pd
from textblob import TextBlob
import re
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from PIL import Image

tweets_number = input('') #取得するツイート数

print('検索したいキーワードを入力してください')
tweetCriteria = got.manager.TweetCriteria().setQuerySearch(input('')).setUsername('realDonaldTrump').setSince("2015-01-01").setUntil('2017-12-31').setMaxTweets(int(tweets_number))
#@realDonaldTrumpの中から任意のキーワードにヒットするツイートを取得
 tweets = got.manager.TweetManager.getTweets(tweetCriteria)   

def clean_tweet(tweet):
    '''
    Utility function to clean the text in a tweet by removing 
    links and special characters using regex.
    '''
    return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())

def analize_sentiment(tweet):
    '''
    Utility function to classify the polarity of a tweet
    using textblob.
    '''
    analysis = TextBlob(clean_tweet(tweet))
    if analysis.sentiment.polarity > 0:
        return 1
    elif analysis.sentiment.polarity == 0:
        return 0
    else:
        return -1
        
df = pd.DataFrame(index=[], columns=['date','text', 'retweets', 'favorites', 'SA'])

for i in tweets:
    print(i.date)
    print(i.text)
    print(i.retweets)
    print(i.favorites)
    print('\n')
    series = pd.Series([i.date, i.text, i.retweets, i.favorites, analize_sentiment(i.text)], index=df.columns)
    df=df.append(series, ignore_index=True)
    
# We construct lists with classified tweets:

pos_tweets = [ tweet for index, tweet in enumerate(df['text']) if df['SA'][index] > 0]
neu_tweets = [ tweet for index, tweet in enumerate(df['text']) if df['SA'][index] == 0]
neg_tweets = [ tweet for index, tweet in enumerate(df['text']) if df['SA'][index] < 0]

# We print percentages:

print("Percentage of positive tweets: {}%".format(len(pos_tweets)*100/len(df['text'])))
print("Percentage of neutral tweets: {}%".format(len(neu_tweets)*100/len(df['text'])))
print("Percentage de negative tweets: {}%".format(len(neg_tweets)*100/len(df['text'])))

x=np.array([len(pos_tweets), len(neu_tweets), len(neg_tweets)])
# グラフの描画先の準備
fig = plt.figure()

label=['positive', 'neutral', 'negative']
colors = ['green', 'orange', 'blue']
plt.pie(x, labels=label, counterclock=False, startangle=90,
        wedgeprops={'linewidth': 3, 'edgecolor':"white"}, autopct="%1.1f%%", colors=colors)
# ファイルに保存
fig.savefig(input('')+".png")

bannedWords = ['said','new','will','york','many','the','total','never','united','states','failing','totally','bad','failed','people','senator',
              'party','one','state','always','absolutely','governor','make','read','anything','always','good','thing','really','job','lost','show','group',
              'nothing','story','television','political','time','cruz','talk','zero','organization', 'guy','even','false','history','looking',
              'reporting','look','country','poll','say','ratings','former','president','press','reporter','politician','magazine',
              'much','debate','debates','times','campaign','presidential','bush','know','columnist',
              'another','lied','chief','ted','record','another','paid','journal','way','got','life',
              'last','dead','street','great','clue','jeb','rate']
              
for word in bannedWords:
    STOPWORDS.add(word)
trump_mask = np.array(Image.open('trump3.png'))

wordcloud = WordCloud(background_color="white", max_words=1500, mask=trump_mask, stopwords=STOPWORDS, contour_width=1,contour_color='orange')

# テキストからワードクラウドを生成する。
wordcloud.generate(str(df['text']))

# ファイルに保存する。
print("wordcloudのファイル名を入力してください（拡張子は不要）")
wordcloud.to_file(input('')+'.png')

# numpy 配列で取得する。
img = wordcloud.to_array()

plt.imshow(wordcloud)
plt.axis("off")
plt.show()
