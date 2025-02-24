import re

def clean_tweet(tweet):
    """
    Cleans tweet text by removing URLs, mentions, hashtags, and extra whitespace.
    """
    tweet = re.sub(r'http\S+', '', tweet)       # Remove URLs
    tweet = re.sub(r'@\w+', '', tweet)            # Remove mentions
    tweet = re.sub(r'#\w+', '', tweet)            # Remove hashtags
    tweet = re.sub(r'\s+', ' ', tweet).strip()    # Remove extra spaces
    return tweet

if __name__ == "__main__":
    sample = "Loving the new features on Twitter! Check out https://example.com @user #awesome"
    print(clean_tweet(sample))
