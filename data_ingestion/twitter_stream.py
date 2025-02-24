import tweepy
import json

# config.py should hold your credentials; here we'll directly define them for simplicity.
API_KEY = 'your_api_key'
API_SECRET = 'your_api_secret'
ACCESS_TOKEN = 'your_access_token'
ACCESS_TOKEN_SECRET = 'your_access_token_secret'

class MyStreamListener(tweepy.StreamListener):
    def on_data(self, data):
        try:
            tweet = json.loads(data)
            # For now, we just print the tweet.
            print(tweet)
            # Future step: Send the tweet to a processing queue or store it.
        except Exception as e:
            print(f"Error: {e}")
        return True

    def on_error(self, status_code):
        print(f"Error with status code: {status_code}")
        # Disconnect on rate limiting
        if status_code == 420:
            return False

if __name__ == "__main__":
    # Set up authentication
    auth = tweepy.OAuthHandler(API_KEY, API_SECRET)
    auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
    listener = MyStreamListener()
    stream = tweepy.Stream(auth=auth, listener=listener)

    # Filter tweets containing specific keywords and in English
    stream.filter(track=['python', 'data'], languages=["en"])
