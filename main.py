from dotenv import load_dotenv
import dspy
import os

load_dotenv()

def main():
    print("Hello from dspy-training!")
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    lm = dspy.LM("openai/gpt-4o-mini", api_key=OPENAI_API_KEY)
    dspy.configure(lm=lm)
    print(lm("Say this is a test!", temperature=0.7))  # => ['This is a test!']
    print(lm(messages=[{"role": "user", "content": "Say this is a test!"}]))  # => ['This is a test!']


if __name__ == "__main__":
    main()
