from dotenv import load_dotenv
import dspy
import os

load_dotenv()

def main():
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    lm = dspy.LM("openai/gpt-4o-mini", api_key=OPENAI_API_KEY)
    dspy.configure(lm=lm)
    math = dspy.ChainOfThought("question -> answer: float")
    print(math(question="2つのサイコロを振った時，2つの出目の合計が2になる確率は？"))

if __name__ == "__main__":
    main()
