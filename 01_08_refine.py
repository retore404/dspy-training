from dotenv import load_dotenv
import dspy
import os
import random

load_dotenv()

def reward_fn(args, pred):
    if pred.answer[-1] == "。":
        return 1.0
    else:
        return 0.0

def main():
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    lm = dspy.LM("openai/gpt-4o-mini", api_key=OPENAI_API_KEY, cache=False)
    dspy.configure(lm=lm)
    # Define a QA module with chain of thought
    qa = dspy.ChainOfThought("question -> answer")

    # Create a refined module that tries up to 3 times
    best_of_3 = dspy.Refine(module=qa, N=3, reward_fn=reward_fn, threshold=1.0)

    # Use the refined module
    result = best_of_3(question="ベルギーの首都はどこでしょうか？").answer

    print(result)
    dspy.inspect_history(n=15)

if __name__ == "__main__":
    main()
