from dotenv import load_dotenv
import dspy
import os
import random

load_dotenv()

def reward_fn(args, pred):
    reward = random.random()
    print(pred.answer, ":", reward)
    return reward

def main():
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    lm = dspy.LM("openai/gpt-4o-mini", api_key=OPENAI_API_KEY, cache=False)
    dspy.configure(lm=lm)
    qa = dspy.ChainOfThought("question -> answer")
    best_of_3 = dspy.BestOfN(module=qa, N=3, reward_fn=reward_fn, threshold=1.0)
    result = best_of_3(question="今日の天気は？").answer
    print(result)

if __name__ == "__main__":
    main()
