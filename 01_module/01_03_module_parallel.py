from dotenv import load_dotenv
import dspy
import os

load_dotenv()

def main():
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    lm = dspy.LM("openai/gpt-4o-mini", api_key=OPENAI_API_KEY)
    dspy.configure(lm=lm)
    math = dspy.ChainOfThought("question -> answer: float")
    questions = [
        "2つのサイコロを振った時，2つの出目の合計が1になる確率は？",
        "2つのサイコロを振った時，2つの出目の合計が2になる確率は？",
        "2つのサイコロを振った時，2つの出目の合計が3になる確率は？",
        "2つのサイコロを振った時，2つの出目の合計が4になる確率は？",
        "2つのサイコロを振った時，2つの出目の合計が5になる確率は？",
        "2つのサイコロを振った時，2つの出目の合計が6になる確率は？",
        "2つのサイコロを振った時，2つの出目の合計が7になる確率は？",
        "2つのサイコロを振った時，2つの出目の合計が8になる確率は？",
        "2つのサイコロを振った時，2つの出目の合計が9になる確率は？",
        "2つのサイコロを振った時，2つの出目の合計が10になる確率は？",
        "2つのサイコロを振った時，2つの出目の合計が11になる確率は？",
        "2つのサイコロを振った時，2つの出目の合計が12になる確率は？",
        "2つのサイコロを振った時，2つの出目の合計が13になる確率は？"
    ]
    exec_list = [(math, {"question": question}) for question in questions]

    exec_parallel = dspy.Parallel()
    results = exec_parallel(exec_list)

    prob_sum=0.0

    for question, result in zip(questions, results):
        print(f"\nQuestion: {question}")
        print(f"Answer: {result.answer}")
        prob_sum += result.answer

    print(f"確率の合計値：{prob_sum}")

if __name__ == "__main__":
    main()
