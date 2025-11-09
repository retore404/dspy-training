from dotenv import load_dotenv
import dspy
import os
from langfuse import get_client
from openinference.instrumentation.dspy import DSPyInstrumentor

# Observe with Langfuse
langfuse = get_client()
DSPyInstrumentor().instrument()

load_dotenv()

def main():
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    lm = dspy.LM("openai/gpt-4o-mini", api_key=OPENAI_API_KEY)
    dspy.configure(lm=lm)

    # 質問
    question="イギリスの歴史上，最も偉大な人物は？"

    # ChainOfThoughtで推論を複数（3つ）生成する
    math = dspy.ChainOfThought("question -> answer: str")
    completions = math(question=question, config=dict(n=3)).completions
    for reasoning_logic in completions.reasoning:
        print("-------------------------------------------")
        print(reasoning_logic)

    # MultiChainComparisonで生成した推論を比較
    compare = dspy.MultiChainComparison("question -> answer: str", M=3)
    final_result = compare(completions, question=question)
    print("===================================")
    print(final_result)

    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    dspy.inspect_history(n=1)

if __name__ == "__main__":
    main()
