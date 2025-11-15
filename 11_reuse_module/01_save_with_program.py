from dotenv import load_dotenv
import dspy
import os
from langfuse import get_client
from openinference.instrumentation.dspy import DSPyInstrumentor

load_dotenv()

# Observe with Langfuse
langfuse = get_client()
DSPyInstrumentor().instrument()

class Answer(dspy.Signature):
    """Answer the question."""
    question: str = dspy.InputField()
    answer: float = dspy.OutputField()

def main():
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    lm = dspy.LM("openai/gpt-4o-mini", api_key=OPENAI_API_KEY, cache=False)
    dspy.configure(lm=lm)
    math = dspy.ChainOfThought(Answer)
    math.save(path="./artifacts/with_program/", saved_program=True)
    print(math(question="2つのサイコロを振った時，2つの出目の合計が4になる確率は？"))
    dspy.inspect_history(n=1)

if __name__ == "__main__":
    main()
