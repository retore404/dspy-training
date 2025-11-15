from dotenv import load_dotenv
import dspy
import os
from langfuse import get_client
from openinference.instrumentation.dspy import DSPyInstrumentor
from datasets import load_dataset

load_dotenv()

# Observe with Langfuse
langfuse = get_client()
DSPyInstrumentor().instrument()

class GenerateResponse(dspy.Signature):
    """Solve the problem and provide the answer in the correct format."""
    problem = dspy.InputField()
    answer = dspy.OutputField()

def init_dataset():
    train_split = load_dataset("AI-MO/aimo-validation-aime")['train']
    train_split = [
        dspy.Example({
            "problem": x['problem'],
            'solution': x['solution'],
            'answer': x['answer'],
        }).with_inputs("problem")
        for x in train_split
    ]
    import random
    random.Random(0).shuffle(train_split)
    tot_num = len(train_split)

    test_split = load_dataset("MathArena/aime_2025")['train']
    test_split = [
        dspy.Example({
            "problem": x['problem'],
            'answer': x['answer'],
        }).with_inputs("problem")
        for x in test_split
    ]

    train_set = train_split[:int(0.5 * tot_num)]
    val_set = train_split[int(0.5 * tot_num):]
    test_set = test_split * 5
    test_set = test_split * 1

    return train_set, val_set, test_set

def metric(example, prediction, trace=None, pred_name=None, pred_trace=None):
    correct_answer = int(example['answer'])
    try:
        llm_answer = int(prediction.answer)
    except ValueError as e:
        return 0
    return int(correct_answer == llm_answer)

def main():
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    lm = dspy.LM("openai/gpt-4.1-mini", api_key=OPENAI_API_KEY, cache=False, temperature=1.0, max_tokens=32000)
    dspy.configure(lm=lm)

    # データの読み込み
    train_set, val_set, test_set = init_dataset()

    # 評価関数の定義
    evaluate = dspy.Evaluate(
        devset=test_set,
        metric=metric,
        num_threads=5,
        display_table=True,
        display_progress=True
    )

    # 最適化前プログラム
    program = dspy.ChainOfThought(GenerateResponse)
    program.load("./artifacts/aime_baseline.json")
    # 最適化後プログラム
    optimized_program = dspy.ChainOfThought(GenerateResponse)
    optimized_program.load("./artifacts/aime_optimized_gepa.json")

    # 評価の実行
    print("最適化前のプログラムの評価")
    evaluate(program)
    print("最適化後のプログラムの評価")
    evaluate(optimized_program)

if __name__ == "__main__":
    main()
