import dspy
from datasets import load_dataset
import os
from langfuse import get_client
from openinference.instrumentation.dspy import DSPyInstrumentor
from dspy import GEPA

# Observe with Langfuse
langfuse = get_client()
DSPyInstrumentor().instrument()

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

    return train_set, val_set, test_set

class GenerateResponse(dspy.Signature):
    """Solve the problem and provide the answer in the correct format."""
    problem = dspy.InputField()
    answer = dspy.OutputField()

def metric(example, prediction, trace=None, pred_name=None, pred_trace=None):
    correct_answer = int(example['answer'])
    try:
        llm_answer = int(prediction.answer)
    except ValueError as e:
        return 0
    return int(correct_answer == llm_answer)

def metric_with_feedback(example, prediction, trace=None, pred_name=None, pred_trace=None):
    correct_answer = int(example['answer'])
    written_solution = example.get('solution', '')
    try:
        llm_answer = int(prediction.answer)
    except ValueError as e:
        feedback_text = f"The final answer must be a valid integer and nothing else. You responded with '{prediction.answer}', which couldn't be parsed as a python integer. Please ensure your answer is a valid integer without any additional text or formatting."
        feedback_text += f" The correct answer is '{correct_answer}'."
        if written_solution:
            feedback_text += f" Here's the full step-by-step solution:\n{written_solution}\n\nThink about what takeaways you can learn from this solution to improve your future answers and approach to similar problems and ensure your final answer is a valid integer."
        return dspy.Prediction(score=0, feedback=feedback_text)

    score = int(correct_answer == llm_answer)

    feedback_text = ""
    if score == 1:
        feedback_text = f"Your answer is correct. The correct answer is '{correct_answer}'."
    else:
        feedback_text = f"Your answer is incorrect. The correct answer is '{correct_answer}'."

    if written_solution:
        feedback_text += f" Here's the full step-by-step solution:\n{written_solution}\n\nThink about what takeaways you can learn from this solution to improve your future answers and approach to similar problems."

    return dspy.Prediction(score=score, feedback=feedback_text)


def main():
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    lm = dspy.LM("openai/gpt-4o-mini", api_key=OPENAI_API_KEY)
    dspy.configure(lm=lm)

    program = dspy.ChainOfThought(GenerateResponse)

    train_set, val_set, test_set = init_dataset()

    evaluate = dspy.Evaluate(
        devset=test_set,
        metric=metric,
        num_threads=32,
        display_table=True,
        display_progress=True
    )

    optimizer = GEPA(
        metric=metric_with_feedback,
        auto="light",
        num_threads=32,
        track_stats=True,
        reflection_minibatch_size=3,
        reflection_lm=dspy.LM(model="gpt-5", temperature=1.0, max_tokens=32000, api_key=OPENAI_API_KEY)
    )

    optimized_program = optimizer.compile(
        program,
        trainset=train_set,
        valset=val_set,
    )

    print("Optimized Program Instructions:")
    print(optimized_program.predict.signature.instructions)

    evaluate(optimized_program)


if __name__ == "__main__":
    main()
