from dotenv import load_dotenv
import dspy
import os

load_dotenv()

def get_calorie_by_food_name(food_name: str) -> float:
    return 100.0

def get_necessary_calorie_each_day_by_age(age: int) -> float:
    return 2000.0

def main():
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    lm = dspy.LM("openai/gpt-4o-mini", api_key=OPENAI_API_KEY, cache=False)
    dspy.configure(lm=lm)
    codeact = dspy.CodeAct("question -> answer: float", tools=[get_calorie_by_food_name, get_necessary_calorie_each_day_by_age])
    print(codeact(question="30歳です．ハンバーグを食べたら1日に必要なカロリーの何％を摂取したことになりますか？"))
    dspy.inspect_history(n=15)

if __name__ == "__main__":
    main()
