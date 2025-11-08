from dotenv import load_dotenv
import dspy
import os

load_dotenv()

def get_weather(city: str) -> str:
    print("get_weather called")
    return f"現在の{city}の天気は雨です．"

def get_sightseeing_spots(city: str) -> list:
    print("get_sightseeing_spots called")
    return [f"{city}博物館", f"{city}公園"]

def main():
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    lm = dspy.LM("openai/gpt-4o-mini", api_key=OPENAI_API_KEY, cache=False)
    dspy.configure(lm=lm)
    react = dspy.ReAct("question -> answer: str", tools=[get_weather, get_sightseeing_spots])
    print(react(question="東京の今の天気は？また，天気を踏まえて今日訪れるべき東京の観光スポットは？"))
    dspy.inspect_history(n=5)

if __name__ == "__main__":
    main()
