from dotenv import load_dotenv
import dspy
import os
from langfuse import get_client
from openinference.instrumentation.dspy import DSPyInstrumentor

load_dotenv()

# Observe with Langfuse
langfuse = get_client()
DSPyInstrumentor().instrument()

class PersonalInformationExtractor(dspy.Signature):
    """Extract personal information from self-introduction text."""
    self_introduction: str = dspy.InputField()
    family_name: str = dspy.OutputField()
    given_name: str = dspy.OutputField()
    age: int = dspy.OutputField()
    birth_place: str|None = dspy.OutputField()
    occupation: str = dspy.OutputField()

def main():
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    lm = dspy.LM("openai/gpt-4o-mini", api_key=OPENAI_API_KEY, cache=False)
    dspy.configure(lm=lm)
    # DSPyプログラム
    program = dspy.Predict(PersonalInformationExtractor)

    # trainset
    trainset = [
        dspy.Example(
            self_introduction="はじめまして。松村健一と申します。35歳で、フリーのビデオクリエイターをしています。映像編集や撮影を中心に活動しています。",
            family_name="松村", given_name="健一", age=35, birth_place=None, occupation="フリーのビデオクリエイター").with_inputs("self_introduction"),
        dspy.Example(
            self_introduction="はじめまして。佐藤健太と申します。29歳で、普段はWebエンジニアとして働いています。福岡県出身ですが、今は東京で暮らしています。趣味は写真を撮ることで写真家を目指しています。",
            family_name="佐藤", given_name="健太", age=29, birth_place="福岡県", occupation="Webエンジニア").with_inputs("self_introduction"),
        dspy.Example(
            self_introduction="こんにちは、山田美咲です。32歳の看護師をしています。北海道の釧路市出身で、のんびりした性格だとよく言われます。",
             family_name="山田", given_name="美咲", age=32, birth_place="北海道釧路市", occupation="看護師").with_inputs("self_introduction"),
        dspy.Example(
            self_introduction="どうも、中村亮介と申します。26歳で、広告代理店の企画職をしています。京都府出身です。旅行が好きで時間があれば海外に行っています。",
            family_name="中村", given_name="亮介", age=26, birth_place="京都府", occupation="広告代理店の企画職").with_inputs("self_introduction"),
        dspy.Example(
            self_introduction="初めまして。高橋由衣です。24歳のイラストレーターです。神奈川県横浜市の出身で、猫と暮らしています。",
            family_name="高橋", given_name="由衣", age=24, birth_place="神奈川県横浜市", occupation="イラストレーター").with_inputs("self_introduction"),
        dspy.Example(
            self_introduction="こんにちは、大森悠人と申します。41歳で、大学の研究員をしています。愛知県豊田市出身です。専門は材料科学です。",
            family_name="大森", given_name="悠人", age=41, birth_place="愛知県豊田市", occupation="大学の研究員").with_inputs("self_introduction"),
        dspy.Example(
            self_introduction="どうも、井上春香です。28歳で、カフェの店長をしています。兵庫県出身で、コーヒーを淹れるのが大好きです。",
            family_name="井上", given_name="春香", age=28, birth_place="兵庫県", occupation="カフェの店長").with_inputs("self_introduction"),
        dspy.Example(
            self_introduction="こんにちは、石原奈々です。22歳で、アパレル販売員として働いています。服を見ることも作ることも好きです。",
            family_name="石原", given_name="奈々", age=22, birth_place=None, occupation="アパレル販売員").with_inputs("self_introduction"),
        dspy.Example(
            self_introduction="どうも、渡辺翔太と申します。30歳で、ゲームプログラマーをしています。宮城県仙台市出身で、RPGが特に好きです。",
            family_name="渡辺", given_name="翔太", age=30, birth_place="宮城県仙台市", occupation="ゲームプログラマー").with_inputs("self_introduction"),
        dspy.Example(
            self_introduction="はじめまして。木村彩です。27歳の小学校教師です。長野県松本市出身で、自然の多い場所が落ち着きます。",
            family_name="木村", given_name="彩", age=27, birth_place="長野県松本市", occupation="小学校教師").with_inputs("self_introduction"),
    ]

    # メトリック関数の定義
    def metric(example, pred, trace=None):
        print("=== QA Metric ===")
        print("苗字の一致:", example.family_name, "/", pred.family_name)
        print("名前の一致:", example.given_name, "/", pred.given_name)
        print("年齢の一致:", example.age, "/", pred.age)
        print("出身地の一致:", example.birth_place, "/", pred.birth_place)
        print("職業の一致:", example.occupation, "/", pred.occupation)
        print("===============")
        # 全ての出力フィールドが一致するかチェック
        return (
            example.family_name == pred.family_name and
            example.given_name == pred.given_name and
            example.age == pred.age and
            example.birth_place == pred.birth_place and
            example.occupation == pred.occupation
        )


    # オプティマイザの定義
    optimizer = dspy.BootstrapFewShot(
        metric=metric,
        max_bootstrapped_demos=4,
        max_labeled_demos=4,
        max_rounds=1,
        max_errors=10
    )

    # 最適化前のプログラムを保存
    program.save("./artifacts/without_optimize.json")

    # 最適化
    optimized_program = optimizer.compile(program, trainset=trainset)

    # 最適化したプログラムを保存
    optimized_program.save("./artifacts/with_optimize.json")

if __name__ == "__main__":
    main()
