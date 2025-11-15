import dspy
from dspy.datasets import HotPotQA
from dspy.evaluate import Evaluate

# LMの設定
dspy.configure(lm=dspy.LM('openai/gpt-4o-mini'))

# HotPotQAデータセットの読み込み
dataset = HotPotQA(train_seed=2024, train_size=100, dev_size=50, test_size=50)
trainset = [x.with_inputs('question') for x in dataset.train]
devset = [x.with_inputs('question') for x in dataset.dev]

# シンプルなQAプログラム
class SimpleQA(dspy.Module):
    def __init__(self):
        self.generate_answer = dspy.ChainOfThought("question -> answer")

    def forward(self, question):
        return self.generate_answer(question=question)

# メトリック関数の定義
def qa_metric(example, pred, trace=None):
    # 正解と予測を正規化して比較
    gold_answer = example.answer.lower().strip()
    pred_answer = pred.answer.lower().strip()
    return gold_answer == pred_answer

# 最適化前のプログラムを評価
program = SimpleQA()
evaluator = Evaluate(devset=devset, metric=qa_metric, num_threads=4, display_progress=True)

# BootstrapFewShotで最適化
optimizer = dspy.BootstrapFewShot(
    metric=qa_metric,
    max_bootstrapped_demos=4,
    max_labeled_demos=4,
    max_rounds=10,
    max_errors=10
)

optimized_program = optimizer.compile(program, trainset=trainset)

# 最適化前の評価
score_before = evaluator(program)
print(f"最適化前のスコア: {score_before.score}")

# 最適化後の評価
score_after = evaluator(optimized_program)
print(f"最適化後のスコア: {score_after.score}")

# 改善率を表示
improvement = score_after.score - score_before.score
print(f"\n改善: {improvement:.1f}ポイント ({score_before.score}% → {score_after.score}%)")

# 最適化したプログラムを保存
optimized_program.save("./artifacts/hotpotqa_optimized.json")