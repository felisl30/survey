import pandas as pd

q = pd.read_csv("data/hotpotqa_mini/questions_s1.csv")
c = pd.read_csv("data/hotpotqa_mini/corpus_s1.csv")
r = pd.read_csv("data/hotpotqa_mini/qrels_s1.csv")

print("Preguntas:", len(q))
print("Chunks:", len(c))
print("Qrels:", len(r))

print("\nPor topic:")
print(q["topic"].value_counts())

print("\nPor tipo:")
print(q["hotpot_type"].value_counts())

print("\nPor dificultad:")
print(q["level"].value_counts())

print("\nPrimeras preguntas:")
print(q[["id", "topic", "hotpot_type", "level", "original_question", "expected_answer"]].head())