from transformers import pipeline

generator = pipeline("text-generation", model="./outputs")

prompt = "Explain overfitting:"
result = generator(prompt, max_length=100)

print(result[0]["generated_text"])
