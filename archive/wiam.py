from transformers import pipeline

def classify_cry(chunk):
    pipe = pipeline("audio-classification", model="Wiam/distilhubert-finetuned-babycry-v7")
    results = pipe(chunk)
    top_pred = max(results, key=lambda x: x["score"])
    message = f"{top_pred['label']} ({top_pred['score']:.2f})"   
    return message