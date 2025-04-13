import evaluate
import numpy as np

bleu = evaluate.load("bleu")
rouge = evaluate.load("rouge")

def compute_bleu(predictions, references):
    result = bleu.compute(
        predictions=predictions,
        references=[[r] for r in references]

    )
    return result["bleu"]

def compute_rouge(predictions, references):
    result = rouge.compute(predictions=predictions, references=references)
    return result

def compute_exact_match(predictions, references):
    return np.mean([int(p.strip() == r.strip()) for p, r in zip(predictions, references)])

if __name__ == "__main__":
    preds = ["The cat sat on the mat.", "The sky is blue."]
    refs = ["The cat is on the mat.", "The sky is blue."]
    
    print("BLEU:", compute_bleu(preds, refs))
    print("ROUGE:", compute_rouge(preds, refs))
    print("Exact Match:", compute_exact_match(preds, refs))
