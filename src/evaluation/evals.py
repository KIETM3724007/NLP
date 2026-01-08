from rouge_score import rouge_scorer

def rouge_eval(articles):
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=True
    )
    for art in articles:
        ref = " ".join(art.summary)
        pred = art.pred_summary

        scores = scorer.score(ref, pred)

        print(f"\nArticle {art.id}")
        print(f"R1: {scores['rouge1'].fmeasure:.4f}")
        print(f"R2: {scores['rouge2'].fmeasure:.4f}")
        print(f"RL: {scores['rougeL'].fmeasure:.4f}")
