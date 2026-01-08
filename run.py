import json
from src.utils.data_utils import read_jsonl
from src.models.mistral_baseline import summarize_article
from src.preprocessing.data import ArticleSample, parse_item, json_to_sample, build_pts_samples
from rouge_score import rouge_scorer
from src.run_pts import merge_section_summaries

def main():
    path = "data/processed/output.txt"
    articles = [parse_item(x) for x in read_jsonl(path)]
    sections = [json_to_sample(build_pts_samples(art)) for art in articles]

    first_article=articles[0]
    first_sections=json_to_sample(build_pts_samples(first_article))
    pred_sections=[]
    for s in first_sections:
        pred_sections.append(summarize_article(s))

    first_article.pred_summary=merge_section_summaries(pred_sections)

if __name__ == "__main__":
    main() 
