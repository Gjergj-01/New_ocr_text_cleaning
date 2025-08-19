from rouge_score import get_rouge
from judgeLLM import get_LLM
from sklearn.metrics import cohen_kappa_score
import json

in_path_file = "datasets/example.json"
out_path_file = "datasets/scores.json"

# Human evaluation domain: [1, 5]
valuation_coefficient = 5
human_annotations = [2, 3]
results = {'human': human_annotations}

rouge_results = get_rouge(in_path_file)
llm_results = get_LLM("Gemini", in_path_file)

assert(len(human_annotations) == len(rouge_results))
assert(len(human_annotations) == len(llm_results))

# Computing Cohen's kappa
rouge_1 = []
rouge_2 = []
rouge_L = []
for r in rouge_results:
    rouge_1.append(round(valuation_coefficient * r['rouge-1']['f']))
    rouge_2.append(round(valuation_coefficient * r['rouge-2']['f']))
    rouge_L.append(round(valuation_coefficient * r['rouge-l']['f']))

print("[Main]: computing cohen's kappa coefficient for rouge score")

score1 = cohen_kappa_score(rouge_1, human_annotations)
score2 = cohen_kappa_score(rouge_2, human_annotations)
scoreL = cohen_kappa_score(rouge_L, human_annotations)
results['score_rouge1'] = rouge_1
results['score_rouge2'] = rouge_2
results['score_rougeL'] = rouge_L
results['cohen_rouge1'] = score1
results['cohen_rouge2'] = score2
results['cohen_rougeL'] = scoreL

print("[Main]: computing cohen's coefficient for llm score")
llm = []
for l in llm_results:
    llm.append(l['score'])

score_llm = cohen_kappa_score(llm, human_annotations)
results['cohen_llm'] = score_llm
results['score_llm'] = llm

print("[Main]: finished computed scores", results)
print("[Main]: writing in output the results")
with open(out_path_file, 'w') as out:
    json.dump(results, out, indent=4)