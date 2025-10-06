from rouge import Rouge
import json

def get_rouge(file_path):
    """
    Restituisce il rouge score -1, -2, -L di una lista di coppie (hypothesis, ref)
    
    Args:       - file path di un file .json
                - Il file json contiene una lista di dizionari. Ogni dizionario ha un campo <hyp> e un campo <ref>
                in cui sono contenuti rispettivamente la predizione dell'LLM e il gold standard

    Returns     - Lista di dizionari. Ciascun dizionario contiene i campi rouge-1, rouge-2, rouge-l. 
                - Ogni campo rouge contiene f1 score, precision e recall.

    Ret Example:
            [
                {
                    "rouge-1": {
                    "f": 0.4786324739396596,
                    "p": 0.6363636363636364,
                    "r": 0.3835616438356164
                    },
                    "rouge-2": {
                    "f": 0.2608695605353498,
                    "p": 0.3488372093023256,
                    "r": 0.20833333333333334
                    },
                    "rouge-l": {
                    "f": 0.44705881864636676,
                    "p": 0.5277777777777778,
                    "r": 0.3877551020408163
                }
            }, ..., ]
    """

    # Load some sentences
    with open(file_path) as f:
        data = json.load(f)


    cleaned = [(d['hyp'], d['ref']) for d in data if d['ref'].strip() != '.']
    hyps, refs = map(list, zip(*cleaned))
    rouge = Rouge()
    scores = rouge.get_scores(hyps, refs)
    return scores