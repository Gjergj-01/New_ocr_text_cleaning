# GOOGLE GEMINI
from google import genai
import json
import re

def get_LLM(model, path_file):
    
    assert((model == "Prometheus") or (model == "Gemini"))
    
    with open('datasets/LLMAsjudge_instructions.json') as file:
        instructions = json.load(file)

    with open(path_file) as file:
        data = json.load(file)

    orig_instruction = instructions["orig_instructions"]
    orig_criteria = instructions["orig_criteria"]
    orig_score1_description = instructions["orig_score1_description"]
    orig_score2_description = instructions["orig_score2_description"]
    orig_score3_description = instructions["orig_score3_description"]
    orig_score4_description = instructions["orig_score4_description"]
    orig_score5_description = instructions["orig_score5_description"]

    if model == "Gemini":

        client = genai.Client(api_key="YOUR_API_KEY")

        outputs = []

        for d in data:

            input = d['in']
            orig_response = d['hyp']
            orig_reference_answer = d['ref']

            print(f"[get_LLM]: elaborating response {orig_response[:50]}...")
            response = client.models.generate_content(
                model="gemini-2.0-flash", 
                contents=f"""
                    ### Task Description:
                        You are given an instruction, an input given to the LLM, its corresponding response to evaluate, a reference answer (representing the ideal answer with score 5), and a detailed scoring rubric.

                        Your task is to:
                        1. Evaluate the quality of the response strictly according to the given evaluation criteria and scoring rubric.
                        2. Compare the response to the reference answer and judge how well it satisfies the rubric.
                        3. Provide a justification for your score based on specific aspects of the response.
                        4. Output the result as follows:
                        "Feedback: (your explanation) - [SCORE] (a number from 1 to 5)"

                        ### The instruction to evaluate:
                        {orig_instruction}

                        ### The input given to the LLM
                        {input}

                        ### Response to evaluate:
                        {orig_response}

                        ### Reference Answer (Score 5):
                        {orig_reference_answer}

                        ### Score Rubric:
                        {orig_criteria}
                        Score 1: {orig_score1_description}  
                        Score 2: {orig_score2_description}  
                        Score 3: {orig_score3_description}  
                        Score 4: {orig_score4_description}  
                        Score 5: {orig_score5_description}

                        ### Feedback:""")
            
            feedback, score = response.text.split("-")
            score = re.search('\d', score).group()
            outputs.append({"feedback": feedback, "score": int(score)})
    
        return outputs