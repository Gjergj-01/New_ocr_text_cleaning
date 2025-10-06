# GOOGLE GEMINI
from google import genai
import json
import re
import time

def get_LLM(model, path_file):
    
    assert((model == "Prometheus") or (model == "Gemini"))
    
    with open('datasets/LLMasjudge_instructions.json') as file:
        instructions = json.load(file)

    #with open(path_file) as file:
    #    data = json.load(file)


    data = data[86:]

    orig_instruction = instructions["orig_instructions"]
    orig_criteria = instructions["orig_criteria"]
    orig_score1_description = instructions["orig_score1_description"]
    orig_score2_description = instructions["orig_score2_description"]
    orig_score3_description = instructions["orig_score3_description"]
    orig_score4_description = instructions["orig_score4_description"]
    orig_score5_description = instructions["orig_score5_description"]

    if model == "Gemini":

        client = genai.Client(api_key="YOUR API KEY")

        outputs = []

        num_requests = 1

        for d in data:

            if num_requests % 15 == 0:
                print("Waiting 70 seconds .....")
                time.sleep(70)      # we can make a maximum of 15 calls per minute. To be sure, 
                                    # we wait 70 second before the next call

            num_requests += 1

            input = d['in']
            orig_response = d['hyp']
            orig_reference_answer = d['ref']

            print(f"[get_LLM]: elaborating response {orig_response}...")
            response = client.models.generate_content(
                #model="gemini-2.0-flash", 
                model="gemini-2.5-flash-lite",
                contents=f"""
                    ### Task Description:
                        You are given an instruction, an input given to the LLM, its corresponding response to evaluate, a reference answer (representing the ideal answer with score 5), and a detailed scoring rubric.

                        Your task is to:
                        1. Evaluate the quality of the response strictly according to the given evaluation criteria and scoring rubric.
                        2. Compare the response to the reference answer and judge how well it satisfies the rubric (be aware that sometimes the refrence answer may contain errors).
                        3. Provide a justification for your score based on specific aspects of the response.
                        4. Output the result as follows:
                        "Feedback: (your short explanation) --- [SCORE] (a number from 1 to 5)"
                        
                        Notice that the sequence "---" must be unique (and must always be present) in the feedback, as it will be used as separator for the score.

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
            print("[get_LLM] Response: ", response.text)
            feedback, score = response.text.split("---")
            score = re.search('\d', score).group()
            info = {"in": input, "hyp": orig_response, "ref": orig_reference_answer, "feedback": feedback, "score": int(score)}
            outputs.append(info)

            # Save results while executing
            with open("gemini_output.jsonl", "a") as f:
                f.write(json.dumps(info) + "\n")
    
        return outputs