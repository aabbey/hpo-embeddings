import json
import sys
from langchain.prompts import SystemMessagePromptTemplate
import pandas as pd
from random import sample


sample_number = 1
disease_name = "Biotinidase"
sample_pheno_path = 'samples/sample disease phenotypes/terms_for_Biotinidase Deficiency.csv'
num_traits_to_use = 20


if __name__ == '__main__':
    full_example_traits = pd.read_csv(sample_pheno_path)
    all_terms_list = full_example_traits['HPO_TERM_NAME'].tolist()
    traits_to_use = sample(all_terms_list, num_traits_to_use) if len(all_terms_list) > num_traits_to_use else all_terms_list
    print(traits_to_use, "\n\n")

    with open('prompts/full_conversation_prompt') as t:
        prompt_template = t.readlines()
        prompt_template = SystemMessagePromptTemplate.from_template(''.join(prompt_template))

    prompt_to_use = prompt_template.format_messages(traits=', '.join(traits_to_use))[0]
    print(f"Prompt to use : {prompt_to_use.content} \n\n")

    print("Paste GPT 4 transcript here, then 'ctrl+d' : ")
    try:
        # Read multi-line input
        lines = []
        while True:
            line = input()
            lines.append(line)
    except EOFError:
        pass  # User has finished entering their multi-line input

    # Join the lines together into a single string
    transcript = '\n'.join(lines)

    data = {
        'transcript': transcript,
        'true_terms': traits_to_use
    }

    with open(f'sample generated transcripts/gpt4-sample{sample_number}-{disease_name}.json', 'w') as json_file:
        json.dump(data, json_file, indent=4)

