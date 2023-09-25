import json

from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.prompts import ChatPromptTemplate, PromptTemplate, SystemMessagePromptTemplate, AIMessagePromptTemplate, HumanMessagePromptTemplate

from hpo_extract.setup_data import HPO_VECTORS, HPO_DF, HPO_TREE

MODEL_NAME = "gpt-3.5-turbo"
#MODEL_NAME = "gpt-4"


def load_input():
    with open("data/hpo_to_follow_up/input_samples/sample1.json", "r") as f:
        input_dict = json.load(f)
        
    return input_dict


def load_prompts():
    with open("data/hpo_to_follow_up/prompts/deepen_llm_prompts.json", "r") as f:
        deepen_llm_prompts = json.load(f)
        
    return deepen_llm_prompts


def initialize_deepen_llm_chats(prompts):
    sysmsg = SystemMessage(content=prompts.get("system_message"))
    init_prompt_template = HumanMessagePromptTemplate.from_template(
        prompts.get("initial_prompt_template"))
    main_prompt_template = HumanMessagePromptTemplate.from_template(
        prompts.get("main_prompt_template"))
    
    return sysmsg, init_prompt_template, main_prompt_template


def get_child_terms(term_list):
    term_to_new_dict = {}
    term_id_list = HPO_DF.loc[HPO_DF['lbl'].isin(term_list), 'id'].tolist()

    for term_id in term_id_list:
        child_term_list = HPO_TREE.get_children(term_id)
        child_term_list = HPO_DF.loc[HPO_DF['id'].isin(child_term_list), 'lbl'].tolist()

        if child_term_list:
            term_lbl = HPO_DF.loc[HPO_DF['id'] == term_id, "lbl"].values[0]
            term_to_new_dict[term_lbl] = []

            for child_term in child_term_list:
                definition = HPO_DF.loc[HPO_DF['lbl'] == child_term, "definition"].values[0]
                comments = HPO_DF.loc[HPO_DF['lbl'] == child_term, "comments"].values[0]

                def_string = f"Definition: {definition} \n" if definition else ""
                com_string = f"Comments: {comments} \n" if comments else ""

                term_def_str = f"""Term: {child_term} \n{def_string}{com_string}"""

                term_to_new_dict[term_lbl].append(term_def_str)

    return term_to_new_dict


def run_deepen_chain(input_dict, llm, prompts, show_work=False):
    patient_description = input_dict.get("patieint_description")
    current_hpo_terms = str(input_dict.get("hpo_terms"))

    child_terms_dict = get_child_terms(input_dict.get("hpo_terms"))

    deepen_llm_sysmsg, deepen_init_prompt_template, deepen_main_prompt_template = initialize_deepen_llm_chats(prompts)
    
    batched_input = []
    for parent_term in child_terms_dict:
        deepen_init_prompt_formated = deepen_init_prompt_template.format_messages(description=patient_description, current_hpo_terms=current_hpo_terms)[0]
        deepen_main_prompt_formated = deepen_main_prompt_template.format_messages(term=parent_term, term_list="\n".join(child_terms_dict[parent_term]))[0]
        batched_input.append([deepen_llm_sysmsg,
                              deepen_init_prompt_formated,
                              deepen_main_prompt_formated])

    deepen_out = llm.generate(batched_input)
    
    if show_work:
        explain_msg = HumanMessage(content=prompts.get("explain_follow_up_prompt"))

        explain_chain_batch_input = [sub_list + [deepen_out.generations[i][0].message, explain_msg] for i, sub_list in enumerate(batched_input)]
        print(explain_chain_batch_input[0], "\n")
        
        deepen_out = llm.generate(explain_chain_batch_input)
        
    return deepen_out
