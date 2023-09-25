import json
import os
import sys
from langchain.memory import ChatMessageHistory
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain import LLMChain
from langchain.prompts import ChatPromptTemplate, PromptTemplate, SystemMessagePromptTemplate, AIMessagePromptTemplate, \
    HumanMessagePromptTemplate
from langchain.schema import AIMessage, HumanMessage, SystemMessage

from preprocessing import hpo_prep

KEY = os.environ["OPENAI_API_KEY"]
MODEL_NAME = "gpt-3.5-turbo"


def most_similar(vector_store, hpo_df, bad_term, top_k=1):
    """
    use to retrieve a term in our hpo_df that is semantically similar to the bad_term
    """
    similar_term_doc = vector_store.similarity_search(bad_term, k=top_k)[0]
    similar_term = hpo_df.loc[hpo_df['text_to_embed'] == similar_term_doc.page_content, 'lbl'].values[0]
    return similar_term


def split_input(vector_store, hpo_df, input):
    terms_cleaned = []
    for term in input["hpo_terms"]:
        if term in hpo_df['lbl'].unique():
            terms_cleaned.append(term)
        else:
            print(f"{term}, not real term, finding similar...")
            new_term = most_similar(vector_store, hpo_df, term)
            print(f"New term : {new_term}")
            terms_cleaned.append(new_term)

    text_format = [f"""Patient description:\n'''\n{input['patieint_description']}\n'''\n'''\nHPO term: {term}\n'''"""
                   for term in terms_cleaned]
    return text_format


def get_child_terms(term_list, hpo_df, hpo_tree):
    term_to_new_dict = {}
    term_id_list = hpo_df.loc[hpo_df['lbl'].isin(term_list), 'id'].tolist()

    for term_id in term_id_list:
        child_term_list = hpo_tree.get_children(term_id)
        child_term_list = hpo_df.loc[hpo_df['id'].isin(child_term_list), 'lbl'].tolist()

        if child_term_list:
            term_lbl = hpo_df.loc[hpo_df['id'] == term_id, "lbl"].values[0]
            term_to_new_dict[term_lbl] = []

            for child_term in child_term_list:
                definition = hpo_df.loc[hpo_df['lbl'] == child_term, "definition"].values[0]
                comments = hpo_df.loc[hpo_df['lbl'] == child_term, "comments"].values[0]

                def_string = f"Definition: {definition} \n" if definition else ""
                com_string = f"Comments: {comments} \n" if comments else ""

                term_def_str = f"""Term: {child_term} \n{def_string}{com_string}"""

                term_to_new_dict[term_lbl].append(term_def_str)

    return term_to_new_dict


if __name__ == '__main__':
    # preprocessing for hpo datastructures
    hpo_vectors = hpo_prep.create_hpo_vector_store()
    hpo_tree = hpo_prep.make_hpo_tree()
    hpo_df = hpo_prep.make_hpo_dataframe()

    # create llms
    broaden_llm = ChatOpenAI(model=MODEL_NAME, temperature=0.0)
    deepen_llm = ChatOpenAI(model=MODEL_NAME, temperature=0.0)
    eval1_deepen_llm = ChatOpenAI(model=MODEL_NAME, temperature=0.0)
    q_gen_deepen_llm = ChatOpenAI(model=MODEL_NAME, temperature=0.5)
    eval2_deepen_llm = ChatOpenAI(model=MODEL_NAME, temperature=0.0)
    broaden_eval_llm = ChatOpenAI(model=MODEL_NAME, temperature=0.0)

    # load prompts and templates

    with open("generation/hpo_to_follow_up/prompts/deepen_llm_prompts.json", "r") as f:
        deepen_llm_prompts = json.load(f)

    with open("generation/hpo_to_follow_up/prompts/eval1_deepen_llm_sysmsg.txt", "r") as f:
        eval1_deepen_llm_sysmsg_txt = f.read()
    with open("generation/hpo_to_follow_up/prompts/q_gen_deepen_llm_sysmsg.txt", "r") as f:
        q_gen_deepen_llm_sysmsg_txt = f.read()
    with open("generation/hpo_to_follow_up/prompts/eval2_deepen_llm_sysmsg.txt", "r") as f:
        eval2_deepen_llm_sysmsg_txt = f.read()
    with open("generation/hpo_to_follow_up/prompts/broaden_eval_llm_sysmsg.txt", "r") as f:
        broaden_eval_llm_sysmsg_txt = f.read()

    with open("generation/hpo_to_follow_up/samples/sample1.json", "r") as f:
        input = json.load(f)

    with open("generation/prompts/clin_note_followup/prompt_1_template.txt", "r") as f:
        p1_template = f.read()

    # initialize chats and system messages

    deepen_llm_sysmsg = SystemMessage(content=deepen_llm_prompts.get("system_message"))
    deepen_init_prompt_template = HumanMessagePromptTemplate.from_template(
        deepen_llm_prompts.get("initial_prompt_template"))
    deepen_main_prompt_template = HumanMessagePromptTemplate.from_template(
        deepen_llm_prompts.get("main_prompt_template"))

    eval1_deepen_llm_sysmsg = SystemMessage(content=eval1_deepen_llm_sysmsg_txt)

    q_gen_deepen_llm_sysmsg = SystemMessage(content=q_gen_deepen_llm_sysmsg_txt)

    eval2_deepen_llm_sysmsg = SystemMessage(content=eval2_deepen_llm_sysmsg_txt)

    broaden_eval_llm_sysmsg = SystemMessage(content=broaden_eval_llm_sysmsg_txt)

    # Deepen chain
    patient_description = input.get("patieint_description")
    current_hpo_terms = str(input.get("hpo_terms"))

    child_terms_dict = get_child_terms(input.get("hpo_terms"), hpo_df, hpo_tree)
    print(child_terms_dict)

    batched_input = [[deepen_llm_sysmsg,
                      deepen_init_prompt_template.format_messages(description=patient_description,
                                                                  current_hpo_terms=current_hpo_terms)[0],
                      deepen_main_prompt_template.format_messages(term=parent_term, term_list="\n".join(child_terms_dict[parent_term]))[0]] for parent_term in child_terms_dict]



    deepen_out = deepen_llm.generate(batched_input)

    print(deepen_out)
    print(deepen_out.llm_output)

    sys.exit()

    # Broaden chain

    prompt_1_template = HumanMessagePromptTemplate.from_template(p1_template)

    llm_chat_hist.append(prompt_1_template.format_messages(clin_notes=clin_notes)[0])

    sys.exit()

