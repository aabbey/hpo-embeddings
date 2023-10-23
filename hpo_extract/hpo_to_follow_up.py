import json
from loguru import logger

from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains.openai_functions import create_openai_fn_chain
from langchain.chains import LLMChain

from hpo_extract.setup_data import HPO_VECTORS, HPO_DF, HPO_TREE
from hpo_extract.funcs_for_llms import pass_list

# MODEL_NAME = "gpt-3.5-turbo"
SELECT_DEEPEN_TERMS_MODEL_NAME = "gpt-3.5-turbo"
# EXTRACT_EXCERPTS_MODEL_NAME = "gpt-3.5-turbo"


MODEL_NAME = "gpt-4"
EXTRACT_EXCERPTS_MODEL_NAME = "gpt-4"
# SELECT_DEEPEN_TERMS_MODEL_NAME = "gpt-4"
MANY_HPO_TERMS = 20


def load_input():
    with open("data/hpo_to_follow_up/input_samples/sample1.json", "r") as f:
        input_dict = json.load(f)

    with open("data/hpo_to_follow_up/input_samples/sample1_clin_notes.txt", "r") as f:
        input_dict["clinician_notes"] = f.read()
    return input_dict


def load_prompts():
    with open("data/hpo_to_follow_up/prompts/deepen_llm_prompts.json", "r") as f:
        deepen_llm_prompts = json.load(f)

    return deepen_llm_prompts


def initialize_broaden_chats(prompts):
    chat_prompt_template = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=prompts.get("system_message")),
            HumanMessagePromptTemplate.from_template(prompts.get("prompt_1")),
        ]
    )

    return chat_prompt_template


def initialize_llm_chats(all_prompts):
    """
    Takes a dictionary of all the prompts associated with all the models in the chain, and converts them into langchain prompts and prompt templates.
    """
    new_all_prompts = {}
    for llm_name, prompts in all_prompts.items():
        prompt_dict = {}
        for p_name, p in prompts.items():
            if p_name == "system_message":
                prompt_dict[p_name] = SystemMessage(content=p)
            else:
                if can_be_formatted(p):
                    prompt_dict[p_name] = HumanMessagePromptTemplate.from_template(p)
                else:
                    prompt_dict[p_name] = HumanMessage(content=p)

        new_all_prompts[llm_name] = prompt_dict

    return new_all_prompts


def get_child_terms(term_list):
    term_to_new_dict = {}
    term_id_list = HPO_DF.loc[HPO_DF["lbl"].isin(term_list), "id"].tolist()

    for term_id in term_id_list:
        child_term_list = HPO_TREE.get_children(term_id)
        child_term_list = HPO_DF.loc[HPO_DF["id"].isin(child_term_list), "lbl"].tolist()

        if child_term_list:
            term_lbl = HPO_DF.loc[HPO_DF["id"] == term_id, "lbl"].values[0]
            term_to_new_dict[term_lbl] = []

            for child_term in child_term_list:
                definition = HPO_DF.loc[
                    HPO_DF["lbl"] == child_term, "definition"
                ].values[0]
                comments = HPO_DF.loc[HPO_DF["lbl"] == child_term, "comments"].values[0]

                term_def_str = {
                    "term": child_term,
                    "definition": definition,
                    "comments": comments,
                }

                term_to_new_dict[term_lbl].append(term_def_str)

    return term_to_new_dict


def get_similar(term, top=1):
    similar_term_doc = HPO_VECTORS.similarity_search(term, k=top)[0]
    return similar_term_doc.page_content


def can_be_formatted(s: str) -> bool:
    try:
        s.format()
        return False
    except:
        return True


def run_deepen_chain(input_dict, llms, prompts, show_work=False):
    """Runs chain to collect follow up question ideas that will help specify HPO terms that are too general.

    Args:
        input_dict (dict): dict of inputs (patient description, hpo term) and clin notes
        llms (dict): each llm used in chain
        prompts (dict): all prompts (as strings) to be used
        show_work (bool, optional): llms explain themselves along the way. Defaults to False.

    Returns:
        str: text output of chain
    """
    patient_description = input_dict.get("patieint_description")
    current_hpo_terms = str(input_dict.get("hpo_terms"))

    child_terms_dict = get_child_terms(input_dict.get("hpo_terms"))
    # subset the hpo terms to those that have children if there are a lot of terms. Only do so if there are more than 2 terms with children.
    if (
        len(input_dict.get("hpo_terms")) > MANY_HPO_TERMS
        and len(list(child_terms_dict.keys())) > 2
    ):
        current_hpo_terms = str(list(child_terms_dict.keys()))

    all_prompts = initialize_llm_chats(
        prompts
    )  # eg. {llm1:{sysmsg:SystemMessege(), p:HumanMessege(), pt:...}, llm2:{...}...}
    main_deepen_chain_prompts = all_prompts["main_deepen_chain"]
    extract_excerpts_prompts = all_prompts["extract_excerpts"]
    select_deepen_terms_prompts = all_prompts["select_deepen_terms"]

    deepen_llm = llms.get("deepen_llm")
    extract_excerpts_llm = llms.get("extract_excerpts_llm")
    select_deepen_terms_llm = llms.get("select_deepen_terms_llm")

    deepen_chat_template = ChatPromptTemplate.from_messages(
        [
            main_deepen_chain_prompts.get("system_message"),
            main_deepen_chain_prompts.get("initial_prompt_template"),
            main_deepen_chain_prompts.get("specify_prompt"),
        ]
    )
    extract_excerpts_chat_template = ChatPromptTemplate.from_messages(
        [
            extract_excerpts_prompts.get("system_message"),
            extract_excerpts_prompts.get("main_prompt_template"),
        ]
    )
    select_terms_chat_template = ChatPromptTemplate.from_messages(
        [
            select_deepen_terms_prompts.get("system_message"),
            select_deepen_terms_prompts.get("main_prompt_template"),
        ]
    )

    # start of chain
    deepen_chain = LLMChain(llm=deepen_llm, prompt=deepen_chat_template)
    deepen_out_1 = deepen_chain.run(
        description=patient_description, current_hpo_terms=current_hpo_terms
    )
    logger.info(deepen_out_1)

    select_term_chain = create_openai_fn_chain(
        [pass_list], select_deepen_terms_llm, select_terms_chat_template
    )

    select_term_out = select_term_chain.run(AI_notes=deepen_out_1)
    logger.info(select_term_out)

    extract_excerpts_chain = LLMChain(
        llm=extract_excerpts_llm, prompt=extract_excerpts_chat_template
    )
    term_excerpt_child_combo = []
    for term in select_term_out.get("in_list"):
        # If the term from the generated list is not a real hpo term, we use most semantically similar term.
        try:
            child_terms = [entry["term"] for entry in child_terms_dict[term]]
        except KeyError:
            term = get_similar(term)
            try:
                child_terms = [entry["term"] for entry in child_terms_dict[term]]
            except KeyError:
                continue

        excerpts_out = extract_excerpts_chain.run(
            hpo_term=term, clinician_notes=input_dict.get("clinician_notes")
        )
        logger.info(excerpts_out)

        term_excerpt_child_combo.append(
            f"{term}:\n{excerpts_out}\nChild terms: {child_terms}"
        )

    term_excerpt_child_combo = "\n\n".join(term_excerpt_child_combo)
    logger.info(term_excerpt_child_combo)

    deepen_chain.prompt = ChatPromptTemplate.from_messages(
        [
            main_deepen_chain_prompts.get("system_message"),
            main_deepen_chain_prompts.get("initial_prompt_template"),
            main_deepen_chain_prompts.get("specify_prompt"),
            AIMessage(content=deepen_out_1),
            main_deepen_chain_prompts.get("form_ques_prompt_template"),
        ]
    )
    logger.info(deepen_chain.prompt)

    deepen_out_2 = deepen_chain.run(
        description=patient_description,
        current_hpo_terms=current_hpo_terms,
        term_excerpt_child_combo=term_excerpt_child_combo,
    )
    logger.info(deepen_out_2)

    return deepen_out_2


def run_broaden_chain(input_dict, llms, prompts, show_work=False):
    chat_prompt_template = initialize_broaden_chats(prompts)

    broaden_chain = LLMChain(llm=llms.get("broaden_llm"), prompt=chat_prompt_template)
    broaden_out_1 = broaden_chain.run(
        description=input_dict.get("patieint_description"),
        terms_list=input_dict.get("hpo_terms"),
    )
    logger.info(broaden_out_1)

    broaden_chain.prompt = ChatPromptTemplate.from_messages(
        chat_prompt_template.messages
        + [
            AIMessage(content=broaden_out_1),
            HumanMessage(content=prompts.get("prompt_2")),
        ]
    )

    broaden_out_2 = broaden_chain.run(
        description=input_dict.get("patieint_description"),
        terms_list=input_dict.get("hpo_terms"),
    )
    logger.info(broaden_out_2)

    broaden_chain.prompt = ChatPromptTemplate.from_messages(
        chat_prompt_template.messages
        + [
            AIMessage(content=broaden_out_2),
            HumanMessage(content=prompts.get("prompt_3")),
        ]
    )

    broaden_out_3 = broaden_chain.run(
        description=input_dict.get("patieint_description"),
        terms_list=input_dict.get("hpo_terms"),
    )
    logger.info(broaden_out_2)

    return broaden_out_3
