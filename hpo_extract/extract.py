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
from langchain.chat_models import ChatOpenAI
from langchain.chains.openai_functions import create_openai_fn_chain
from langchain.chains import LLMChain

from hpo_extract.setup_data import HPO_VECTORS, HPO_DF, HPO_TREE
from hpo_extract.funcs_for_llms import pass_list, get_term


# MODEL_NAME = "gpt-3.5-turbo"
# TERM_LIST_MODEL_NAME = "gpt-3.5-turbo"
# SIM_SEARCH_MODEL = "gpt-3.5-turbo"
MODEL_NAME = "gpt-4"
TERM_LIST_MODEL_NAME = "gpt-4"
SIM_SEARCH_MODEL = "gpt-4"


def load_prompts():
    with open("data/extract/prompts/extract_prompts.json", "r") as f:
        extract_prompts = json.load(f)

    return extract_prompts


def load_input():
    with open("data/extract/input_samples/sample1_clin_notes.txt", "r") as f:
        clin_notes_input = f.read()

    return clin_notes_input


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


def can_be_formatted(s: str) -> bool:
    try:
        s.format()
        return False
    except:
        return True


def augmented_sim_search(text, llm, k_length=15):
    sysmsg = "You are a midecal expert that understands phenotypes and HPO terms."
    prompt_temp = "A report has proposed an HPO term for a patient phenotype that says: '{wrong_term}', however that is not a term that is currently in the HPO database. Take your best guess as to which of these actual HPO terms is most likely meant by the proposed term. Terms to choose from: {sim_terms_list}\nYou must choose exactly one term from the list."

    sim_terms_list = [
        HPO_VECTORS.similarity_search(text, k=k_length)[x].page_content
        for x in range(k_length)
    ]

    prompt_temp = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template("{sysmsg}"),
            HumanMessagePromptTemplate.from_template(prompt_temp),
        ]
    )
    select_term_chain = create_openai_fn_chain([get_term], llm, prompt_temp)
    response = select_term_chain.run(
        sysmsg=sysmsg, wrong_term=text, sim_terms_list=sim_terms_list
    )
    return response.get("single_hpo_term")


def correct_invalid_terms(term_set):
    wrong_terms = term_set - set(HPO_DF["lbl"])
    valid_terms = term_set - wrong_terms
    sim_search_llm = ChatOpenAI(model=SIM_SEARCH_MODEL, temperature=0)

    for term in wrong_terms:
        valid_terms.add(augmented_sim_search(text=term, llm=sim_search_llm))

    return valid_terms


def run_extract_chain(in_text, llms, prompts, show_work=False):
    """Runs chain to extract HPO terms from clinitian notes.

    Args:
        in_text (str): clinitian notes
        llms (dict): each llm used in chain
        prompts (dict): all prompts (as strings) to be used
        show_work (bool, optional): llms explain themselves along the way. Defaults to False.

    Returns:
        set: output of chain, set of valid terms
    """
    all_prompts = initialize_llm_chats(
        prompts
    )  # eg. {llm1:{sysmsg:SystemMessege(), p:HumanMessege(), pt:...}, llm2:{...}...}
    main_extract_prompts = all_prompts["main_extract_llm"]
    term_list_prompts = all_prompts["get_term_list_llm"]

    extract_llm = llms.get("extract_llm")
    term_list_llm = llms.get("term_list_llm")

    extract_chat_template = ChatPromptTemplate.from_messages(
        [
            main_extract_prompts.get("system_message"),
            main_extract_prompts.get("prompt"),
        ]
    )

    term_list_chat_template = ChatPromptTemplate.from_messages(
        [
            term_list_prompts.get("system_message"),
            term_list_prompts.get("prompt"),
        ]
    )

    # start of chain
    extract_chain = LLMChain(llm=extract_llm, prompt=extract_chat_template)
    extract_out_1 = extract_chain.run(clinician_notes=in_text)
    logger.info(extract_out_1)

    term_list_chain = create_openai_fn_chain(
        [pass_list], term_list_llm, term_list_chat_template
    )
    term_output = term_list_chain.run(term_evidence=extract_out_1)
    logger.info(term_output)

    corrected_terms = correct_invalid_terms(set(term_output.get("in_list")))
    logger.info(corrected_terms)

    return corrected_terms
