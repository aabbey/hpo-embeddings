from loguru import logger

from langchain.chat_models import ChatOpenAI

from hpo_extract import hpo_to_follow_up
from hpo_extract import extract


def run_hpo_to_follow_up(input_dict):
    deepen_llm = ChatOpenAI(model=hpo_to_follow_up.MODEL_NAME, temperature=0.0)
    extract_excerpts_llm = ChatOpenAI(
        model=hpo_to_follow_up.EXTRACT_EXCERPTS_MODEL_NAME, temperature=0.0
    )
    select_deepen_terms_llm = ChatOpenAI(
        model=hpo_to_follow_up.SELECT_DEEPEN_TERMS_MODEL_NAME, temperature=0.0
    )
    llms = {
        "deepen_llm": deepen_llm,
        "extract_excerpts_llm": extract_excerpts_llm,
        "select_deepen_terms_llm": select_deepen_terms_llm,
    }
    all_deepen_llm_prompts = hpo_to_follow_up.load_prompts()
    deepen_chain_output = hpo_to_follow_up.run_deepen_chain(
        input_dict=input_dict,
        llms=llms,
        prompts=all_deepen_llm_prompts,
    )

    return deepen_chain_output


def run_extract(input_text):
    extract_llm = ChatOpenAI(model=extract.MODEL_NAME, temperature=0)
    term_list_llm = ChatOpenAI(model=extract.TERM_LIST_MODEL_NAME, temperature=0)

    llms = {"extract_llm": extract_llm, "term_list_llm": term_list_llm}

    all_extract_prompts = extract.load_prompts()

    extract_chain_output = extract.run_extract_chain(
        in_text=input_text, llms=llms, prompts=all_extract_prompts
    )

    return extract_chain_output


if __name__ == "__main__":
    clin_notes = extract.load_input()
    extract_output = run_extract(clin_notes)

    print(extract_output)

    hpo_to_follow_up_input = hpo_to_follow_up.load_input()
    deepen_chain_output = run_hpo_to_follow_up(hpo_to_follow_up_input)

    print(deepen_chain_output)
