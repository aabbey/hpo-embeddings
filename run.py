from langchain.chat_models import ChatOpenAI

from hpo_extract import hpo_to_follow_up


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


if __name__ == "__main__":
    hpo_to_follow_up_input = hpo_to_follow_up.load_input()

    deepen_chain_output = run_hpo_to_follow_up(hpo_to_follow_up_input)

    print(deepen_chain_output)
