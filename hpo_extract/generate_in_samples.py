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
from langchain.chains import LLMChain
from langchain.chains.openai_functions import create_openai_fn_chain

from hpo_extract.funcs_for_llms import pass_list


# HPO_DIRECT_MODEL_NAME = "gpt-4"
HPO_DIRECT_MODEL_NAME = "gpt-3.5-turbo"
SELECT_TERMS_MODEL_NAME = "gpt-3.5-turbo"


def load_clin_from_hpo_prompts():
    with open("data/generate_in_samples/prompts/clin_from_hpo.json", "r") as f:
        promts = json.load(f)

    chats_init = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=promts.get("system_message")),
            HumanMessagePromptTemplate.from_template(promts.get("main_prompt")),
            HumanMessage(content=promts.get("second_prompt")),
        ]
    )

    return chats_init


def load_hpo_from_nord_prompts():
    with open("data/generate_in_samples/prompts/hpo_from_nord.json", "r") as f:
        prompts = json.load(f)

    chats_init = {}
    chats_init["main_prompts"] = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=prompts["main_prompts"]["system_message"]),
            HumanMessagePromptTemplate.from_template(
                prompts["main_prompts"]["prompt_1"]
            ),
            HumanMessage(content=prompts["main_prompts"]["prompt_2"]),
        ]
    )
    chats_init["select_terms_prompts"] = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=prompts["select_terms_prompts"]["system_message"]),
            HumanMessagePromptTemplate.from_template(
                prompts["select_terms_prompts"]["prompt"]
            ),
        ]
    )

    return chats_init


def load_clin_from_nord_prompts():
    with open("data/generate_in_samples/prompts/clin_from_nord.json", "r") as f:
        promts = json.load(f)

    chats_init = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=promts.get("system_message")),
            HumanMessagePromptTemplate.from_template(promts.get("main_prompt")),
            HumanMessagePromptTemplate.from_template(promts.get("second_prompt")),
        ]
    )

    return chats_init


def clin_from_hpo(hpo_terms, llm):
    chats_init = load_clin_from_hpo_prompts()

    chain = LLMChain(llm=llm, prompt=chats_init)
    clin_notes = chain.run(term_list=str(hpo_terms))

    return {"clin_notes": clin_notes, "og_hpo_terms_input": hpo_terms}


def clin_from_nord(nord_text_dict, llm):
    chats_init = load_clin_from_nord_prompts()

    chain = LLMChain(llm=llm, prompt=chats_init)
    logger.info(chain)
    clin_notes = chain.run(
        disease_name=nord_text_dict.get("disease_name"),
        disease_overview=nord_text_dict.get("disease_overview"),
        signs_symptoms=nord_text_dict.get("signs_symptoms"),
        causes=nord_text_dict.get("causes"),
        standard_therapies=nord_text_dict.get("standard_therapies"),
    )

    return {"clin_notes": clin_notes, "og_nord_input": nord_text_dict}


def hpo_from_nord(nord_text_dict, llms, prompts):
    chats_init = load_hpo_from_nord_prompts()
    main_llm = llms["main_llm"]
    select_terms_llm = llms["select_terms_llm"]

    chain = LLMChain(llm=main_llm, prompt=chats_init["main_prompts"])
    chain_out = chain.run(
        disease_name=nord_text_dict["disease_name"],
        disease_overview=nord_text_dict["disease_overview"],
        signs_symptoms=nord_text_dict["signs_symptoms"],
        causes=nord_text_dict["causes"],
    )
    logger.info(chain_out)

    select_terms_chain = create_openai_fn_chain(
        functions=[pass_list],
        llm=select_terms_llm,
        prompt=chats_init["select_terms_prompts"],
    )
    hpo_terms_uncorrected = select_terms_chain.run(ai_output=chain_out)
    logger.info(hpo_terms_uncorrected)

    return hpo_terms_uncorrected["in_list"]
