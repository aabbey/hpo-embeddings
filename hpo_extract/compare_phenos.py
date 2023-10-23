import asyncio
from loguru import logger

from sklearn.metrics import pairwise_distances
import numpy as np
from langchain.chat_models import ChatOpenAI
from langchain.chains.openai_functions import create_openai_fn_chain
from langchain.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

from hpo_extract.setup_data import (
    HPO_VECTORS,
    HPO_DF,
    HPO_TREE,
)
from hpo_extract.funcs_for_llms import get_term

# SIM_SEARCH_MODEL = "gpt-4"
SIM_SEARCH_MODEL = "gpt-3.5-turbo"


async def async_augmented_sim_search(text, llm, k_length=15):
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(None, augmented_sim_search, text, llm, k_length)
    return result


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
    logger.info(text)
    logger.info(sim_terms_list)
    logger.info(select_term_chain)

    response = select_term_chain.run(
        sysmsg=sysmsg, wrong_term=text, sim_terms_list=sim_terms_list
    )
    logger.info(response)

    return response["single_hpo_term"]


async def correct_invalid_terms(term_set):
    wrong_terms = term_set - set(HPO_DF["lbl"])
    valid_terms = term_set - wrong_terms
    sim_search_llm = ChatOpenAI(model=SIM_SEARCH_MODEL, temperature=0)
    logger.info(wrong_terms)

    async def get_close_term(term):
        close_term = await async_augmented_sim_search(text=term, llm=sim_search_llm)
        if close_term in HPO_DF["lbl"].values:
            valid_terms.add(close_term)

    tasks = [get_close_term(term) for term in wrong_terms]
    await asyncio.gather(*tasks)

    return valid_terms


def calc_dist(set_a, set_b, term_ids_vecs):
    """pheno distance for comparing two phenotypes (sets of HPO terms)

    Args:
        set_a (set): hpo terms a
        set_b (set): hpo terms b
        term_ids_vecs (dict): all the hpo terms and their vectors

    Returns:
        _type_: _description_
    """
    a_vecs = np.zeros((len(set_a), 1536))
    b_vecs = np.zeros((len(set_b), 1536))

    for i, a in enumerate(set_a):
        try:
            a_vecs[i] = term_ids_vecs[a]["vector"]
        except KeyError:
            llm = ChatOpenAI(model=SIM_SEARCH_MODEL, temperature=0)
            key = augmented_sim_search(a, llm)
            try:
                a_vecs[i] = term_ids_vecs[key]["vector"]
            except KeyError:
                key = HPO_VECTORS.similarity_search(key, k=1)[0].page_content
                a_vecs[i] = term_ids_vecs[key]["vector"]
    for i, b in enumerate(set_b):
        try:
            b_vecs[i] = term_ids_vecs[b]["vector"]
        except KeyError:
            llm = ChatOpenAI(model=SIM_SEARCH_MODEL, temperature=0)
            key = augmented_sim_search(b, llm)
            try:
                b_vecs[i] = term_ids_vecs[key]["vector"]
            except KeyError:
                key = HPO_VECTORS.similarity_search(key, k=1)[0].page_content
                b_vecs[i] = term_ids_vecs[key]["vector"]

    distance_matrix = pairwise_distances(a_vecs, b_vecs, metric="euclidean")

    a_min = np.min(distance_matrix, axis=0)
    b_min = np.min(distance_matrix, axis=1)

    loss = np.linalg.norm(a_min) / np.sqrt(len(a_min)) + np.linalg.norm(
        b_min
    ) / np.sqrt(len(b_min))

    return loss
