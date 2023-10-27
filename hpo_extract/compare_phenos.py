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
        a_vecs[i] = term_ids_vecs[a]["vector"]
    for i, b in enumerate(set_b):
        b_vecs[i] = term_ids_vecs[b]["vector"]

    distance_matrix = pairwise_distances(a_vecs, b_vecs, metric="euclidean")

    a_min = np.min(distance_matrix, axis=0)
    b_min = np.min(distance_matrix, axis=1)

    loss = np.linalg.norm(a_min) / np.sqrt(len(a_min)) + np.linalg.norm(
        b_min
    ) / np.sqrt(len(b_min))

    return loss


def get_distance_matrix(set_a, set_b, term_ids_vecs):
    a_vecs = np.zeros((len(set_a), 1536))
    b_vecs = np.zeros((len(set_b), 1536))

    for i, a in enumerate(set_a):
        a_vecs[i] = term_ids_vecs[a]["vector"]
    for i, b in enumerate(set_b):
        b_vecs[i] = term_ids_vecs[b]["vector"]

    distance_matrix = pairwise_distances(a_vecs, b_vecs, metric="euclidean")

    return distance_matrix


def make_terms_dist_table(unique_terms1, unique_terms2, distance_matrix):
    """makes table to compare 2 sets of terms

    Args:
        terms1 (dict): "sample_name": "sample", "hpo_terms": [terms]
        terms2 (dict): "sample_name": "sample", "hpo_terms": [terms]
        distance_matrix (np array): distance metric between each term in the 2 inputs
    """

    list1 = unique_terms1["hpo_terms"]
    list2 = unique_terms2["hpo_terms"]
    distance_matrix_og = distance_matrix.copy()

    table = {
        unique_terms1["sample_name"]: [],
        unique_terms2["sample_name"]: [],
        "dist": [],
    }

    for i in range(min(np.shape(distance_matrix))):
        x, y = np.unravel_index(distance_matrix.argmin(), distance_matrix.shape)
        table[unique_terms1["sample_name"]].append(list1[x])
        table[unique_terms2["sample_name"]].append(list2[y])
        table["dist"].append(distance_matrix[x][y])
        list1 = np.delete(list1, x)
        list2 = np.delete(list2, y)
        distance_matrix = np.delete(distance_matrix, x, axis=0)
        distance_matrix = np.delete(distance_matrix, y, axis=1)

    if len(list1) != 0:
        for term in list1:
            table[unique_terms1["sample_name"]].append(term)
            table[unique_terms2["sample_name"]].append("")
            table["dist"].append(np.nan)
    if len(list2) != 0:
        for term in list2:
            table[unique_terms2["sample_name"]].append(term)
            table[unique_terms1["sample_name"]].append("")
            table["dist"].append(np.nan)

    return table


def find_connected_components(adj_matrix):
    def dfs(node, visited, component_nodes):
        visited[node] = True
        component_nodes.append(node)
        for neighbor, isConnected in enumerate(adj_matrix[node]):
            if isConnected and not visited[neighbor]:
                dfs(neighbor, visited, component_nodes)

    n = len(adj_matrix)
    visited = [False] * n
    components = 0

    for i in range(n):
        if not visited[i]:
            component_nodes = []
            dfs(i, visited, component_nodes)
            if len(component_nodes) > 1:
                components += 1

    return components
