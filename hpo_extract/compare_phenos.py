import asyncio
import os
from loguru import logger
import json

from sklearn.metrics import pairwise_distances
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
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

from hpo_extract.setup_data import HPO_VECTORS, HPO_DF, HPO_TREE, TERM_IDS_VECS
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

    distance_matrix = pairwise_distances(a_vecs, b_vecs, metric="cosine")

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

    distance_matrix = pairwise_distances(a_vecs, b_vecs, metric="cosine")

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


def load_cluster_input(input_dir):
    all_sample_phenos = {}
    try:
        for filename in os.listdir(input_dir):
            if filename.endswith(".json"):
                with open(os.path.join(input_dir, filename), "r") as file:
                    data = json.load(file)
                    all_sample_phenos[filename[:-5]] = data
    except:
        # input is json
        with open(input_dir, "r") as file:
            all_sample_phenos = json.load(file)
    logger.info(all_sample_phenos)

    return all_sample_phenos


def make_graph(all_sample_phenos, sample_key, sim_thresh, cluster_factor):
    master_list = []
    mapping_key = {}
    for filename, data in all_sample_phenos.items():
        disease_name = data[sample_key].replace(" ", "_")
        for idx, terms_set in enumerate(data["hpo_terms_sets"]):
            master_list.append(terms_set)
            mapping_key[(len(master_list) - 1)] = f"{disease_name}_{idx+1}"

    matrix_size = len(master_list)
    matrix = [[0.0 for _ in range(matrix_size)] for _ in range(matrix_size)]

    for i in range(matrix_size):
        for j in range(matrix_size):
            if i != j:
                matrix[i][j] = calc_dist(
                    set(master_list[i]), set(master_list[j]), TERM_IDS_VECS
                )
    matrix_np = np.array(matrix)
    logger.info(mapping_key)

    matrix_flipped = matrix_np**-1
    matrix_flipped[matrix_flipped == np.inf] = 0

    matrix_flipped[matrix_flipped < sim_thresh] = 0

    matrix_scaled = matrix_flipped**cluster_factor

    G = nx.Graph()
    for key, value in mapping_key.items():
        G.add_node(key, label=value, disease=value)

    for i in range(matrix_size):
        for j in range(matrix_size):
            if i != j and matrix_scaled[i][j] != 0:
                G.add_edge(i, j, weight=matrix_scaled[i][j])
    logger.info(G)

    return G


def calc_sim(list_a, list_b, ic_a, ic_b):
    distance_matrix = get_distance_matrix(list_a, list_b, TERM_IDS_VECS)
    ic_matrix = ic_b + ic_a[:, None]
    ic_matrix[ic_matrix == 0] = -1
    ic_matrix = -((-ic_matrix) ** 0.5)
    similarity_matrix = -10 * ((distance_matrix + 0.001) * 10 * (ic_matrix)) ** -1

    norm = np.linalg.norm(similarity_matrix)
    return norm / np.sqrt(len(list_a) * len(list_b))


def get_term_freq(data):
    term_freq = {}
    for key, value in data.items():
        for term_set in value["hpo_terms_sets"]:
            for term in term_set:
                if term in term_freq:
                    term_freq[term] += 1
                else:
                    term_freq[term] = 1
    return term_freq


def ic_list(term_list, freq_data):
    ic_list = []
    for term in term_list:
        ic_list.append(freq_data[term])
    return -np.log2(np.array(ic_list))


def make_sim_graph(all_sample_phenos, sample_key, sim_thresh, cluster_factor):
    master_list = []
    master_list_ic = []
    mapping_key = {}
    term_freq = get_term_freq(all_sample_phenos)

    for filename, data in all_sample_phenos.items():
        disease_name = data[sample_key].replace(" ", "_")
        for idx, terms_set in enumerate(data["hpo_terms_sets"]):
            terms_list = list(terms_set)
            master_list.append(terms_list)
            master_list_ic.append(ic_list(terms_list, term_freq))
            mapping_key[(len(master_list) - 1)] = f"{disease_name}_{idx+1}"

    matrix_size = len(master_list)
    matrix = [[0.0 for _ in range(matrix_size)] for _ in range(matrix_size)]

    for i in range(matrix_size):
        for j in range(matrix_size):
            if i != j:
                matrix[i][j] = calc_sim(
                    master_list[i], master_list[j], master_list_ic[i], master_list_ic[j]
                )
    matrix_np = np.array(matrix)
    logger.info(mapping_key)

    matrix_np[matrix_np < sim_thresh] = 0

    matrix_scaled = matrix_np**cluster_factor

    G = nx.Graph()
    for key, value in mapping_key.items():
        G.add_node(key, label=value, disease=value)

    for i in range(matrix_size):
        for j in range(matrix_size):
            if i != j and matrix_scaled[i][j] != 0:
                G.add_edge(i, j, weight=matrix_scaled[i][j])
    logger.info(G)

    return G


def save_graph(G, all_sample_phenos, sample_key, out_png):
    colors = list(mcolors.CSS4_COLORS.keys())
    diseases = list(set([data[sample_key] for data in all_sample_phenos.values()]))
    color_map = {disease: colors[i] for i, disease in enumerate(diseases)}
    node_colors = [
        color_map[" ".join(G.nodes[node]["disease"].split("_")[:-1])]
        for node in G.nodes()
    ]

    pos = nx.spring_layout(G, weight="weight", seed=13)
    labels = {node: data["label"] for node, data in G.nodes(data=True)}
    nx.draw(
        G,
        pos,
        labels=labels,
        node_color=node_colors,
        with_labels=True,
        node_size=500,
        font_size=8,
    )
    plt.savefig(out_png, format="PNG")
