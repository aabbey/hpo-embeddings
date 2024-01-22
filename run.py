from loguru import logger
import sys
import json
import os
import glob
import asyncio

from langchain.chat_models import ChatOpenAI
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import leidenalg
import igraph as ig

# import argparse
import fire

from hpo_extract import hpo_to_follow_up, extract, generate_in_samples
from hpo_extract import compare_phenos
from hpo_extract.setup_data import TERM_IDS_VECS


def run_hpo_to_follow_up(input_dict_path, out_path=""):
    """Creates follow up questions that will broaden and deepen our understanding of the patient phenotype

    Args:
        input_dict_path (str): contains patieint_description and hpo_terms.
        out_path (str): output json with deepen questions and broaden questions.

    Returns:
        _type_: _description_
    """
    with open(input_dict_path, "r") as f:
        input_dict = json.load(f)

    broaden_llm = ChatOpenAI(model=hpo_to_follow_up.MODEL_NAME, temperature=0.0)
    llms = {"broaden_llm": broaden_llm}
    with open("data/hpo_to_follow_up/prompts/broaden_prompts.json", "r") as f:
        broaden_prompts = json.load(f)

    broaden_chain_output = hpo_to_follow_up.run_broaden_chain(
        input_dict=input_dict, llms=llms, prompts=broaden_prompts
    )

    # Deepen Chain
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

    if not out_path == "":
        out_json = {
            "deepen_chain_output": deepen_chain_output,
            "broaden_chain_output": broaden_chain_output,
        }
        with open(out_path, "w") as f:
            json.dump(out_json, f)

    return deepen_chain_output, broaden_chain_output


def run_extract(input_text, out_path=""):
    extract_llm = ChatOpenAI(model=extract.MODEL_NAME, temperature=0)
    term_list_llm = ChatOpenAI(model=extract.TERM_LIST_MODEL_NAME, temperature=0)

    llms = {"extract_llm": extract_llm, "term_list_llm": term_list_llm}

    all_extract_prompts = extract.load_prompts()

    extract_chain_output = extract.run_extract_chain(
        in_text=input_text, llms=llms, prompts=all_extract_prompts
    )

    if not out_path == "":
        with open(out_path, "w") as f:
            json.dump(extract_chain_output, f)

    return extract_chain_output


def run_make_description(input_text):
    desc_llm = ChatOpenAI(model=extract.DESC_MODEL_NAME, temperature=0)

    llms = {"ex_desc_llm": desc_llm}

    all_extract_prompts = extract.load_desc_prompts()

    desc_chain_output = extract.run_extract_desc_chain(
        in_text=input_text, llms=llms, prompts=all_extract_prompts
    )

    return desc_chain_output


def run_make_input_dict(input_text_path, hpo_terms_path=None, out_path=""):
    input_dict = {}
    with open(input_text_path, "r") as f:
        input_text = f.read()

    if hpo_terms_path:
        with open(hpo_terms_path, "r") as f:
            hpo_terms = json.load(f)
    else:
        hpo_terms = run_extract(input_text)
    desc = run_make_description(input_text)

    input_dict["patieint_description"] = desc
    input_dict["hpo_terms"] = list(hpo_terms)
    input_dict["clinician_notes"] = input_text

    if not out_path == "":
        with open(out_path, "w") as f:
            json.dump(input_dict, f)

    return input_dict


def create_clin_notes(src, from_src="hpo_terms", save=False):
    valid_src = {"hpo_terms", "nord"}
    if from_src not in valid_src:
        raise ValueError(
            f"Invalid source '{from_src}'. Expected one of {', '.join(valid_src)}"
        )

    if from_src == "hpo_terms":
        llm = ChatOpenAI(model=extract.MODEL_NAME, temperature=0.7)
        clin_notes_sample = generate_in_samples.clin_from_hpo(
            src, llm
        )  # dict, {"clin_notes":"...text...", "other_stuff":...}
    elif from_src == "nord":
        llm = ChatOpenAI(model=extract.MODEL_NAME, temperature=0.7)
        clin_notes_sample = generate_in_samples.clin_from_nord(
            src, llm
        )  # dict, {"clin_notes":"...text...", "other_stuff":...}
        clin_notes_sample["disease_name"] = src.get("disease_name")

    if save:
        base_path = "data/generate_in_samples/out_samples/"
        base_name = clin_notes_sample.get("disease_name").replace(" ", "_")
        ext = ".json"
        full_path = os.path.join(base_path, base_name + ext)

        counter = 1
        while os.path.exists(full_path):
            full_path = os.path.join(base_path, f"{base_name}_{counter}{ext}")
            counter += 1

        with open(full_path, "w") as f:
            json.dump(clin_notes_sample, f)

    return clin_notes_sample


def create_follow_up_samples(clin_notes):
    pass


def sync_to_async_adapter(async_function, *args, **kwargs):
    if "IPython" in sys.modules:
        # We're in a Jupyter Notebook
        import nest_asyncio

        nest_asyncio.apply()
        # Create a new event loop for this specific call and run the async function on it
        loop = asyncio.new_event_loop()
        return loop.run_until_complete(async_function(*args, **kwargs))
    else:
        # We're outside Jupyter, so just run the async function normally
        return asyncio.run(async_function(*args, **kwargs))


def create_hpo_direct(nord_text, save_path=""):
    llms = {
        "main_llm": ChatOpenAI(
            model=generate_in_samples.HPO_DIRECT_MODEL_NAME, temperature=1.0
        ),
        "select_terms_llm": ChatOpenAI(
            model=generate_in_samples.SELECT_TERMS_MODEL_NAME, temperature=0.0
        ),
    }
    with open("data/generate_in_samples/prompts/hpo_from_nord.json", "r") as f:
        prompts = json.load(f)
    hpo_terms_out = generate_in_samples.hpo_from_nord(nord_text, llms, prompts)

    hpo_terms_corrected = sync_to_async_adapter(
        compare_phenos.correct_invalid_terms, set(hpo_terms_out)
    )

    return hpo_terms_corrected


def run_create_hpo_direct(nord_path):
    with open(nord_path, "r") as f:
        nord_dict = json.load(f)

    hpo_terms = create_hpo_direct(nord_dict)

    save_terms_dict = {
        "disease_name": nord_dict["disease_name"],
        "hpo_terms": list(hpo_terms),
    }

    return save_terms_dict


async def run_create_hpo_direct_async(nord_path):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, run_create_hpo_direct, nord_path)


async def run_multiple_create_hpo_direct(input_folder, output_folder, n):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    input_files = glob.glob(os.path.join(input_folder, "*.json"))
    logger.info(input_files)

    for input_file in input_files:
        print(f"Processing {input_file}")
        tasks = [run_create_hpo_direct_async(input_file) for _ in range(n)]
        results = await asyncio.gather(*tasks)

        output_data = {
            "disease_name": results[0]["disease_name"],
            "hpo_terms_sets": [result["hpo_terms"] for result in results],
        }
        base_name = os.path.basename(input_file)
        output_file = os.path.join(output_folder, base_name)
        logger.info(output_file)
        with open(output_file, "w") as f:
            json.dump(output_data, f)


def run_hpo_graph_cluster(
    input_dir, out_png, sample_key, metrics="default", cluster_factor=4, sim_thresh=0.95
):
    """_summary_

    Args:
        input_dir (_type_): _description_
        out_png (_type_): _description_
        sample_key (_type_): _description_
        metrics (_type_): _description_
        cluster_factor (int, optional): _description_. Defaults to 4.
        sim_thresh (float, optional): _description_. Defaults to 0.95.
    """
    all_sample_phenos = compare_phenos.load_cluster_input(input_dir=input_dir)

    if metrics == "IC":
        graph = compare_phenos.make_sim_graph(
            all_sample_phenos=all_sample_phenos,
            sample_key=sample_key,
            sim_thresh=sim_thresh,
            cluster_factor=cluster_factor,
        )
    else:
        graph = compare_phenos.make_graph(
            all_sample_phenos=all_sample_phenos,
            sample_key=sample_key,
            sim_thresh=sim_thresh,
            cluster_factor=cluster_factor,
        )

    compare_phenos.save_graph(
        all_sample_phenos=all_sample_phenos,
        G=graph,
        sample_key=sample_key,
        out_png=out_png,
    )


if __name__ == "__main__":
    fire.Fire(
        {
            "follow_up": run_hpo_to_follow_up,
            "extract": run_extract,
            "make_input_dict": run_make_input_dict,
            "create_clin_notes": create_clin_notes,
        }
    )
