from loguru import logger
import sys
import json
import os
import glob
import asyncio

from langchain.chat_models import ChatOpenAI

from hpo_extract import hpo_to_follow_up, extract, generate_in_samples, compare_phenos


def run_hpo_to_follow_up(input_dict):
    broaden_llm = ChatOpenAI(model=hpo_to_follow_up.MODEL_NAME, temperature=0.0)
    llms = {"broaden_llm": broaden_llm}
    with open("data/hpo_to_follow_up/prompts/broaden_prompts.json", "r") as f:
        broaden_prompts = json.load(f)

    broaden_chain_output = hpo_to_follow_up.run_broaden_chain(
        input_dict=input_dict, llms=llms, prompts=broaden_prompts
    )

    print(broaden_chain_output)
    1 / 0

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


if __name__ == "__main__":
    pass
