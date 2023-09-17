
import sys
import re
import openai
from langchain import LLMChain, PromptTemplate
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
from langchain.prompts import SystemMessagePromptTemplate
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Chroma, Pinecone
import numpy as np
import json
import os
import pandas as pd
import pinecone

llm_model_name = "gpt-3.5-turbo"
#llm_model_name = "text-davinci-003"
MAX_LAYER_DEPTH = 10

with open('prompts/hpo_extractor_template') as t:
    template = t.readlines()
    template = ''.join(template)


def make_hpo_dataframe():
    def gather_definition(meta_dict):
        if isinstance(meta_dict, dict):
            if 'definition' in meta_dict.keys():
                return meta_dict['definition']['val']
            else:
                return np.nan
        else:
            return np.nan

    def gather_comments(meta_dict):
        if isinstance(meta_dict, dict):
            if 'comments' in meta_dict.keys():
                return meta_dict['comments']
            else:
                return np.nan
        else:
            return np.nan

    def str_to_embed(df_row):
        string_template = ""
        if pd.notnull(df_row['lbl']):
            string_template += f"label: {df_row['lbl']} \n"
        if pd.notnull(df_row['definition']):
            string_template += f"definition: {df_row['definition']} \n"
        if pd.notnull(df_row['comments']):
            string_template += "comments: "
            for comment in df_row['comments']:
                string_template += f"{comment} \n"
        return string_template

    with open('hp.json') as f:
        data = json.load(f)

    hpo_data = data['graphs'][0]['nodes']

    hpo_data_df = pd.DataFrame(hpo_data)

    simple_hpo_df = pd.DataFrame(columns=['id', 'lbl', 'definition', 'comments'])

    simple_hpo_df['id'] = hpo_data_df['id']
    simple_hpo_df['type'] = hpo_data_df['id'].apply(lambda x: str(x)[-11:-8])
    simple_hpo_df['lbl'] = hpo_data_df['lbl']
    simple_hpo_df['definition'] = hpo_data_df['meta'].apply(gather_definition)
    simple_hpo_df['comments'] = hpo_data_df['meta'].apply(gather_comments)

    simple_hpo_df['text_to_embed'] = simple_hpo_df.apply(str_to_embed, axis=1)

    return simple_hpo_df


def make_hpo_tree():
    class Tree:
        def __init__(self):
            self.tree = {}
            self.parents = {}

        def add_edge(self, sub, obj):
            if obj in self.tree:
                self.tree[obj].add(sub)
            else:
                self.tree[obj] = set([sub])

            self.parents[sub] = obj

        def get_children(self, id):
            if id in self.tree:
                return self.tree[id]
            else:
                return []

        def get_parent(self, id):
            if id in self.parents:
                return self.parents[id]
            else:
                return None

    def create_tree_from_list(list_of_dicts):
        tree = Tree()
        for dic in list_of_dicts:
            tree.add_edge(dic["sub"], dic["obj"])
        return tree

    with open('hp.json') as f:
        data = json.load(f)

    edges_list = data['graphs'][0]['edges']
    tree_of_ids = create_tree_from_list(edges_list)
    return tree_of_ids


def specify_term(hpo_term):
    """
    use this to find children of given hpo term to find a more specific term to match to the patient
    :param hpo_term: parent term
    :return: list of child terms
    """
    parent_term_id = hpo_df.loc[hpo_df['lbl'] == hpo_term, 'id'].values[0]
    child_id_list = list(hpo_tree.get_children(parent_term_id))
    child_list = hpo_df.loc[hpo_df['id'].isin(child_id_list), 'lbl'].values

    return child_list


def extract_terms(transcript, verbose=True):
    list_of_extracted_terms = []
    if llm_model_name == "text-davinci-003":
        llm = OpenAI(model_name=llm_model_name)
    else:
        llm = ChatOpenAI(model_name=llm_model_name)
        list_of_extracted_terms = []
        for line in transcript.split("\n"):
            print(line)
            if line.startswith("Parent:"):
                if verbose:
                    print("--------------------\n\n", line, '\n\n')

                tree_searching = True
                ids_to_search = ['http://purl.obolibrary.org/obo/HP_0000118']
                loop_num = 0
                while tree_searching:
                    if verbose:
                        print(f"Searching tree loop {loop_num} ...\n")

                    for id in ids_to_search:
                        child_list = list(hpo_tree.get_children(id))

                        child_list_docs = hpo_df['text_to_embed'].loc[hpo_df['id'].isin(child_list)].to_list()
                        concat_doc = ""
                        for doc in child_list_docs:
                            concat_doc += doc + "\n\n"

                        if verbose:
                            print(concat_doc)

                        prompt = PromptTemplate(
                            input_variables=["parents_text", "hpo_doc"],
                            template=template,
                        )

                        chain = LLMChain(llm=llm, prompt=prompt)

                        # Run the chain only specifying the input variable.
                        llm_response = chain.run({"parents_text": line, "hpo_doc": concat_doc})
                        next_ids = []
                        for term in llm_response.split(', '):
                            term = term.strip()
                            if term.lower().rstrip('.') != 'none':
                                next_ids.append(hpo_df['id'].where(hpo_df['lbl'] == term))
                            else:
                                if ids_to_search[0] == 'http://purl.obolibrary.org/obo/HP_0000118':
                                    tree_searching = False
                                    break
                                else:
                                    list_of_extracted_terms.append(hpo_df['lbl'].where(hpo_df['id'] == id))
                        ids_to_search.remove(id)
                        ids_to_search += next_ids
                        if not ids_to_search:
                            tree_searching = False
                        if verbose:
                            print(llm_response)
        set_of_extracted_terms = set(list_of_extracted_terms)
        return set_of_extracted_terms



    embedding = OpenAIEmbeddings(openai_api_key=os.environ['OPENAI_API_KEY'])
    pinecone.init(
        api_key=os.environ['PINECONE_API_KEY'],
        environment=os.environ['PINECONE_API_ENV']
    )
    index_name = "hpo-embeddings"
    index = pinecone.Index(index_name)

    vectorstore = Pinecone(index=index, embedding_function=embedding.embed_query, text_key='text')

    for line in transcript.split("\n"):
        print(line)
        if line.startswith("Parent:"):
            if verbose:
                print("--------------------\n\n", line, '\n\n')
            similar_terms = vectorstore.similarity_search(line, k=10)

            concat_doc = ""
            for doc in similar_terms:
                concat_doc += doc.page_content + "\n\n"

            if verbose:
                print(concat_doc)

            prompt = PromptTemplate(
                input_variables=["parents_text", "hpo_doc"],
                template=template,
            )

            chain = LLMChain(llm=llm, prompt=prompt)

            # Run the chain only specifying the input variable.
            llm_response = chain.run({"parents_text": line, "hpo_doc": concat_doc})
            for term in llm_response.split(', '):
                term = term.strip()
                if term.lower().rstrip('.') != 'none':
                    list_of_extracted_terms.append(term)
            if verbose:
                print(llm_response)

    set_of_extracted_terms = set(list_of_extracted_terms)
    if verbose:
        print("\n----------------------------\n\n")
        print("Extracted terms : ", set(list_of_extracted_terms))
    return set_of_extracted_terms


def terms_to_docs(term_list):
    under_docs = hpo_df.loc[hpo_df['lbl'].isin(term_list), 'text_to_embed'].values
    return "\n\n".join(under_docs)


def most_similar(bad_term):
    """
    use to retrieve a term in our hpo_df that is semantically similar to the bad_term
    """
    similar_term_doc = vector_store.similarity_search(bad_term, k=1)[0]
    similar_term = hpo_df.loc[hpo_df['text_to_embed'] == similar_term_doc.page_content, 'lbl'].values[0]
    return similar_term


def layer_loop(parent_text, current_upper_term):
    """
    use this to get llm reasoning about the parent text line in context of the tree of hpo terms.
    it decides whether to go down another level to find the most specific term.
    :param parent_text: line from the parent
    :param current_upper_term: general term about patient phenotype
    :return: more specific term, and flag to stop loop
    """
    under_terms = specify_term(current_upper_term)
    if not under_terms.any():
        return current_upper_term, True
    concat_doc = terms_to_docs(under_terms)
    cur_term_doc = terms_to_docs([current_upper_term])
    concat_doc = cur_term_doc + "\n\n" + concat_doc

    ch4_hist[0] = sys_msg_prompt4.format_messages(cur_trait_doc=cur_term_doc, concated_under_term_docs=concat_doc)[0]
    response4 = llm(ch4_hist)
    print(response4.content)
    clean_response4 = response4.content.strip().lstrip("label: ").rstrip(".")

    # correct if not a real term
    if clean_response4 not in hpo_df['lbl'].unique():
        print("Picking close hpo term...")
        clean_response4 = most_similar(clean_response4)
        print(clean_response4)

    return clean_response4, False



if __name__ == "__main__":
    # create vector_store for semantic similarity search later
    embedding = OpenAIEmbeddings(openai_api_key=os.environ['OPENAI_API_KEY'])
    pinecone.init(
        api_key=os.environ['PINECONE_API_KEY'],
        environment=os.environ['PINECONE_API_ENV']
    )
    index_name = "hpo-embeddings"
    index = pinecone.Index(index_name)
    vector_store = Pinecone(index=index, embedding_function=embedding.embed_query, text_key='text')

    with open('samples/sample generated transcripts/gpt4-sample1-Biotinidase.json', 'r') as json_file:
        data = json.load(json_file)
    with open("prompts/etrct_prompt_1.txt", 'r') as p:
        prompt1 = p.read()
    with open("prompts/extract_prompt_2.txt", 'r') as p:
        prompt2 = p.read()
    with open("prompts/extract_prompt_3.txt", 'r') as p:
        prompt3 = p.read()
    with open("prompts/extract_prompt_4.txt", 'r') as p:
        prompt4 = p.read()

    transcript = data['transcript']
    true_terms = set(data['true_terms'])

    hpo_tree = make_hpo_tree()
    hpo_df = make_hpo_dataframe()

    llm = ChatOpenAI(model_name=llm_model_name, temperature=0)
    list_of_extracted_terms = []

    # start the first llm with a sys message. the loop bellow will then add human msgs
    sys_msg_prompt = SystemMessage(content=prompt1)
    ch1_hist = [sys_msg_prompt]
    sys_msg_prompt2 = SystemMessage(content=prompt2)
    ch2_hist = [sys_msg_prompt2]
    sys_msg_prompt3 = SystemMessagePromptTemplate.from_template(prompt3)
    ch3_hist = [sys_msg_prompt3]
    sys_msg_prompt4 = SystemMessagePromptTemplate.from_template(prompt4)
    ch4_hist = [sys_msg_prompt4]

    extracted_terms = []
    for line in transcript.split("\n"):
        print(line)
        if line.startswith("Parent:"):
            print("--------------------\n\n", line, '\n\n')

            ch1_hist.append(HumanMessage(content=line))
            response = llm(ch1_hist)
            print(response.content)
            ch1_hist.append(response)

            if response.content.lower().strip().rstrip('.') == "yes":
                ch2_hist.append(HumanMessage(content=line))
                response2 = llm(ch2_hist)
                print(response2.content)
                ch2_hist.append(response2)

                clean_response2 = response2.content.lower().strip().rstrip('.')
                if clean_response2 != "one":
                    ch3_hist[0] = sys_msg_prompt3.format_messages(number=clean_response2)[0]
                    ch3_hist.append(HumanMessage(content=line))
                    response3 = llm(ch3_hist)
                    print(response3.content)
                    ch3_hist.append(response3)

                    # remove prepending "line num: " from llm output
                    response3 = re.sub(r"line \d: ", "", response3.content)
                    response_hits = [resp for resp in response3.split('\n')]
                else:
                    response_hits = [response2.content]

                for resp in response_hits:
                    current_upper_term = "Phenotypic abnormality"
                    ch4_hist.append(HumanMessage(content=resp))
                    for layer in range(1, MAX_LAYER_DEPTH):
                        next_term, stop_loop = layer_loop(resp, current_upper_term)
                        if stop_loop:
                            extracted_terms.append(current_upper_term)
                            break
                        if next_term == current_upper_term or next_term.lower().strip().rstrip('.') == "none":
                            extracted_terms.append(current_upper_term)
                            break
                        current_upper_term = next_term

    print(f"\n\n{ch1_hist} \n\n{ch2_hist} \n\n{ch2_hist} \n\n{ch2_hist} \n\n")
    print("Extracted Terms : ", set(extracted_terms))
    sys.exit()

    extracted_terms = extract_terms(transcript=transcript, verbose=True)
