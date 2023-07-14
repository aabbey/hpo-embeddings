
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
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate
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

    with open('../hp.json') as f:
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

    with open('../hp.json', 'r') as f:
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


def terms_to_docs(term_list):
    under_docs = hpo_df.loc[hpo_df['lbl'].isin(term_list), 'text_to_embed'].values
    return "\n\n".join(under_docs)


def most_similar(bad_term, top_k=1):
    """
    use to retrieve a term in our hpo_df that is semantically similar to the bad_term
    """
    similar_term_doc = vector_store.similarity_search(bad_term, k=top_k)[0]
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
    under_terms = specify_term(current_upper_term).tolist()
    if not under_terms:
        return current_upper_term, True
    under_terms.append(current_upper_term)

    terms_to_search = hpo_df.index[hpo_df['lbl'].isin(under_terms)].tolist()
    similar_term_doc = vector_store.similarity_search(parent_text, k=1, filter={"df_index": {"$in": terms_to_search}})[0]
    similar_term = hpo_df.loc[hpo_df['text_to_embed'] == similar_term_doc.page_content, 'lbl'].values[0]
    print(similar_term)

    return similar_term, False



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

    with open('raw_text_inputs/sample_jessime_interview_notes.txt', 'r') as t:
        notes = t.read()
    with open("prompts/jes_note_sysmsg.txt", 'r') as p:
        sys_prompt1 = p.read()
    with open("prompts/jes_note_1.txt", 'r') as p:
        prompt1 = p.read()

    hpo_tree = make_hpo_tree()
    hpo_df = make_hpo_dataframe()

    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    list_of_extracted_terms = []

    # start the first llm with a sys message. the loop bellow will then add human msgs
    sys_msg_prompt = SystemMessage(content=sys_prompt1)
    ch1_hist = [sys_msg_prompt]
    hm_msg_prompt_template = HumanMessagePromptTemplate.from_template(prompt1)

    ch1_hist.append(hm_msg_prompt_template.format_messages(doc=notes)[0])

    response = llm(ch1_hist)
    print(response.content)

    extracted_terms = []
    for line in response.content.split("\n"):
        print(line)
        if "HP:" in line:
            terms_in_line = []
            id1 = line.find(": ")
            id2 = [m.start() for m in re.finditer("HP:", line)]
            idall = [id1] + id2
            for t in range(len(idall)-1):
                terms_in_line.append(line[idall[t] + len(line[idall[t]]): idall[t+1] - 1].strip())

            for term in terms_in_line:
                if term in hpo_df['lbl'].unique():
                    extracted_terms.append(term)
                else:
                    print("finding similar...")
                    print(term)
                    print(most_similar(term))
                    extracted_terms.append(most_similar(term))

    print("Extracted Terms : ", set(extracted_terms))
    sys.exit()

    extracted_terms = extract_terms(transcript=transcript, verbose=True)
