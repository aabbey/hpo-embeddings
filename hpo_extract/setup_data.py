import os
import json

import pandas as pd
from langchain.vectorstores import Pinecone
import pinecone
from langchain.embeddings import OpenAIEmbeddings


def make_hpo_dataframe():
    def gather_definition(meta_dict):
        if isinstance(meta_dict, dict):
            if "definition" in meta_dict.keys():
                return meta_dict["definition"]["val"]
            else:
                return ""
        else:
            return ""

    def gather_comments(meta_dict):
        if isinstance(meta_dict, dict):
            if "comments" in meta_dict.keys():
                return meta_dict["comments"]
            else:
                return ""
        else:
            return ""

    def str_to_embed(df_row):
        string_template = ""
        if pd.notnull(df_row["lbl"]):
            string_template += f"label: {df_row['lbl']} \n"
        if pd.notnull(df_row["definition"]):
            string_template += f"definition: {df_row['definition']} \n"
        if pd.notnull(df_row["comments"]):
            string_template += "comments: "
            for comment in df_row["comments"]:
                string_template += f"{comment} \n"
        return string_template

    with open("data/hp.json") as f:
        data = json.load(f)

    hpo_data = data["graphs"][0]["nodes"]

    hpo_data_df = pd.DataFrame(hpo_data)

    simple_hpo_df = pd.DataFrame(columns=["id", "lbl", "definition", "comments"])

    simple_hpo_df["id"] = hpo_data_df["id"]
    simple_hpo_df["type"] = hpo_data_df["id"].apply(lambda x: str(x)[-11:-8])
    simple_hpo_df["lbl"] = hpo_data_df["lbl"]
    simple_hpo_df["definition"] = hpo_data_df["meta"].apply(gather_definition)
    simple_hpo_df["comments"] = hpo_data_df["meta"].apply(gather_comments)

    simple_hpo_df["text_to_embed"] = simple_hpo_df.apply(str_to_embed, axis=1)

    return simple_hpo_df


def make_hpo_dag():
    class DAG:
        def __init__(self):
            self.graph = {}
            self.parents = {}

        def add_edge(self, sub, obj):
            # Add sub as a child of obj
            if obj in self.graph:
                self.graph[obj].add(sub)
            else:
                self.graph[obj] = set([sub])

            # Add obj as a parent of sub
            if sub in self.parents:
                self.parents[sub].add(obj)
            else:
                self.parents[sub] = set([obj])

        def get_children(self, id):
            return self.graph.get(id, set())

        def get_parents(self, id):
            return self.parents.get(id, set())

        def get_ancestors(self, node_id):
            """Returns the upper set (all ancestors) of a node."""
            ancestors = set()
            direct_parents = self.get_parents(node_id)
            ancestors.update(direct_parents)
            for parent in direct_parents:
                ancestors.update(self.get_ancestors(parent))
            return ancestors

    def create_dag_from_list(list_of_dicts):
        dag = DAG()
        for dic in list_of_dicts:
            dag.add_edge(dic["sub"], dic["obj"])
        return dag

    with open("data/hp.json", "r") as f:
        data = json.load(f)

    edges_list = data["graphs"][0]["edges"]
    tree_of_ids = create_dag_from_list(edges_list)
    return tree_of_ids


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

    with open("data/hp.json", "r") as f:
        data = json.load(f)

    edges_list = data["graphs"][0]["edges"]
    tree_of_ids = create_tree_from_list(edges_list)
    return tree_of_ids


def create_hpo_vector_store():
    embedding = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])
    pinecone.init(
        api_key=os.environ["PINECONE_API_KEY"],
        environment=os.environ["PINECONE_API_ENV"],
    )
    index_name = "hpo-term-embeddings"
    index = pinecone.Index(index_name)
    vector_store = Pinecone(index, embedding, "text")
    return vector_store


def get_terms_vectors():
    with open("/Users/alex.abbey/Projects/hpo-extraction/term_ids_vecs.json", "r") as f:
        term_ids_vecs = json.load(f)
    return term_ids_vecs


HPO_VECTORS = create_hpo_vector_store()
HPO_TREE = make_hpo_tree()
HPO_DAG = make_hpo_dag()
HPO_DF = make_hpo_dataframe()
TERM_IDS_VECS = get_terms_vectors()
