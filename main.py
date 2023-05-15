from langchain import LLMChain, PromptTemplate
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.vectorstores import Chroma, Pinecone
import numpy as np
import json
import os
import pandas as pd
import pinecone


template = """You are a medical ai assistant designed to help collect a patient's phenotype. The following excerpt {patient_text} is from a patient's recording with their physician. Using the following document of labeled HPO terms {hpo_doc}, try to find a term that matches the patient's symptoms. If none of the terms match the patient too well, simply write "none". If there is a match, write the term and a brief explanation for why the patient has that trait. Use only the information in the document to form your decision. """
patient_text = "the arms and legs are smaller around. I think it's hard to crawl for him"


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


if __name__ == "__main__":
    with open('hp.json') as f:
        data = json.load(f)

    hpo_data = data['graphs'][0]['nodes']
    hpo_data_df = pd.DataFrame(hpo_data)

    simple_hpo_df = pd.DataFrame(columns=['id', 'lbl', 'definition', 'comments'])

    simple_hpo_df['id'] = hpo_data_df['id'].apply(lambda x: str(x)[-10:])
    simple_hpo_df['lbl'] = hpo_data_df['lbl']
    simple_hpo_df['definition'] = hpo_data_df['meta'].apply(gather_definition)
    simple_hpo_df['comments'] = hpo_data_df['meta'].apply(gather_comments)

    simple_hpo_df['text_to_embed'] = simple_hpo_df.apply(str_to_embed, axis=1)

    #print(simple_hpo_df['text_to_embed'][98])

    embedding = OpenAIEmbeddings(openai_api_key=os.environ['OPENAI_API_KEY'])
    pinecone.init(
        api_key=os.environ['PINECONE_API_KEY'],
        environment=os.environ['PINECONE_API_ENV']
    )
    index_name = "hpo-embeddings"
    index = pinecone.Index(index_name)

    # only run once to embed hpo terms
    # embsearch = Pinecone.from_texts(simple_hpo_df['text_to_embed'].tolist(), embedding, index_name=index_name)

    vectorstore = Pinecone(index=index, embedding_function=embedding.embed_query, text_key='text')

    similar_terms = vectorstore.similarity_search(patient_text, k=10)

    concat_doc = ""
    for doc in similar_terms:
        concat_doc += doc.page_content + "\n\n"

    #print(concat_doc)

    llm = OpenAI(model_name='text-davinci-003')
    prompt = PromptTemplate(
        input_variables=["patient_text", "hpo_doc"],
        template=template,
    )

    chain = LLMChain(llm=llm, prompt=prompt)

    # Run the chain only specifying the input variable.
    print(chain.run({"patient_text": patient_text, "hpo_doc": concat_doc}))
