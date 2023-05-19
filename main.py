from langchain import LLMChain, PromptTemplate
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.vectorstores import Chroma, Pinecone
import numpy as np
import json
import os
import pandas as pd
import pinecone

llm_model_name = "text-davinci-003"
template = """You are a medical AI assistant designed to help collect a patient's phenotype. The patient is a child. The following excerpt {patient_text} is from a recording of the parent talking to the physician regarding their child. The parent will speak in plain language and will likely use terms that aren't medically accurate. Take this into account. Using the following document of labeled HPO terms {hpo_doc}, find terms that match the patient's (child's) symptoms. If none of the terms match the patient too well, simply write "none". If there is at least one match, write all of the terms and a brief explanation for why the patient (child) has those traits. Use only the information in the document to form your decision. """
template1 = """You are a medical ai assistant designed to help collect a patient's phenotype. The following excerpt {patient_text} is from a patient's recording with their physician. Using the following document of labeled HPO terms {hpo_doc}, try to find a term that matches the patient's symptoms. If none of the terms match the patient too well, simply write "none". If there is at least one match, write all of the terms and a brief explanation for why the patient has those traits. Use only the information in the document to form your decision. """
patient_text3 = "He doesn't seem to understand what his peers are saying. He also seems to overreact to being touched, like a handshake or high five."
patient_text = "My child's head looks weird. It's too big."
patient_text4 = "My child's leg is sensitive to touch. His head also looks very large."
patient_text5 = "She walks on her tippy toes and her balance seems off, like she sometimes teeters and falls."
patient_text2 = "My 2-year-old daughter walks on the tips of her toes and never on flat feet."


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

    print(concat_doc)

    llm = OpenAI(model_name=llm_model_name)
    prompt = PromptTemplate(
        input_variables=["patient_text", "hpo_doc"],
        template=template,
    )

    chain = LLMChain(llm=llm, prompt=prompt)

    # Run the chain only specifying the input variable.
    print(chain.run({"patient_text": patient_text, "hpo_doc": concat_doc}))
