import sys
import openai
from langchain import LLMChain, PromptTemplate
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Chroma, Pinecone
import numpy as np
import json
import os
import pandas as pd
import pinecone

#llm_model_name = "gpt-3.5-turbo"
llm_model_name = "text-davinci-003"

with open('prompts/hpo_extractor_template') as t:
    template = t.readlines()
    template = ''.join(template)


def extract_terms(conversation_path, verbose=True):
    with open(conversation_path) as t:
        conversation = t.readlines()
    if llm_model_name == "text-davinci-003":
        llm = OpenAI(model_name=llm_model_name)
    else:
        llm = ChatOpenAI(model_name=llm_model_name)
    embedding = OpenAIEmbeddings(openai_api_key=os.environ['OPENAI_API_KEY'])
    pinecone.init(
        api_key=os.environ['PINECONE_API_KEY'],
        environment=os.environ['PINECONE_API_ENV']
    )
    index_name = "hpo-embeddings"
    index = pinecone.Index(index_name)

    vectorstore = Pinecone(index=index, embedding_function=embedding.embed_query, text_key='text')

    list_of_extracted_terms = []
    for line in conversation:
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
    if verbose:
        print("\n----------------------------\n\n")
    set_of_extracted_terms = set(list_of_extracted_terms)
    if verbose:
        print("Extracted terms : ", set(list_of_extracted_terms))
    return set_of_extracted_terms


if __name__ == "__main__":
    extract_terms(conversation_path="sample generated transcripts/gpt3.5-sample2-angelman-syndrome")