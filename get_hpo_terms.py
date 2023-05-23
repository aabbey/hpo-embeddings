import sys

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

llm_model_name = "gpt-3.5-turbo"
template = """You are a medical AI assistant designed to help collect a patient's phenotype. The patient is a child. The following excerpt {patient_text} is from a recording of the parent talking to the physician regarding their child. The parent will speak in plain language and will likely use terms that aren't medically accurate. Take this into account. Using the following document of labeled HPO terms {hpo_doc}, find terms that match the patient's (child's) symptoms. If none of the terms match the patient too well, simply write "none". If there is at least one match, write all of the terms and a brief explanation for why the patient (child) has those traits. Use only the information in the document to form your decision. Keep responses short, no more than 40 words."""
template1 = """You are a medical ai assistant designed to help collect a patient's phenotype. The following excerpt {patient_text} is from a patient's recording with their physician. Using the following document of labeled HPO terms {hpo_doc}, try to find a term that matches the patient's symptoms. If none of the terms match the patient too well, simply write "none". If there is at least one match, write all of the terms and a brief explanation for why the patient has those traits. Use only the information in the document to form your decision. """


with open('sample generated transcripts/gpt3.5-sample1') as t:
    conversation = t.readlines()
    #conversation = ''.join(conversation)

if __name__ == "__main__":
    llm = ChatOpenAI(model_name=llm_model_name)
    embedding = OpenAIEmbeddings(openai_api_key=os.environ['OPENAI_API_KEY'])
    pinecone.init(
        api_key=os.environ['PINECONE_API_KEY'],
        environment=os.environ['PINECONE_API_ENV']
    )
    index_name = "hpo-embeddings"
    index = pinecone.Index(index_name)

    vectorstore = Pinecone(index=index, embedding_function=embedding.embed_query, text_key='text')

    for line in conversation:
        if line.startswith("Parent:"):
            print("--------------------\n\n", line, '\n\n')
            similar_terms = vectorstore.similarity_search(line, k=10)

            concat_doc = ""
            for doc in similar_terms:
                concat_doc += doc.page_content + "\n\n"

            print(concat_doc)

            prompt = PromptTemplate(
                input_variables=["patient_text", "hpo_doc"],
                template=template,
            )

            chain = LLMChain(llm=llm, prompt=prompt)

            # Run the chain only specifying the input variable.
            print(chain.run({"patient_text": line, "hpo_doc": concat_doc}))
