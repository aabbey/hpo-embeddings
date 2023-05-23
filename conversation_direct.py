import json
import os
from random import sample
import sys
import pandas as pd
from langchain.memory import ChatMessageHistory
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain import LLMChain
from langchain.prompts import ChatPromptTemplate, PromptTemplate, SystemMessagePromptTemplate, AIMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.schema import AIMessage, HumanMessage, SystemMessage


full_example_traits = pd.read_csv('sample disease phenotypes/Angelman syndrome 1')

with open('prompts/full_conversation_prompt') as t:
    prompt_template = t.readlines()
    prompt_template = SystemMessagePromptTemplate.from_template(''.join(prompt_template))

if __name__ == '__main__':
    all_terms_list = full_example_traits['HPO_TERM_NAME'].tolist()
    traits_to_use = sample(all_terms_list, 20)
    print(traits_to_use)

    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.5)

    chat_hist = prompt_template.format_messages(traits=''.join(traits_to_use))

    print(llm(chat_hist).content)

