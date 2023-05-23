import json
import os
import sys
from langchain.memory import ChatMessageHistory
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain import LLMChain
from langchain.prompts import ChatPromptTemplate, PromptTemplate, SystemMessagePromptTemplate, AIMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.schema import AIMessage, HumanMessage, SystemMessage

trait_gen_prompt_template = """What are some physical and behavioral traits of a young child or infant with the disease {disease}? Respond only with traits listed one at a time with commas separating them. Do not add any extra unnecessary words or filler."""
sample_disease = "microdeletion syndrome"

with open('full_conversation_prompt') as t:
    prompt_template = t.readlines()
    prompt_template = SystemMessagePromptTemplate.from_template(''.join(prompt_template))

if __name__ == '__main__':
    trait_gen_llm = OpenAI(model_name="text-davinci-003", temperature=0.5)
    trait_gen_prompt = PromptTemplate(input_variables=['disease'], template=trait_gen_prompt_template)
    trait_gen_chain = LLMChain(llm=trait_gen_llm, prompt=trait_gen_prompt)
    generated_traits = trait_gen_chain.run(sample_disease)

    print(generated_traits, '\n')

    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.5)

    chat_hist = prompt_template.format_messages(traits=generated_traits)

    print(llm(chat_hist).content)

