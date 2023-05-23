import json
import os
import sys
from langchain.memory import ChatMessageHistory
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain import LLMChain
from langchain.prompts import ChatPromptTemplate, PromptTemplate, SystemMessagePromptTemplate, AIMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.schema import AIMessage, HumanMessage, SystemMessage

KEY = os.environ["OPENAI_API_KEY"]
qa_sys_msg_text = """You are a physician working with a the parent of a patient with a rare disease. The patient is a young child or infant. Your job is to ask the parent questions that will lead to discovering phenotypic traits about the patient (child). Make sure to use language that is conversational and easy to understand. Don't ask too many questions at once, and keep your response under 40 words."""
p_sys_msg_text = """You are the parent of a young child with a rare disease. You are talking with a physician about your child's traits and symptoms. The physician will ask questions regarding your child, and your job is to answer these questions based on your child's traits in a concise, conversational, and helpful manner. You should come up with a name for yourself and your child. You do not understand technical medical terminology, so describe things as best you can in simple layman's terms. Keep your response under 40 words. Some of the traits your child has are listed below. 
Traits: {traits}"""
trait_gen_prompt_template = """What are some physical and behavioral traits of a young child or infant with the disease {disease}? Respond only with traits listed one at a time with commas separating them. Do not add any extra unnecessary words or filler."""
sample_disease = "microdeletion syndrome"
conversation_length = 8

if __name__ == '__main__':
    #trait_gen_llm = OpenAI(model_name="text-davinci-003", temperature=0.5)
    #trait_gen_prompt = PromptTemplate(input_variables=['disease'], template=trait_gen_prompt_template)
    #trait_gen_chain = LLMChain(llm=trait_gen_llm, prompt=trait_gen_prompt)
    #generated_traits = trait_gen_chain.run(sample_disease)

    #print(generated_traits, '\n')

    generated_traits = "Delayed growth, delayed development, speech delays, communication difficulties, behavioral issues, low muscle tone, poor coordination, vision problems, hearing problems, seizures, heart defects. "

    qa_llm = ChatOpenAI(temperature=0.2)
    p_llm = ChatOpenAI(temperature=0.5)
    qa_system_msg = SystemMessage(content=qa_sys_msg_text)
    p_system_msg = SystemMessagePromptTemplate.from_template(p_sys_msg_text)
    qa_chat_hist = [qa_system_msg]
    p_chat_hist = p_system_msg.format_messages(traits=generated_traits)

    first_qa_response = qa_llm(qa_chat_hist)
    print("qa says : ", first_qa_response.content, '\n\n')
    qa_chat_hist.append(first_qa_response)

    p_chat_hist.append(HumanMessage(content=first_qa_response.content))
    first_p_response = p_llm(p_chat_hist)
    print("p says : ", first_p_response.content, '\n\n')
    p_chat_hist.append(first_p_response)
    qa_chat_hist.append(HumanMessage(content=first_p_response.content))

    for chat_number in range(conversation_length):
        # qa conv
        qa_response = qa_llm(qa_chat_hist)  # generate doctor's response from his perspective chat history
        qa_chat_hist.append(qa_response)
        print("qa says : ", qa_response.content, '\n\n')

        # p conv
        p_chat_hist.append(HumanMessage(content=qa_response.content))  # qa_response is the human from p's perspective
        p_response = p_llm(p_chat_hist)  # generate patient's parent's response from his perspective chat history
        p_chat_hist.append(p_response)
        print("p says : ", p_response.content, '\n\n')

        qa_chat_hist.append(HumanMessage(content=p_response.content))  # p_response is the human from qa's perspective

    print("\n------------------\n\n")
    print("qa chat history : \n", qa_chat_hist, '\n\n')
    print("p chat history : \n", p_chat_hist, '\n\n')
