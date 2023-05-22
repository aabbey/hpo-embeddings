import json
import os
import sys
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

trait_gen_prompt_template = """What are some physical and behavioral traits of a young child or infant with the disease {disease}? Respond only with traits listed one at a time with commas separating them. Do not add any extra unnecessary words or filler."""
sample_disease = "microdeletion syndrome"

with open('qa_prompt') as t:
    qa_prompt = t.readlines()[0]

if __name__ == '__main__':
    trait_gen_llm = openai(model_name="text-davinci-003", temperature=0.5)
    trait_gen_prompt = trait_gen_prompt_template.format({"disease": sample_disease})
    generated_traits = trait_gen_llm(trait_gen_prompt)

    print(generated_traits, '\n')

    qa_llm = openai(model_name="gpt-3.5-turbo", temperature=0.5)
    first_qa_response = qa_llm(qa_prompt)
    print(first_qa_response, '\n')
    sys.exit()
    p_llm = ChatOpenAI(temperature=0.5)
    qa_system_msg = SystemMessage(content=qa_sys_msg_text)
    p_system_msg = SystemMessagePromptTemplate.from_template(p_sys_msg_text)
    qa_chat_hist = [qa_system_msg]
    p_chat_hist = p_system_msg.format_messages(traits=generated_traits)

    first_qa_response = qa_llm(qa_chat_hist)
    print("qa says : ", first_qa_response.content, '\n\n')
    sys.exit()

    qa_chat_hist.append(first_qa_response)
    p_chat_hist.append(HumanMessage(content=first_qa_response.content))
    first_p_response = p_llm(p_chat_hist)
    print("p says : ", first_p_response.content, '\n\n')
    p_chat_hist.append(first_p_response)

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
