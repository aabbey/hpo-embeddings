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
MODEL_NAME = "gpt-4"


if __name__ == '__main__':

    llm = ChatOpenAI(model=MODEL_NAME, temperature=0.0)

    with open("generation/prompts/clin_note_followup/sysmsg.txt", "r") as f:
        sysmsg_file = f.read()

    with open("samples/sample_clinician_notes/gpt_gen_1.txt", "r") as f:
        clin_notes = f.read()

    with open("generation/prompts/clin_note_followup/prompt_1_template.txt", "r") as f:
        p1_template = f.read()

    system_msg = SystemMessage(content=sysmsg_file)
    llm_chat_hist = [system_msg]

    prompt_1_template = HumanMessagePromptTemplate.from_template(p1_template)

    llm_chat_hist.append(prompt_1_template.format_messages(clin_notes=clin_notes)[0])

    print(llm_chat_hist)
    print("\n\n")


    first_p_response = llm(llm_chat_hist)
    print(first_p_response.content)

    # next step : should be more specific about hpo terms.

    sys.exit()

