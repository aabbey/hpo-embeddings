from langchain.chat_models import ChatOpenAI

from hpo_extract import hpo_to_follow_up




def run_hpo_to_follow_up(input_dict):    
    deepen_llm = ChatOpenAI(model=hpo_to_follow_up.MODEL_NAME, temperature=0.0)
    deepen_llm_prompts = hpo_to_follow_up.load_prompts()
    deepen_chain_output = hpo_to_follow_up.run_deepen_chain(input_dict=input_dict,
                                                            llm=deepen_llm,
                                                            prompts=deepen_llm_prompts,
                                                            show_work=True)
    
    
        
    
    return deepen_chain_output


if __name__ == '__main__':
    hpo_to_follow_up_input = hpo_to_follow_up.load_input()
    
    deepen_chain_output = run_hpo_to_follow_up(hpo_to_follow_up_input)
    
    for generation in deepen_chain_output.generations:
        print(generation[0].text)

    print(deepen_chain_output.llm_output)
