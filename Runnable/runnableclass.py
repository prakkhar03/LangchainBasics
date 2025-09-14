from abc import ABC ,abstractmethod
 

import random




class chainn:
    def __init__(self,llm,prompt):
        self.llm=llm
        self.prompt=prompt
    def run(self,input_dict):
        final_prompt=self.promt_template.Format(input_dict)
        result=self.modelllm.predict(final_prompt)

class Runnable:
    @abstractmethod
    def invoke(input_data):
        pass

class modelllm(Runnable):
    def __init__(self):
        print("LLM Created")
    def invoke(self,prompt):
        response_list=[
            "Delhi is the capital of India",
            "AI stands for artificial intellegence"
            "Ipl is a cricket league"
        ]
    def predict(self,prompt):
        response_list=[
            "Delhi is the capital of India",
            "AI stands for artificial intellegence"
            "Ipl is a cricket league"
        ]
        return {'response':random.choice(response_list),'message':'Can be removed in future use invoke'}
llm=modelllm()
print(llm.predict("hii"))
 
class prompt_template(Runnable):
    def __init__(self,template,input_variables):
        self.template=template
        self.input_variables=input_variables
    def invoke(self,input_dict):
        return self.template.format(**input_dict)
    def Format(self,input_dict):
        return self.template.format(**input_dict)
template=prompt_template(
    template="Write a poem on {topic}",
    input_variables=['topic']
)
print(template.Format({'topic':'india'}))
