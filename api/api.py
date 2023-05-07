import os
from typing import Any
import openai

class Api(object):
    def __init__(self) -> None:
        self.input = None
        self.api_key = "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"
        self.organization = "org-a4iUD3goyQ9UWdLR0KHZ11QL"

    #read a given file, return everything in the file as a string
    def read_file(self,file_name):
        file = open(file_name, "r")
        return file.read()
    
    def __call__(self, input) -> Any:
        self.input = input
        openai.organization = self.organization
        openai.api_key = self.api_key
        completion = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "user", "content": input},
            {"role": "system", "content": self.read_file("/Users/kevin/Desktop/cs1470/final_project/api/control.txt")}
        ]
        )
        return completion["choices"][0]["message"]["content"]


def main():
    api = Api()
    print(api("Give me a dance move like letter 7."))

if __name__ == '__main__':
    main()

from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.llms import OpenAI
import os
os.environ["OPENAI_API_KEY"] = "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"
os.environ["SERPAPI_API_KEY"] = "70648d04190456c757bc729131b02e870027055d4c88c54b88cd04182913a5e7"

llm = OpenAI(temperature=0)
tools = load_tools(["serpapi", "llm-math"], llm=llm)
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
text = "who sing the song 'Hello', is it adele?"
print(agent.run(text))

