import os
from typing import Any
import openai
from keras.models import load_model
from PIL import Image
import numpy as np
import cv2 as cv

class GPT4Api(object):
    def __init__(self) -> None:
        self.input = None
        self.api_key = "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"
        self.organization = "org-a4iUD3goyQ9UWdLR0KHZ11QL"

    #read a given file, return everything in the file as a string
    def read_file(self,file_name):
        file = open(file_name, "r")
        return file.read()
    
    def constraint(self, image_file):
        classes = { 1:'Speed limit (20km/h)',
            2:'Speed limit (30km/h)', 
            3:'Speed limit (50km/h)', 
            4:'Speed limit (60km/h)', 
            5:'Speed limit (70km/h)', 
            6:'Speed limit (80km/h)', 
            7:'End of speed limit (80km/h)', 
            8:'Speed limit (100km/h)', 
            9:'Speed limit (120km/h)', 
            10:'No passing', 
            11:'No passing veh over 3.5 tons', 
            12:'Right-of-way at intersection', 
            13:'Priority road', 
            14:'Yield', 
            15:'Stop', 
            16:'No vehicles', 
            17:'Veh > 3.5 tons prohibited', 
            18:'No entry', 
            19:'General caution', 
            20:'Dangerous curve left', 
            21:'Dangerous curve right', 
            22:'Double curve', 
            23:'Bumpy road', 
            24:'Slippery road', 
            25:'Road narrows on the right', 
            26:'Road work', 
            27:'Traffic signals', 
            28:'Pedestrians', 
            29:'Children crossing', 
            30:'Bicycles crossing', 
            31:'Beware of ice/snow',
            32:'Wild animals crossing', 
            33:'End speed + passing limits', 
            34:'Turn right ahead', 
            35:'Turn left ahead', 
            36:'Ahead only', 
            37:'Go straight or right', 
            38:'Go straight or left', 
            39:'Keep right', 
            40:'Keep left', 
            41:'Roundabout mandatory', 
            42:'End of no passing', 
            43:'End no passing veh > 3.5 tons' }
        model = load_model("my_model.h5")

        #use model to predict image: /Users/kevin/Desktop/cs1470/RoadMaster_GPT/prompt/data/images/0.jpg
        image = Image.open(image_file)
        image = image.resize((30,30))
        image = np.expand_dims(image, axis=0)
        image = np.array(image)
        pred = model.predict(image)
        label = np.argmax(pred[0])
        print(classes[label+1])
        return classes[label+1]


    def __call__(self, prompt, image_file_name, current_motion) -> Any:
        self.input = prompt
        openai.organization = self.organization
        openai.api_key = self.api_key
        completion = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "user", "content": prompt},
            {"role": "system", "content": ' Your car should not violate the traffic sign' + self.constraint(image_file_name) 
             + self.read_file("/Users/kevin/Desktop/cs1470/RoadMaster_GPT/api/control.txt")+ 'based on current motion'+ current_motion}
        ]
        )
        return completion["choices"][0]["message"]["content"]

class SelectionModel(object):
    def __init__(self) -> None:
        self.input = None
        self.api_key = "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"
        self.organization = "org-a4iUD3goyQ9UWdLR0KHZ11QL"
        openai.organization = self.organization
        openai.api_key = self.api_key

    #read a given file, return everything in the file as a string
    def read_file(self,file_name):
        file = open(file_name, "r")
        return file.read()
    
    def constraint(self, image_file):
        classes = { 1:'Speed limit (20km/h)',
            2:'Speed limit (30km/h)', 
            3:'Speed limit (50km/h)', 
            4:'Speed limit (60km/h)', 
            5:'Speed limit (70km/h)', 
            6:'Speed limit (80km/h)', 
            7:'End of speed limit (80km/h)', 
            8:'Speed limit (100km/h)', 
            9:'Speed limit (120km/h)', 
            10:'No passing', 
            11:'No passing veh over 3.5 tons', 
            12:'Right-of-way at intersection', 
            13:'Priority road', 
            14:'Yield', 
            15:'Stop', 
            16:'No vehicles', 
            17:'Veh > 3.5 tons prohibited', 
            18:'No entry', 
            19:'General caution', 
            20:'Dangerous curve left', 
            21:'Dangerous curve right', 
            22:'Double curve', 
            23:'Bumpy road', 
            24:'Slippery road', 
            25:'Road narrows on the right', 
            26:'Road work', 
            27:'Traffic signals', 
            28:'Pedestrians', 
            29:'Children crossing', 
            30:'Bicycles crossing', 
            31:'Beware of ice/snow',
            32:'Wild animals crossing', 
            33:'End speed + passing limits', 
            34:'Turn right ahead', 
            35:'Turn left ahead', 
            36:'Ahead only', 
            37:'Go straight or right', 
            38:'Go straight or left', 
            39:'Keep right', 
            40:'Keep left', 
            41:'Roundabout mandatory', 
            42:'End of no passing', 
            43:'End no passing veh > 3.5 tons' }
        model = load_model("my_model.h5")

        #use model to predict image: /Users/kevin/Desktop/cs1470/RoadMaster_GPT/prompt/data/images/0.jpg
        image = Image.open(image_file)
        image = image.resize((30,30))
        image = np.expand_dims(image, axis=0)
        image = np.array(image)
        pred = model.predict(image)
        label = np.argmax(pred[0])
        return classes[label+1]

    
    def __call__(self, prompt, image_file_name, current_motion) -> Any:
        output = []
        modules = self.module(prompt)
        print(modules)
        modules = modules.split(",")
        for module in modules:
            module_name = module.split(":")[0]
            module_prompt = module.split(":")[1]
            controller = self.controller(module_name, module_prompt, image_file_name, current_motion)
            output.append(controller)
        return output
    
    def module(self, prompt) -> Any:
        self.input = prompt
        completion = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": self.read_file('/Users/kevin/Desktop/cs1470/RoadMaster_GPT/api/module.txt')},
            {"role": "user", "content": prompt},
        ]
        )
        return completion["choices"][0]["message"]["content"]
    
    
    def controller(self, module_name, prompt, image_file_name, current_motion) -> Any:
        self.input = prompt
        completion = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": 'Do not do not return any other information other than the commands in return format '  + self.read_file('/Users/kevin/Desktop/cs1470/RoadMaster_GPT/api/controller.txt') + 'module_name' + module_name + ' '
             + 'traffice signs ' + self.constraint(image_file_name) + 'current motion'+ current_motion},
            {"role": "user", "content": prompt},
        ]
        )
        return completion["choices"][0]["message"]["content"]

def main():
    api = GPT4Api()
    ada = SelectionModel()
    print(ada("turn right, and turn left while beeping", "/Users/kevin/Desktop/cs1470/RoadMaster_GPT/sign/archive/Meta/99 8.05.23 PM.jpg", "current motion v_x=0,v_y=0,v_z=0"))

def langchain():
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

if __name__ == '__main__':
    main()



