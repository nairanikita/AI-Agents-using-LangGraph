from dotenv import load_dotenv
import google.generativeai as genai
import os
import re

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
# model=genai.GenerativeModel(model_name="gemini-1.5-flash", # Or the model you prefer
#     system_instruction="You are a helpful assistant. Always respond in a friendly and informative tone, avoiding overly technical jargon. Ensure your answers are complete and well-structured, using bullet points for lists and bolding key terms. If a question is unclear, politely ask for clarification."
# )
# chat=model.start_chat(history=[])

# def get_gemini_response(question):
#     response=chat.send_message(question)
#     return response
# response=get_gemini_response("What is the capital of France?")
# print(response.text)
# print(chat.history)
# response=get_gemini_response("What is the capital of India?")
# print(response.text)
# print(chat.history)

class Agent:
    def __init__(self, system=""):
        self.system_instruction={system}
        model=genai.GenerativeModel(model_name="gemini-1.5-flash", 
        system_instruction=self.system_instruction)
        self.chat = model.start_chat(history=[])
        

    def __call__(self, message):
        response = self.chat.send_message(message)
        # print(self.chat.history)
        return response.text
    
       
prompt = """
You run in a loop of Thought, Action, PAUSE, Observation.
At the end of the loop you output an Answer
Use Thought to describe your thoughts about the question you have been asked.
Use Action to run one of the actions available to you - then return PAUSE.
Observation will be the result of running those actions.

Your available actions are:

calculate:
e.g. calculate: 4 * 7 / 3
Runs a calculation and returns the number - uses Python so be sure to use floating point syntax if necessary

average_dog_weight:
e.g. average_dog_weight: Collie
returns average weight of a dog when given the breed

Example session:

Question: How much does a Bulldog weigh?
Thought: I should look the dogs weight using average_dog_weight
Action: average_dog_weight: Bulldog
PAUSE

You will be called again with this:

Observation: A Bulldog weights 51 lbs

You then output:

Answer: A bulldog weights 51 lbs
""".strip()

def calculate(what):
    return eval(what)

def average_dog_weight(name):
    if name in "Scottish Terrier": 
        return("Scottish Terriers average 20 lbs")
    elif name in "Border Collie":
        return("a Border Collies average weight is 37 lbs")
    elif name in "Toy Poodle":
        return("a toy poodles average weight is 7 lbs")
    else:
        return("An average dog weights 50 lbs")

known_actions = {
    "calculate": calculate,
    "average_dog_weight": average_dog_weight
}
abot = Agent(prompt)
result = abot("What is the average weight of toy poodle and Scottish Terrier?")
print(result)
while(result.strip()[-5:]=="PAUSE"):
    

    fetch_action=result[result.find("Action:")+7:].split("\n")[0].strip()
    action_name,action_args=fetch_action.split(":")
    action_name=action_name.strip()
    action_args=action_args.strip()
    print(f"Action:{action_name},Args:{action_args}")
    if action_name in known_actions:
        result=known_actions[action_name](action_args)
    else:
        result="I don't know how to do that"
    next_prompt="Observation:{}".format(result)
    result=abot(next_prompt)
    print(result)

# print(result)
# result = average_dog_weight("Toy Poodle")
# next_prompt = "Observation: {}".format(result)
# result=abot(next_prompt)
# print(result)
# result = average_dog_weight("Scottish Terrier")
# next_prompt = "Observation: {}".format(result)
# result=abot(next_prompt)
# print(result)
# result=calculate("7+20")
# next_prompt = "Observation: {}".format(result)
# result=abot(next_prompt)

# print(result)  # Should print the final answer
# print(chat.history)