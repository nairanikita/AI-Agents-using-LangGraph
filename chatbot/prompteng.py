# import openai
from google import genai
from google.genai import types
from dotenv import load_dotenv,find_dotenv
import os
load_dotenv()
client=genai.Client()
# chatgpt_api_ket=os.getenv("CHATGPT_API_KEY")
# text = f"""
# You should express what you want a model to do by  
# providing instructions that are as clear and  
# specific as you can possibly make them. 
# This will guide the model towards the desired output, 
# and reduce the chances of receiving irrelevant 
# or incorrect responses. Don't confuse writing a 
# clear prompt with writing a short prompt. 
# In many cases, longer prompts provide more clarity 
# and context for the model, which can lead to 
# more detailed and relevant outputs.
# """
text = f"""
Got this for my daughter for her birthday cuz she keeps taking \
mine from my room.  Yes, adults also like pandas too.  She takes \
it everywhere with her, and it's super soft and cute.  One of the \
ears is a bit lower than the other, and I don't think that was \
designed to be asymmetrical. It's a bit small for what I paid for it \
though. I think there might be other options that are bigger for \
the same price.  It arrived a day earlier than expected, so I got \
to play with it myself before I gave it to my daughter.
"""
prompt=f"proofread and correct this review: ```{text}```"
conditions=f"""
1. The movie should be released after 2000.
2. The genre should be either Action or Comedy."""
response=client.models.generate_content(
    model= "gemini-2.5-flash",
    contents=prompt,
    # contents=f"""Summarize the text delimited by triple backticks into a single sentence.```{text}```""",
    # contents=f"Generate a list of movies with their genres and release years in JSON Format.After that filter the movies based on the following conditions given in backticks```{conditions}```",
    # config=types.GenerateContentConfig(
    #     thinking_config=types.ThinkingConfig(budget=0)
    #     ),
     )

print(response.text)
# from redlines import Redlines
# from IPython.display import display, Markdown


# diff = Redlines(text,response.text)
# print(display(Markdown(diff.output_markdown)))
chat=client.start_chat()
print("🤖 Gemini Chatbot is ready! Type 'exit' to quit.\n")
while True:
    user_input=imput("You: ")
    if user_input.lower()=="exit":
        print("Goodbye")
        break
    try:
        response=chat.send_message(user_input)
        print(f"🤖 Gemini: {response.text}")
    except Exception as e:
        print(f"Error:{e}")