# Impot necessary libraries
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain import LLMChain
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

# Load environment variables
load_dotenv()

# create instance of chat model
chat = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# The system message acts as the initial instruction for the AI.
# It sets the context for the conversation.
# In our case, the AI is a helpful assistant that translates
# English to California surfer slang.
sys_msg_prompt = SystemMessagePromptTemplate.from_template(
    "You are helpful assistant that translates English text to California surfer slang"
)

# few shot prompting
example_human = HumanMessagePromptTemplate.from_template("Hi")
example_ai = AIMessagePromptTemplate.from_template("What's Up, dude?")

# We also specify a template for future human messages.
# In this case, it's just the text of the message.
human_msg_prompt = HumanMessagePromptTemplate.from_template("{text}")

# We then create a chat prompt from all these templates. The chat prompt is what we will use to guide the AI in the conversation.
chat_prompt = ChatPromptTemplate.from_messages(
    [sys_msg_prompt, example_human, example_ai, human_msg_prompt]
)

# Create LangChain instance with our chat model and chat prompt
chain = LLMChain(llm=chat, prompt=chat_prompt)

# print the text
print(chain.run("I love programming !"))
