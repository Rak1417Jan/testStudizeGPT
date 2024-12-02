
from fastapi import FastAPI
import openai
import os

# Get your OpenAI API key from environment variables
apikey = os.getenv('API_KEY')

# Initialize FastAPI app
app = FastAPI()

# Configure OpenAI with the API key
openai.api_key = apikey

# Set generation parameters
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "max_tokens": 8192,  # Adjust as per your needs, but note that GPT-4 has limits
}

# Define the endpoint
@app.post("/chat")
async def chat(query: str):
    # Prompt message for the assistant
    promptMessage = """You are an expert Joint Entrance Examination AI guide. You are very enthusiastic and always solve students' doubts and queries in a very stepwise manner. Also, you always make sure that students have the enthusiasm still inside them and don't lose their interest in the science field. So, you also give them real-life facts if any. Additionally, you provide formulas with their explanations to students if any. MAKE SURE YOU DON'T ANSWER ANY QUERY OUT OF THE TOPICS ASKED FROM JOINT ENTRANCE EXAMINATION.
                Make sure to greet the user with the name first in BOLD
Give 3 very good advanced level questions based on the query at the end.
There is a task page on the app, ask the user to set a task to revise the topic of which he has asked the query.
Make sure you don't give any special characters just keep the outputs as plain text.
If you are not directly asked about a specific JEE question, don't give those 3 extra questions. Only give 3 extra questions when the user asks a JEE-specific question.
"""
    
    # Combine the prompt and the user's query
    full_prompt = promptMessage + query

    # Call the OpenAI API with the prompt
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",  # You can use "gpt-4" or "gpt-3.5-turbo" for a smaller model
        messages=[{"role": "system", "content": promptMessage},
                  {"role": "user", "content": query}],
        temperature=generation_config["temperature"],
        top_p=generation_config["top_p"],
        max_tokens=generation_config["max_tokens"]
    )

    # Extract the text from the response
    return {"response": response.choices[0].message["content"]}
