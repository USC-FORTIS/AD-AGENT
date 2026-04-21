import openai

def query_openai(messages, model="gpt-4o"):
    client = openai.OpenAI()
    response = client.chat.completions.create(
        model=model,
        messages=messages
    )
    return response.choices[0].message.content