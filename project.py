#!/usr/bin/env python
# coding: utf-8

# # Custom Chatbot Project

# I chose the character_descriptions dataset. With a context provided, I can ask the AI specific questions about characters, using their names for example, whereas without context, the AI would have no idea which characters I'm referring to.

# ## Data Wrangling
# 
# TODO: In the cells below, load your chosen dataset into a `pandas` dataframe with a column named `"text"`. This column should contain all of your text data, separated into at least 20 rows.

# In[3]:


import pandas as pd
import numpy as np


# In[4]:


df = pd.read_csv("data/character_descriptions.csv")
df.head()


# In[5]:


# Initialize an empty list to store the concatenated strings
concatenated_texts = []

# Loop through each row in the DataFrame
for index, row in df.iterrows():
    concatenated_text = f"{row['Name']}: {row['Description']}, {row['Medium']}, {row['Setting']}"
    concatenated_texts.append(concatenated_text)

# Create a new DataFrame with the concatenated texts
df = pd.DataFrame(concatenated_texts, columns=['text'])
df


# ## Custom Query Completion
# 
# TODO: In the cells below, compose a custom query using your chosen dataset and retrieve results from an OpenAI `Completion` model. You may copy and paste any useful code from the course materials.

# In[6]:


import openai
openai.api_base = "https://openai.vocareum.com/v1"
openai.api_key = # SECRET


# In[7]:


# Using Code from 4.24 Case Study Workspace
# Generate and save embeddings

EMBEDDING_MODEL_NAME = "text-embedding-ada-002"
batch_size = 100
embeddings = []
for i in range(0, len(df), batch_size):
    # Send text data to OpenAI model to get embeddings
    response = openai.Embedding.create(
        input=df.iloc[i:i+batch_size]["text"].tolist(),
        engine=EMBEDDING_MODEL_NAME
    )
    
    # Add embeddings to list
    embeddings.extend([data["embedding"] for data in response["data"]])

# Add embeddings list to dataframe
df["embeddings"] = embeddings
df.to_csv("embeddings.csv")
df


# In[8]:


# Using Code from 4.24 Case Study Workspace
# Load embeddings from file
#df = pd.read_csv("embeddings.csv", index_col=0)
#df["embeddings"] = df["embeddings"].apply(eval).apply(np.array)


# In[9]:


# Using Code from 4.24 Case Study Workspace

from openai.embeddings_utils import get_embedding, distances_from_embeddings

def get_rows_sorted_by_relevance(question, df):
    """
    Function that takes in a question string and a dataframe containing
    rows of text and associated embeddings, and returns that dataframe
    sorted from least to most relevant for that question
    """
    
    # Get embeddings for the question text
    question_embeddings = get_embedding(question, engine=EMBEDDING_MODEL_NAME)
    
    # Make a copy of the dataframe and add a "distances" column containing
    # the cosine distances between each row's embeddings and the
    # embeddings of the question
    df_copy = df.copy()
    df_copy["distances"] = distances_from_embeddings(
        question_embeddings,
        df_copy["embeddings"].values,
        distance_metric="cosine"
    )
    
    # Sort the copied dataframe by the distances and return it
    # (shorter distance = more relevant so we sort in ascending order)
    df_copy.sort_values("distances", ascending=True, inplace=True)
    return df_copy


# In[10]:


get_rows_sorted_by_relevance("retired soldier", df)


# In[11]:


# Using Code from 4.24 Case Study Workspace
# Create prompt, pay attention to max tokens

import tiktoken

def create_prompt(question, df, max_token_count):
    """
    Given a question and a dataframe containing rows of text and their
    embeddings, return a text prompt to send to a Completion model
    """
    # Create a tokenizer that is designed to align with our embeddings
    tokenizer = tiktoken.get_encoding("cl100k_base")
    
    # Count the number of tokens in the prompt template and question
    prompt_template = """
Answer the question based on the context below, and if the question
can't be answered based on the context, say "I don't know"

Context: 

{}

---

Question: {}
Answer:"""
    
    current_token_count = len(tokenizer.encode(prompt_template)) + \
                            len(tokenizer.encode(question))
    
    context = []
    for text in get_rows_sorted_by_relevance(question, df)["text"].values:
        
        # Increase the counter based on the number of tokens in this row
        text_token_count = len(tokenizer.encode(text))
        current_token_count += text_token_count
        
        # Add the row of text to the list if we haven't exceeded the max
        if current_token_count <= max_token_count:
            context.append(text)
        else:
            break

    return prompt_template.format("\n\n###\n\n".join(context), question)


# In[12]:


print(create_prompt("Who is currently in a relationship?", df, 200))


# In[13]:


# Using Code from 4.24 Case Study Workspace
# Answer a question

COMPLETION_MODEL_NAME = "gpt-3.5-turbo-instruct"

def answer_question(question, max_answer_tokens=150):
    response = openai.Completion.create(
        model=COMPLETION_MODEL_NAME,
        prompt=question,
        max_tokens=max_answer_tokens
    )
    return response["choices"][0]["text"].strip()

def answer_question_with_context(
    question, df, max_prompt_tokens=1800, max_answer_tokens=150
):
    """
    Given a question, a dataframe containing rows of text, and a maximum
    number of desired tokens in the prompt and response, return the
    answer to the question according to an OpenAI Completion model
    
    If the model produces an error, return an empty string
    """
    
    prompt = create_prompt(question, df, max_prompt_tokens)
    
    try:
        response = openai.Completion.create(
            model=COMPLETION_MODEL_NAME,
            prompt=prompt,
            max_tokens=max_answer_tokens
        )
        return response["choices"][0]["text"].strip()
    except Exception as e:
        print(e)
        return ""


# ## Custom Performance Demonstration
# 
# TODO: In the cells below, demonstrate the performance of your custom query using at least 2 questions. For each question, show the answer from a basic `Completion` model query as well as the answer from your custom query.

# ### Question 1

# In[14]:


question1 = "Which characters are above 30?"


# In[15]:


answer1 = answer_question(question1)
print(answer1)


# In[16]:


custom_answer1 = answer_question_with_context(question1, df)
print(custom_answer1)


# ### Question 2

# In[17]:


question2 = "Who is currently in a relationship?"


# In[18]:


answer2 = answer_question(question2)
print(answer2)


# In[19]:


custom_answer2 = answer_question_with_context(question2, df)
print(custom_answer2)


# ### Question 3

# In[20]:


question3 = "Is Rachel in a relationship?"


# In[21]:


answer3 = answer_question(question3)
print(answer3)


# In[22]:


custom_answer3 = answer_question_with_context(question3, df)
print(custom_answer3)


# ### Question 4

# In[26]:


question4 = "Which of the characters play an evil role?"


# In[27]:


answer4 = answer_question(question4)
print(answer4)


# In[28]:


custom_answer4 = answer_question_with_context(question4, df)
print(custom_answer4)


# In[ ]:




