

import streamlit as st
st.set_option('deprecation.showfileUploaderEncoding', False)
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from PIL import Image
from huggingface_hub import hf_hub_download
from llama_cpp import Llama
import torch
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from together import Together

@st.cache(allow_output_mutation=True)  # Cache models
def load_models():
    # Model initialization
    processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
    model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf", torch_dtype=torch.float16) 
    model.to("cuda:0")

    model_name = "aaditya/OpenBioLLM-Llama3-8B-GGUF"
    model_file = "openbiollm-llama3-8b.Q5_K_M.gguf"
    model_path = hf_hub_download(model_name, filename=model_file, local_dir='/content')
    llm = Llama(model_path=model_path.lstrip(), n_gpu_layers=-1)
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    vectordb_diabetes = Chroma(persist_directory="/content/drive/MyDrive/266/diabetes/data", embedding_function=embeddings)
    vectordb_asthma = Chroma(persist_directory="/content/drive/MyDrive/266/asthma/data", embedding_function=embeddings)
    vectordb_dialysis = Chroma(persist_directory="/content/drive/MyDrive/266/dialysis/data", embedding_function=embeddings)

    return processor, model, llm, embeddings, vectordb_diabetes, vectordb_asthma, vectordb_dialysis

# Load models
processor, model, llm, embeddings, vectordb_diabetes, vectordb_asthma, vectordb_dialysis = load_models()

def getcontext_diabetes(query):
  docs = vectordb_diabetes.similarity_search(query,k=5)
  return docs

def greet_diabetes(question):
  client = Together(api_key="API_KEY")
  context =getcontext_diabetes(question)[0].page_content
  response = client.chat.completions.create(
      model="meta-llama/Llama-3-8b-chat-hf",
      messages=[{"role": "user","content": "[INST] Answer the question in detail based on the following context: "+context+ " Question: "+question+"[/INST]"}],
  )
  # return(response.choices[0].message.content)
  print(response.choices[0].message.content)
  st.text_area("Output", response.choices[0].message.content, height=300)

def getcontext_asthma(query):
  docs = vectordb_asthma.similarity_search(query,k=5)
  return docs

def greet_asthma(question):
  client = Together(api_key="API_KEY")
  context =getcontext_asthma(question)[0].page_content
  response = client.chat.completions.create(
      model="meta-llama/Llama-3-8b-chat-hf",
      messages=[{"role": "user","content": "[INST] Answer the question in detail based on the following context: "+context+ " Question: "+question+"[/INST]"}],
  )
  # return(response.choices[0].message.content)
  print(response.choices[0].message.content)
  st.text_area("Output", response.choices[0].message.content, height=300)


def getcontext_dialysis(query):
  docs = vectordb_dialysis.similarity_search(query,k=5)
  return docs

def greet_dialysis(question):
  client = Together(api_key="API_KEY")
  context =getcontext_dialysis(question)[0].page_content
  response = client.chat.completions.create(
      model="meta-llama/Llama-3-8b-chat-hf",
      messages=[{"role": "user","content": "[INST] Answer the question in detail based on the following context: "+context+ " Question: "+question+"[/INST]"}],
  )
  # return(response.choices[0].message.content)
  print(response.choices[0].message.content)
  st.text_area("Output", response.choices[0].message.content, height=300)

# Function definitions
def answer_question(image_url):
    # Function to answer question from image
    image = Image.open(image_url)
    prompt = "[INST] <image>\nWhat is shown in this image? [/INST]"
    inputs = processor(prompt, image, return_tensors="pt").to("cuda:0")
    output = model.generate(**inputs, max_new_tokens=1000)
    print(processor.decode(output[0], skip_special_tokens=True))
    return processor.decode(output[0], skip_special_tokens=True)

def process_response(prompt, response):
    # Function to process response
    Question = response
    prompt = f"{prompt} Medical Question: {Question} Medical Answer:"
    response = llm(prompt, max_tokens=10000)['choices'][0]['text']
    return response

def process_response_question(prompt):
    # Function to process response to question
    Question = prompt
    prompt = f"You are an expert and experienced from the healthcare and biomedical domain with extensive medical knowledge and practical experience. Your name is OpenBioLLM, and you were developed by Saama AI Labs with Open Life Science AI. who's willing to help answer the user's query with explanation. In your explanation, leverage your deep medical expertise such as relevant anatomical structures, physiological processes, diagnostic criteria, treatment guidelines, or other pertinent medical concepts. Use precise medical terminology while still aiming to make the explanation clear and accessible to a general audience. Medical Question: {Question} Medical Answer:"
    response = llm(prompt, max_tokens=10000)['choices'][0]['text']
    print(response)
    st.text_area("Output", response, height=300)


def process_image_and_question(final_prompt, image):
    # Function to process image and question
    if image is not None:
        response = answer_question(image)
        final_response = process_response(final_prompt, response)
        print(final_response)
        st.text_area("Output", final_response, height=300)



def update_visibility(selected_option):
    # Function to update visibility based on dropdown selection
    if selected_option == "General Question Doctor":
        q = st.text_input("Enter Question")
        if st.button("Run"):
            process_response_question(q)
    elif selected_option == "Prescription Questions":
        image = st.file_uploader("Upload Image", type=["jpg", "png"])
        final_prompt = st.text_area("Enter Text")
        if st.button("Run"):
            process_image_and_question(final_prompt, image)

    elif selected_option == "Diabetes Doctor":
        q = st.text_input("Enter Question")
        if st.button("Run"):
            greet_diabetes(q)

    elif selected_option == "Asthma Doctor":
        q = st.text_input("Enter Question")
        if st.button("Run"):
            greet_asthma(q)

    elif selected_option == "Dialysis Doctor":
      q = st.text_input("Enter Question")
      if st.button("Run"):
          greet_dialysis(q)

    else:
        st.empty()

# Streamlit UI
option_dropdown = st.selectbox("Select Option", ["Select Option", "General Question Doctor", "Prescription Questions", "Diabetes Doctor", "Asthma Doctor", "Dialysis Doctor"])
update_visibility(option_dropdown)
