import os
from transformers import AutoModel, AutoConfig, AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
from torch import cuda, bfloat16, LongTensor
import transformers
import torch
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFacePipeline
from langchain.chains import ConversationalRetrievalChain
from streamlit_chat import message
import streamlit as st

# Set your Hugging Face token
os.environ["HUGGINGFACE_TOKEN"] = "Your Huggingface Token"

model_id = 'meta-llama/Llama-2-7b-chat-hf'

device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'

# Set quantization configuration to load large model with less GPU memory
# This requires the `bitsandbytes` library
bnb_config = transformers.BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=bfloat16
)

# Begin initializing HF items, you need an access token
hf_auth = 'Your Huggingface Token'
model_config = AutoConfig.from_pretrained(
    model_id,
    use_auth_token=hf_auth
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    config=model_config,
    quantization_config=bnb_config,
    device_map='auto',
    use_auth_token=hf_auth
)

# Enable evaluation mode to allow model inference
model.eval()


tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    use_auth_token=hf_auth
)

stop_list = ['\nHuman:', '\n```\n']

stop_token_ids = [tokenizer(x)['input_ids'] for x in stop_list]
stop_token_ids = [LongTensor(x).to(device) for x in stop_token_ids]

# Define custom stopping criteria object
class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for stop_ids in stop_token_ids:
            if torch.eq(input_ids[0][-len(stop_ids):], stop_ids).all():
                return True
        return False

stopping_criteria = StoppingCriteriaList([StopOnTokens()])
generate_text = transformers.pipeline(
    model=model,
    tokenizer=tokenizer,
    return_full_text=True,  # langchain expects the full text
    task='text-generation',
    # We pass model parameters here too
    stopping_criteria=stopping_criteria,# without this model rambles during chat
    temperature=0.1,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
    max_new_tokens=512,  # max number of tokens to generate in the output
    repetition_penalty=1.1  # without this output begins repeating
)

loader = PyPDFDirectoryLoader("/content")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
all_splits = text_splitter.split_documents(documents)


model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {"device": "cuda"}

embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)

# Storing embeddings in the vector store
vectorstore = FAISS.from_documents(all_splits, embeddings)

vectorstore.save_local("faiss_index")
# Load embedding
loaded_vectorstore = FAISS.load_local("faiss_index", embeddings)

llm = HuggingFacePipeline(pipeline=generate_text)

chain = ConversationalRetrievalChain.from_llm(llm, vectorstore.as_retriever(), return_source_documents=True)


# testing the model

chat_history = []

def get_response(user_input):
    global chat_history
    query = user_input
    result = chain({"question": query, "chat_history": chat_history})
    chat_history = [(query, result["answer"])]
    return result

st.title("LegalEase")

if 'user_input' not in st.session_state:
    st.session_state['user_input'] = []

if 'response' not in st.session_state:
    st.session_state['response'] = []

if 'response_source_documents' not in st.session_state:
    st.session_state['response_source_documents'] = []

def get_text():
    input_text = st.text_input("write here", key="input")
    return input_text


user_input = get_text()

if user_input:
    result = get_response(user_input)
    response_answer = result['answer']
    response_source_documents = result['source_documents']

    # Store the output
    st.session_state.response.append(result['answer'])
    st.session_state.user_input.append(user_input)
    st.session_state.response_source_documents.append(result['source_documents'])


message_history = st.empty()

if st.session_state['user_input']:

    for i in range(len(st.session_state['user_input']) - 1, -1, -1):
        # This function displays user input
        message(st.session_state["user_input"][i],
                key=str(i), avatar_style="icons")

        # This function displays response
        message(st.session_state['response'][i],
                avatar_style="miniavs", is_user=True
                , key=str(i) + 'response_by_ai')

        # This function displays response source documents
        for j, doc in enumerate(st.session_state['response_source_documents'][i]):
            formatted_doc = f"Document {j + 1}\nSource: {doc.metadata['source']}, Page: {doc.metadata['page']}\n\n{doc.page_content}"
            message(formatted_doc,
                    avatar_style="miniavs", is_user=True,
                    key=f"{i}_doc{j}_by_ai")
