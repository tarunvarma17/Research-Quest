################################# IMPORT LIBRARIES #################################
import fitz
import time
import re
import collections
import torch
import timeit
import os
from langchain.docstore.document import Document
import langchain_text_splitters
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import (
    sentence_transformer,
)
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.query_constructor.schema import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
import google.generativeai as genai 
from langchain_google_genai import ChatGoogleGenerativeAI
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import TextIteratorStreamer
torch.set_default_device("cuda")
from threading import Thread
from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser
from langchain.prompts import ChatPromptTemplate
################################# ENVIRONMENTAL VARIABLES #################################
GOOGLE_API_KEY='AIzaSyDfxSN9QK3yK595HLxp0_1ZHGz6Y8lVQp8'
article_name = "Attention Is All You Need"
MODEL_NAME = "TheBloke/laser-dolphin-mixtral-2x7b-dpo-GPTQ"
MODEL_TOKENIZER = "TheBloke/laser-dolphin-mixtral-2x7b-dpo-GPTQ"

################################# DEFINING LLM MODEL #################################

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME,
                                            device_map="auto",
                                            trust_remote_code=False,
                                            revision="main")
# model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", torch_dtype="auto", flash_attn=True, flash_rotary=True, fused_dense=True, device_map="cuda", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_TOKENIZER, trust_remote_code=True, device_map="cuda")
    

def main(fname, query):
    ################################# READ PDF #################################
    doc = fitz.open(fname) # open a document
    out = open("output.txt", "wb") # create a text output
    for page in doc: # iterate the document pages
        text = page.get_text().encode("utf8") # get plain text (is in UTF-8)
        out.write(text) # write text of page
        out.write(bytes((20,))) # write page delimiter (form feed 0x0C)
    out.close()
    toc = doc.get_toc() # get table of Contents
    # Open the file in read mode
    with open('output.txt', 'r') as file:
        content = file.read()
    docs = None

    if len(toc) > 5:
        i = 0
        j = 1
        st_ind = 0
        d = collections.defaultdict(list)
        heading_count = 0
        while j < len(toc):
            cur_topic_number, cur_topic, cur_topic_page_number = toc[i]
            cur_topic = ''.join(i for i in cur_topic if i not in "1234567890")
            next_topic_number, next_topic, next_topic_page_number = toc[j]
            next_topic = ''.join(i for i in next_topic if i not in "1234567890")
            if cur_topic_number == 1 and next_topic_number == 1:
                heading_count += 1
                st_ind = content.find(f'{heading_count}\n{cur_topic}') 
                end_ind = content.find(f'{heading_count+1}\n{next_topic}')
                if st_ind == -1 or end_ind == -1:
                    st_ind = content.find(f'{heading_count}.{cur_topic}')
                    end_ind = content.find(f'{heading_count+1}.{next_topic}')
                d[cur_topic].append(content[st_ind:end_ind])
                metadata = [article_name, cur_topic_page_number,cur_topic]
                d[cur_topic].append(metadata)
                i = j
                j += 1
            elif cur_topic_number != 1 and next_topic_number != 1:
                i += 1
                j += 1
            elif cur_topic_number != 1 and next_topic_number == 1:
                i = j
                j += 1
            elif cur_topic_number == 1 and next_topic_number != 1:
                j += 1

        for i in range(len(toc)-1,-1,-1):
            if toc[i][0] == 1:
                heading_count += 1
                cur_topic = toc[i][1]
                st_ind = content.find(f'{heading_count}\n{cur_topic}')
                cur_topic_page_number = toc[i][2]
                break
            d[cur_topic].append(content[st_ind:])
            d[cur_topic].append([article_name,cur_topic_page_number,cur_topic])

        ################################# STORE PDF INTO VECTOR DB #################################
        docs = []
        for k,v in d.items():
            if not d[k]:
                continue
            docs.append(Document(page_content=d[k][0],
                    metadata = {
                        "section" : d[k][1][2],
                        "page_number" : d[k][1][1], 
                        "type" : "text"
                    }))


    ################################# STORE PDF INTO VECTOR DB #################################
    embedding_function = sentence_transformer.SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap  = 20,
        length_function = len,
    )

    if not docs:
        docs = [Document(content)]
    dbdocs = text_splitter.split_documents(docs)
    db = Chroma.from_documents(dbdocs, embedding_function)



    ################################# DEFINING META DATA INFO #################################
    meta_data_field_info = [
        AttributeInfo(
            name= "page_number",
            description= "This contains the page number of the content in the research paper",
            type= "int"
        ),

        AttributeInfo(
            name= "section",
            description= "This describes the heading or the sub-heading under which the content was found in the reseach paper",
            type= "string"
        ),

        # AttributeInfo(
        #     name= "section_sequence",
        #     description= "This describes the paragraph sequence of a given section in the research paper",
        #     type= "int"
        # ),

        AttributeInfo(
            name= "type",
            description= "This field describes the type of the content inside a given section, can take text, image or table as the value",
            type= "string",
        )
    ]


    ################################# DEFINING SPECFS FOR GEMINI #################################
    ################################# CREATING SELF QUERY RAG #################################


    genai.configure(api_key=GOOGLE_API_KEY)
    ragmodel = genai.GenerativeModel(model_name = "gemini")
    llm = ChatGoogleGenerativeAI(model="gemini-pro",google_api_key=GOOGLE_API_KEY)
    document_content_description = "Sections of a research paper"
    retriever = SelfQueryRetriever.from_llm(
        llm,
        db,
        document_content_description,
        meta_data_field_info,
    )

    res  = retriever.invoke(query)



    ################################# DEFINING LLM MODEL #################################

    response = ResponseSchema(name="response",
                                description="answer to the question based on the context's")

    # suggestion_question_1 = ResponseSchema(name="suggestion_question_1",
    #                                       description="suggestion question 1 based on the given context's")
    # suggestion_question_1_answer = ResponseSchema(name="suggestion_question_1_answer",
    #                                       description="answer to the suggested question 1 based on the given context's")

    # suggestion_question_2 = ResponseSchema(name="suggestion_question_2",
    #                                       description="suggestion question 2 based on the given context's")
    # suggestion_question_2_answer = ResponseSchema(name="suggestion_question_2_answer",
    #                                       description="answer to the suggested question 2 based on the given context")

    response_schemas = [response, ]
                        # suggestion_question_1, suggestion_question_1_answer,
                        # suggestion_question_2, suggestion_question_2_answer]

    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = output_parser.get_format_instructions()


    ################################# TEMPLATE STRING #################################
    template_string = """
    <|im_start|>system
    You are assisting a researcher in his research. With the given context, help reseacher to answer his questions about the implemmentations of the paper.<|im_end|>
    <|im_start|>user
    Use the following context's (below delimited by triple backticks) and answer the following question (enclosed in triple backticks)
    if the contexts provided are not relating to the query, provide the below given default response (enclosed in triple backticks).

    # context 1: ```{context_1}```

    # context 2: ```{context_2}```

    # context 3: ```{context_3}```

    question : ```{query}```

    #############################################################################################################
    {format_instructions}
    #############################################################################################################
    <|im_end|>
    <|im_start|>assistant
    """
    prompt = ChatPromptTemplate.from_template(template=template_string)
    messages = prompt.format_messages(context_1=res[0], context_2=res[1] if len(res) >=1 else "" , context_3=res[2] if len(res) >=1 else "", query=query,
                                    format_instructions=format_instructions)

    inputs_tokenizer = tokenizer(messages[0].content, return_tensors="pt", return_attention_mask=False)
    # Setup the text streamer
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True)
    outputs = model.generate(**inputs_tokenizer, max_length=2000, streamer=streamer)
    generated_output = tokenizer.batch_decode(outputs, )[0]
    generated_output = generated_output[len(messages[0].content):].strip()


    # Define a regex pattern to extract content within triple backticks
    pattern = re.compile(r'```(.*?)```', re.DOTALL)

    # Find all matches
    matches = pattern.findall(generated_output)

    # Extracted content
    if matches:
        extracted_content = f'''```{matches[0]}```'''

        # Remove the last occurring comma
        extracted_content = re.sub(r',\s*}', '}```', extracted_content)
    else:
        extracted_content = '''
        ```json 
        {
            'response': '',
            'suggestion_question_1': '',
            'suggestion_question_1_answer': '',
            'suggestion_question_2': '',
            'suggestion_question_2_answer': ''
        }
        ```'''
    response_as_dict = output_parser.parse(extracted_content)
    del db
    return response_as_dict