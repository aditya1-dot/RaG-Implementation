# Import necessary modules and classes
import PIL.Image
import google.generativeai as genai
from IPython.display import display
import ollama
from PIL import Image
import os
from typing import Union
import fitz
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings


# Define the HuggingFace BGE embeddings
model_name = "BAAI/bge-large-en"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
embeddings = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)


class DocumentChatting:
    '''
    Attributes:
        Gapi: str   Google Api
        model: str  Model name
        doc_path: str   Document/Image File Path
    '''
    def __init__(self, doc_path: str, question:str) :
        self.Gapi = 'AIzaSyAo0-zQWQ-Y3XQSTij2bVwIpdC2TeXunCY'
        genai.configure(api_key=self.Gapi)
        self.model = genai.GenerativeModel('gemini-pro-vision')
        self.doc_path = doc_path
        self.question=question
        

    def type_checker(self):
        # Get the file extension
        _, file_extension = os.path.splitext(self.doc_path)
        # Check if the file extension corresponds to PDF or image
        if file_extension.lower() == '.pdf':
            print("PDF uploaded")
            return self.pdf_reader()  # PDF file
        elif file_extension.lower() in ['.jpg', '.jpeg', '.png', '.gif', '.bmp']:
            print("Image uploaded")
            return self.img_reader()  # Image file
        else:
            print("File type not supported!")  # Unknown file type
   
    def pdf_reader(self):
        # Open the PDF file
        pdf_document = fitz.open(self.doc_path)
        text = []
        # Create the output directory if it doesn't exist
        output_directory = "output/"
        os.makedirs(output_directory, exist_ok=True)
        
        for page_num in range(len(pdf_document)):
            # Get the page
            page = pdf_document.load_page(page_num)
            # Convert the page to an image
            pix = page.get_pixmap()
            image_path = f"output/{page_num}i.png"
            pix.pil_save(image_path)
            
        for path_suffix in range(len(pdf_document)):
                image_path = f'output/{path_suffix}i.png'
                image = Image.open(image_path)
                # Generate content from the image
                response = self.model.generate_content(["Extract all the text from the image and display it ", image], stream=True)
                response.resolve()
                text.append(response.text)
        print(text)
        self.create_text_file(text)
        return text
    
    def create_text_file(self, data: Union[list[str], str]):
        file_path = self.doc_path.split('.')[0] + ".txt"
        with open(file_path, 'w') as file:
            # If data is a list, write each element on a new line
            if isinstance(data, list):
                for item in data:
                    file.write(item + '\n')
            # If data is a string, write it directly to the file
            elif isinstance(data, str):
                file.write(data)
        return self.txtReader_embedding(file_path)
                
    def img_reader(self):
        img = PIL.Image.open(self.doc_path)
        response = self.model.generate_content(["Extract all the text from the image and display it ", img], stream=True)
        response.resolve()
        self.create_text_file(response.text)
        return response.text    
    
    def txtReader_embedding(self, file_path):
        # Load the text and process it for embedding
        loader = TextLoader(file_path)
        print(file_path)
        data = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=10)
        splits = text_splitter.split_documents(data)
        print(data,"\n")
        vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings, persist_directory="stores/docs")
        def format_docs(data):
            return "\n\n".join(doc.page_content for doc in data)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 1})
        retrieved_docs = retriever.invoke(self.question)
        formatted_context = format_docs(retrieved_docs)
        return self.ollama_llm(formatted_context)
    
    def ollama_llm(self, context):
        formatted_prompt = f"Question: {self.question}\n\nContext: {context}"
        response = ollama.chat(model='mistral', 
                               messages=[{'role': 'user','content': formatted_prompt}],
                               stream=False)
        print(response["message"]["content"])

parsed = DocumentChatting(doc_path="CTScan.jpeg",question="Give me the results of CT Scan")
display(parsed.type_checker())
