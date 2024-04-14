
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.schema.output_parser import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import PromptTemplate
from llmsherpa.readers import LayoutPDFReader

llmsherpa_api_url = "https://readers.llmsherpa.com/api/document/developer/parseDocument?renderFormat=all"
pdf_reader = LayoutPDFReader(llmsherpa_api_url)
doc = pdf_reader.read_pdf(pdf_url)

class ChatPDF:
    vector_store = None
    retriever = None
    chain = None

    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100, length_function=len,)
        self.model = Ollama(model="cyberlis/saiga-mistral:7b-lora-custom-q4_K", temperature=0.01)
        self.prompt = PromptTemplate.from_template(""" Используй следующие фрагменты контекста, чтобы в конце ответить на вопрос.
            Если ты не нашел ответа, просто скажи, что не знаешь ответа. Не пытайся выдумывать ответ. 
            {context}
            Вопрос: {question}
            Полезный ответ: """
        )

 
    def ingest(self, pdf_file_path: str):
        docs = PyPDFLoader(file_path=pdf_file_path).load_and_split(text_splitter=self.text_splitter)

        embedding_func = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-m3", model_kwargs={"device": "mps"}, encode_kwargs={"normalize_embeddings": True})
        vector_store = Chroma.from_documents(documents=docs, embedding=embedding_func)
        self.retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 2},
        )

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        self.chain = ({"context": self.retriever | format_docs, "question": RunnablePassthrough()}
                      | self.prompt
                      | self.model
                      | StrOutputParser())

    def ask(self, query: str):
        if not self.chain:
            return "Сначала добавьте документ."

        return self.chain.invoke(query)

    def clear(self):
        self.vector_store = None
        self.retriever = None
        self.chain = None