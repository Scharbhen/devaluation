from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import Ollama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from termcolor import colored


data_path = "./data/itogs.pdf"

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    length_function=len,)

documents = PyPDFLoader(data_path).load_and_split(text_splitter=text_splitter)

embedding_func = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-m3", model_kwargs={"device": "mps"}, encode_kwargs={"normalize_embeddings": True})
vectordb = Chroma.from_documents(documents, embedding=embedding_func)

retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 2})
llm=Ollama(model="cyberlis/saiga-mistral:7b-lora-custom-q4_K", temperature=0.01)

template = """Используй следующие фрагменты контекста, чтобы в конце ответить на вопрос.
Если ты не нашел ответа, просто скажи, что не знаешь ответа. Не пытайся выдумывать ответ.
Используй максимум три предложения и старайся отвечать максимально кратко. 
{context}
Вопрос: {question}
Полезный ответ: """

custom_rag_prompt = PromptTemplate.from_template(template)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | custom_rag_prompt
    | llm
    | StrOutputParser()
)

questions = [
    "Кто автор документа?",
    "Каким подразделениям адресовано письмо?",
    "Есть ли положительное решение о заключении договора?",
    "На какую сумму заключается договор?",
    "С каким контрагентом необходимо заключить договор?",
    "Какой гарантированный обьем работ в человекоднях планируется в договоре?"
]

for q in questions:
    print(colored("Вопрос: "+q,"yellow"))
    for chunk in rag_chain.stream(q):
        print(chunk, end="", flush=True)
    print("\n")