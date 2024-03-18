import streamlit as st
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import AnalyzeDocumentChain
from langchain_community.callbacks import get_openai_callback
from langchain.chains.question_answering import load_qa_chain
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

BASE_DIR = os.path.join(os.getcwd())

def get_all_documents():
    all_docs = []
    for root, dirs, files in os.walk(BASE_DIR):
        for file in files:
            if file.endswith('.pdf'):
                path = os.path.join(root, file)
                all_docs.append({'title': file, 'path': path})
    return all_docs

def get_documents(category):
    category_path = os.path.join(BASE_DIR, category)
    return sorted([doc for doc in os.listdir(category_path) if doc.endswith('.pdf')])

def get_categories():
    try:
        # Update the path to target preloaded_docs folder
        preloaded_docs_path = os.path.join(BASE_DIR, 'preloaded_docs')
        return sorted(next(os.walk(preloaded_docs_path))[1])
    except StopIteration:
        st.error("Fout bij het openen van categorieën. Controleer of de map bestaat en niet leeg is.")
        return []


def get_documents(category):
    # Construct the correct path to the category within preloaded_docs
    category_path = os.path.join(BASE_DIR, 'preloaded_docs', category)
    try:
        return sorted([doc for doc in os.listdir(category_path) if doc.endswith('.pdf')])
    except FileNotFoundError:
        st.error(f"Map niet gevonden: {category_path}. Controleer of de map bestaat.")
        return []



def extract_text_from_pdf_by_page(file_path):
    pages_text = []
    with open(file_path, 'rb') as file:
        reader = PdfReader(file)
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages_text.append(text)
    return pages_text

def process_document(document_path, user_question):
    with st.spinner('Denken...'):
        # Extract text from the document
        document_pages = extract_text_from_pdf_by_page(document_path)
        if not document_pages or all(page.strip() == "" for page in document_pages):
        
            st.error("Geen tekst gevonden in het document, controleer of de pdf goed is geladen.")
            return

        embeddings = OpenAIEmbeddings()
        knowledge_base = FAISS.from_texts(document_pages, embeddings)
        docs = knowledge_base.similarity_search(user_question)
        document_text = " ".join([doc.page_content for doc in docs])

        template = """
        Je bent expert in pensioenen en in het analyeren van juridische documenten. Je hebt een diepe kennis van de documenten die zijn worden geselecteerd.
        De geuploade documenten zijn de pensioenafspraken van een specifieke klant, deze klant houden we anoniem, maar het is dus wel specifiek voor die klant.
        Je geeft concreet en duidelijk antwoord.
        Rond geen getallen af, deze geef je altijd precies.
        Wees extra accuraat op nummers, getallen, en specifieke details.
        Als het je antwoord sterker maakt, gebruik dan ook een directe quote.
        Bij een vraag over percentages in opbouw et cetera, bereken het precieze getal en maak geen afrondingen. 
        Zorg ervoor dat je accurraat bent omdat het om juridische documenten gaat, dus wees daar heel scherp op.
        De gebruiker is een medewerker van de pensioenafdeling en zal dus vragen stellen die ofwel van de klant komenof intern. 
        Analyseer de vraag en geef duidelijke instructies als antwoord op de vraag, disclaimers en verdere informatie is niet nodig.
        Je enige doel is de vraag beantwoorden en de gebruiker efficient met het systeem om te laten gaan.

        Geef aan het eind van je antwoord een korte conclusie waarin je de vraag van de gebruiker zo direct mogelijk beantwoord.

        Gegeven de tekst uit de documenten: '{document_text}', en de vraag van de gebruiker: '{user_question}', hoe zou je deze vraag beantwoorden met inachtneming van de bovenstaande instructies?
        """
        
        prompt = ChatPromptTemplate.from_template(template)

        
        # Perform similarity search
        llm = ChatOpenAI(api_key=st.secrets["OPENAI_API_KEY"], model="gpt-4-turbo-preview", temperature=0, streaming=True)
        chain = prompt | llm | StrOutputParser() 
        return chain.stream({
            "document_text": document_text,
            "user_question": user_question,
        })
    


    

def main():
    st.title("Pensioenbot - testversie 0.1.2.")

    # Get categories (clients) and allow user selection
    clients = get_categories()
    selected_client = st.selectbox("Kies een bedrijf:", clients)

    # Update to fetch documents from the selected client folder
    documents = get_documents(selected_client)
    selected_doc_title = st.selectbox("Kies een document:", documents)
    selected_document_path = os.path.join(BASE_DIR, 'preloaded_docs', selected_client, selected_doc_title)
    
    with open(selected_document_path, "rb") as pdf_file:
        st.download_button(
            label="Download PDF",
            data=pdf_file,
            file_name=selected_doc_title,
            mime="application/pdf"
        )

    user_question = st.text_input("Wat wil je graag weten?")
    if user_question:
        answer = process_document(selected_document_path, user_question)
        st.write(answer)
    
if __name__ == "__main__":
    main()
