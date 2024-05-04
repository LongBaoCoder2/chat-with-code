import os 
from dotenv import load_dotenv

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import DeepLake
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain


def main():
    load_dotenv(os.path.join(os.curdir, ".env"))

    embedding_model_name = 'text-embedding-3-small'
    embeddings = OpenAIEmbeddings(model=embedding_model_name, disallowed_special=()) 

    # TODO: Make timeout embedding data on database
    def download_repo(repo, destination_dir):
        os.system(f"git clone {repo} {destination_dir}")
        return True
  
    # repo = input("Input repo: ")
    # destination_dir = input("Input destination dir: ")
    # try:
    #     download_repo(repo, destination_dir)
    # except:
    #     print("Error when download github repo.")
    #     return
    
    # Use embedded vector repo before testing on any repo
    username = os.environ['USERNAME']
    db = DeepLake(dataset_path=f"hub://{username}/Torch-Pruning-embeddings", read_only=True, embedding=embeddings)

    retriever = db.as_retriever()
    retriever.search_kwargs['distance_metric'] = 'cos'
    retriever.search_kwargs['fetch_k'] = 100
    # retriever.search_kwargs['maximal_marginal_relevance'] = True
    retriever.search_kwargs['k'] = 10

    model_name = 'gpt-3.5-turbo'
    llm = ChatOpenAI(model=model_name, temperature=0) # Set 0 for testing
    qa = ConversationalRetrievalChain.from_llm(llm, retriever=retriever)

    questions = ['How to prune a network by using Torch-Pruning library?',
             'Which pruning method does Torch-Pruning library support?',
             'What pruning method is appropriate to each common network such as CNN, RNN, MLP?']
    chat_history = []

    for question in questions:
        result = qa({'question': question, 'chat_history': chat_history})
        chat_history.append((question, result['answer']))
        print(f"\tQuestion: {question}\n")
        print(f"\tAnswer: {result['answer']}\n")

if __name__ == "__main__":
    main()    