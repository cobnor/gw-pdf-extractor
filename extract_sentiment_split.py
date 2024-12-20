from transformers import pipeline
from dotenv import load_dotenv
import os
load_dotenv()
api_key=os.getenv("API_KEY")

import sys
import string
import csv
from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import nest_asyncio

nest_asyncio.apply()

def extract_relevant_text(nlp, question, content, min_length=100, chunk_size=512, overlap=100):
    
    # Split the content into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    chunks = text_splitter.split_text(content)
    
    # Run QA model on each chunk
    best_chunk = None
    best_score = 0
    best_answer = None
    best_start = 0
    
    for chunk in chunks:
        result = nlp(question=question, context=chunk)
        answer = result["answer"]
        score = result["score"]
        start = result["start"]

        # If the score is better and answer is valid
        if score > best_score and len(answer) >= min_length:
            best_chunk = chunk
            best_score = score
            best_answer = answer
            best_start = start
    
    print(best_chunk)
    # Extend the answer if it"s too short
    if best_chunk and len(best_answer) < min_length:
        # Expand slightly before and slightly after
        start_index = max(0, best_start - min_length/2)
        end_index = min(len(best_chunk), best_start + len(best_answer) + min_length/2)
        best_answer = best_chunk[start_index:end_index]
    
    return best_answer if best_answer else None

# Extract key data from the pdf using a QA pipeline and sentiment analysis
def summarize(filename, writeFile=False):
    # Set up parser
    parser = LlamaParse(
        api_key = api_key,
        result_type="markdown" #better results than text output
    )

    # Use SimpleDirectoryReader to parse the file
    file_extractor = {".pdf": parser}
    try:
        documents = SimpleDirectoryReader(input_files=[f"data/{filename}.pdf"], file_extractor=file_extractor).load_data()
    except ValueError:
        return

    content = ""

    # Remove all non-printable characters
    for page in documents:
        content += page.text.replace("  ", "") + "\n"

    printable = set(string.printable)
    content = "".join(filter(lambda x: x in printable, content))
    if writeFile:
        with open(filename + ".txt", "w+", encoding="utf-8") as f:
            f.write(content)

    with open("data/questions.txt", "r") as f:
        questions = f.read().split("\n")

    # Model for QA pipeline

    #NOTE: using modernBERT would likely improve accuracy -> need to fine-tune modernBERT for QA
    model_name = "deepset/roberta-base-squad2"


    # Initialize QA pipeline
    qa_nlp = pipeline("question-answering", model=model_name, tokenizer=model_name)
    
    # finbert-tone seems to give better results, distilbert seems to be biased towards overly positive scores
    sentiment_nlp = pipeline("sentiment-analysis", model="yiyanghkust/finbert-tone")

    ans = {}
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=100)
    chunks = text_splitter.split_text(content)

    # QA on each question
    for question in questions:
        print(question)
        best_answer = None
        best_score = 0

        # QA on each chunk
        for chunk in chunks:
            output = qa_nlp(question=question, context=chunk)
            # Only record the best scoring response from all the chunks
            if output["score"] > best_score:
                best_answer = output["answer"]
                best_score = output["score"]

        print("answer:" + best_answer + ", score:", best_score)

        # Must have a score over 0.5. If score below 0.5, assume answer not in text and do sentiment analysis
        if best_score > 0.5:
            ans[question] = best_answer
        # Sentiment analysis
        else:
            # Find text relevant to question that's over a given length
            relevant = extract_relevant_text(qa_nlp, question, content)
            print("\n--------------\n",relevant,"\n--------------\n")
            if relevant:
                sentiment_result = sentiment_nlp(relevant)
                sentiment = sentiment_result[0]["label"]
                sentiment_score = sentiment_result[0]["score"]
            else:
                # If no relevant text can be found of sufficient length
                sentiment = "NONE"
                sentiment_score = 0
            ans[question] = f"{sentiment} (score: {sentiment_score:.2f})"
            print(ans[question])
    return ans

def main():
    results = []

    # First argument specifies filename containing list of files to read from
    with open(sys.argv[1], "r") as f:
        filename = f.readline().strip()
        while filename:
            result = summarize(filename, writeFile=True)
            results.append((filename, result))
            filename = f.readline().strip()

    print(results)

    # Second argument specifies filename to output to
    with open(sys.argv[2], mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        # Write the header row
        writer.writerow(["Report Name"] + [question for question, answer in results[0][1].items()])

        # Write the data rows
        for report_name, questions in results:
            row = [report_name]
            for question, answer in questions.items():
                row.append(answer)
            writer.writerow(row)

if __name__ == "__main__":
    main()
