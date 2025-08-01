from transformers import pipeline

# Load the QA pipeline
qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
def answer_question(question: str, context: str) -> str:
    try:
        result = qa_pipeline(question=question, context=context)
        return result['answer']
    except Exception as e:
        return f"Error: {str(e)}"
