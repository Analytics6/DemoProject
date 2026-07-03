from augmentation.rag import RagModel


def run_demo() -> None:
    rag = RagModel(data_dir="data")
    sample_questions = [
        "Is Wireless Earbuds available in stock?",
        "Do you have any promotions on apparel?",
        "What is your return policy?",
    ]

    for question in sample_questions:
        result = rag.agentic_rag(question, model="openai")
        print("=" * 80)
        print(f"Question: {question}")
        print(f"Detected intent: {result.get('intent')}")
        print(f"Answer:\n{result['answer']}")


if __name__ == "__main__":
    run_demo()
