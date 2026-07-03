from typing import Dict, List

from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate

class Prompting:
    def singleshot_prompting(self, user_query: str, context_docs: List[Dict]) -> str:
        context = "\n".join(f"- {doc['chunk']['text']}" for doc in context_docs)
        return self.singleshot_prompting_template().format(context=context, question=user_query)

    def multishot_prompting(self, user_query: str, context_docs: List[Dict]) -> str:
        context = "\n".join(f"- {doc['chunk']['text']}" for doc in context_docs)
        return self.multishot_prompting_template().format(context=context, question=user_query)

    @staticmethod
    def singleshot_prompting_template() -> PromptTemplate:
        return PromptTemplate(
            input_variables=["context", "question"],
            template=(
                "You are a customer support assistant for a retail store.\n"
                "Answer using only the context below.\n"
                "If information is missing, clearly say so.\n\n"
                "Context:\n{context}\n\n"
                "Question: {question}\n"
                "Answer:"
            ),
        )

    @staticmethod
    def multishot_prompting_template() -> FewShotPromptTemplate:
        examples = [
            {
                "question": "Do you have earbuds in stock?",
                "answer": "Yes, Wireless Earbuds (RT-2001) are available with current stock.",
            },
            {
                "question": "Any promotions on apparel?",
                "answer": "Yes, Summer 10% Off applies to all apparel items until 2026-08-31.",
            },
        ]
        example_prompt = PromptTemplate(
            input_variables=["question", "answer"], template="Q: {question}\nA: {answer}"
        )
        return FewShotPromptTemplate(
            examples=examples,
            example_prompt=example_prompt,
            suffix=(
                "You are a helpful retail support assistant.\n"
                "Use examples to keep answers concise and practical.\n\n"
                "Context:\n{context}\n\n"
                "Question: {question}\n"
                "Answer:"
            ),
            input_variables=["context", "question"],
        )
