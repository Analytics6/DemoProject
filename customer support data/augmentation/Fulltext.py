from typing import Dict, List

from langchain_core.documents import Document


def build_full_text_documents(
    inventory_docs: List[Dict], promotion_docs: List[Dict], faq_docs: List[Dict]
) -> List[Document]:
    """Build unified LangChain documents for retrieval."""
    full_docs: List[Document] = []

    for item in inventory_docs:
        text = (
            f"Inventory Item {item['sku']}: {item['name']}. "
            f"Category: {item['category']}. Price: ${item['price']}. "
            f"Stock: {item['stock']}. Description: {item['description']}."
        )
        full_docs.append(
            Document(
                page_content=text,
                metadata={"doc_id": f"INV-{item['sku']}", "type": "inventory", **item},
            )
        )

    for item in promotion_docs:
        text = (
            f"Promotion {item['promo_id']}: {item['title']}. "
            f"Description: {item['description']}. Category: {item['category']}. "
            f"Discount: {item['discount_percent']} percent. Valid until: {item['valid_until']}."
        )
        full_docs.append(
            Document(
                page_content=text,
                metadata={"doc_id": f"PRO-{item['promo_id']}", "type": "promotions", **item},
            )
        )

    for item in faq_docs:
        text = f"General FAQ {item['id']}: Q: {item['question']} A: {item['answer']}"
        full_docs.append(
            Document(
                page_content=text,
                metadata={"doc_id": f"FAQ-{item['id']}", "type": "general", **item},
            )
        )

    return full_docs


def build_ticket_documents(ticket_docs: List[Dict]) -> List[Document]:
    """Convert support tickets into LangChain documents for RAG indexing."""
    documents: List[Document] = []
    for item in ticket_docs:
        resolution = item.get("resolution") or "Pending resolution."
        text = (
            f"Support Ticket #{item['id']}: {item['subject']}. "
            f"Customer: {item['customer_name']}. Category: {item.get('category', 'general')}. "
            f"Status: {item['status']}. Priority: {item['priority']}. "
            f"Order: {item.get('order_id', 'N/A')}. Product: {item.get('product', 'N/A')}. "
            f"Description: {item.get('description', '')}. "
            f"Resolution: {resolution}."
        )
        documents.append(
            Document(
                page_content=text,
                metadata={
                    "doc_id": f"TKT-{item['id']}",
                    "type": "ticket",
                    "ticket_id": item["id"],
                    "category": item.get("category", "general"),
                    "status": item["status"],
                    "priority": item["priority"],
                    **{k: v for k, v in item.items() if isinstance(v, (str, int, float))},
                },
            )
        )
    return documents
