from semantic_router import Route, SemanticRouter
from semantic_router.encoders import HuggingFaceEncoder


encoder = HuggingFaceEncoder(
    name="sentence-transformers/all-MiniLM-L6-v2"
)


faq = Route(
    name="faq",
    utterances=[
        "What is the return policy?",
        "What is your return policy?",
        "How can I return a product?",
        "How do I return an item?",
        "Can I return my order?",
        "I want to return a product",
        "How many days do I have to return a product?",
        "What are the return rules?",
        "return policy",
        "return period",
        "Can I get a refund?",
        "What is your refund policy?",
        "How do I get a refund?",
        "refund status",
        "refund time",
        "Why has my refund not been received?",
        "What is the exchange policy?",
        "Can I exchange a product?",
        "replace item",
        "replace product",
        "damaged product received",
        "wrong product",
        "How do I track my order?",
        "Where is my order?",
        "How can I track my order?",
        "track order",
        "cancel order",
        "Can I cancel my order?",
        "change delivery address",
        "order delayed",
        "What payment methods do you accept?",
        "Which payment methods are accepted?",
        "payment methods",
        "UPI pay",
        "payment failed",
        "money deducted no order",
        "cash on delivery",
        "delivery time",
        "delivery date",
        "Do you offer international shipping?",
        "Can you ship internationally?",
        "forgot password",
        "change mobile number",
        "create account",
        "apply coupon",
        "coupon not working",
        "contact support",
        "customer support",
        "share OTP",
        "suspicious message"
    ]
)


sql = Route(
    name="sql",
    utterances=[
        "show me camera",
        "show me cameras",
        "show me best camera",
        "show me 10 best cameras",
        "show me best 10 camera",
        "find cameras",
        "recommend top rated camera",
        "I want to buy Nike shoes that have 50% discount",
        "Show me all the products that are available in red color",
        "What are the top 10 best-selling products in the last month?",
        "Find me all the products that are priced between 50 and 100",
        "Show me the products that have been added in the last 7 days",
        "show me laptops",
        "find cheap laptops",
        "show me shoes",
        "show me mobiles",
        "recommend me a good television",
        "show best rated products",
        "search for headphones",
        "list top smartwatches",
        "show me products",
        "find cheap products"
    ]
)


router = SemanticRouter(
    encoder=encoder,
    routes=[faq, sql],
    auto_sync="local"
)


def ask_with_context(query):
    from faq import faq_chain_with_context
    from sql import sql_chain_with_context

    route_choice = router(query)
    route_name = route_choice.name if route_choice is not None else "sql"

    if route_name == "faq":
        answer, context = faq_chain_with_context(query)
        return answer, context, "faq", None
    else:
        answer, context, sql_query = sql_chain_with_context(query)
        return answer, context, "sql", sql_query



if __name__ == "__main__":

    queries = [
        "what is the return policy?",
        "i want to buy nike shoes that have 50% discount",
        "how can I return a product?",
        "show me red products",
        "what payment methods do you accept?"
    ]

    for query in queries:
        result = router(query)

        print(f"\nQuery: {query}")
        print(f"Result: {result}")

        if result is not None:
            print(f"Route: {result.name}")
        else:
            print("Route: NONE")