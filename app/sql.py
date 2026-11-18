import os
import re
import sqlite3
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
import time
from groq import Groq, RateLimitError

def call_groq_with_retry(groq_client, **kwargs):
    for attempt in range(10):
        try:
            return groq_client.chat.completions.create(**kwargs)
        except (RateLimitError, Exception) as e:
            if "429" in str(e) or "RateLimit" in str(e) or isinstance(e, RateLimitError):
                time.sleep(5.0)
            else:
                raise e
    return groq_client.chat.completions.create(**kwargs)


# ============================================================
# 1. PATHS & ENV
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

GROQ_MODEL = os.getenv("GROQ_MODEL")

if not GROQ_MODEL:
    raise ValueError(
        "GROQ_MODEL is not set in your .env file."
    )

CSV_PATH = BASE_DIR / "resources" / "products.csv"
DB_PATH = BASE_DIR / "resources" / "products.db"




# ============================================================
# 3. GROQ CLIENT
# ============================================================

client = Groq()


# ============================================================
# 4. CREATE SQLITE DATABASE FROM CSV
# ============================================================

def initialize_database(force=False):
    """
    Create products.db from products.csv if the database
    does not already exist or if discount column is stored as TEXT.
    """

    needs_recreation = force or not DB_PATH.exists()

    if DB_PATH.exists() and not force:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.execute("PRAGMA table_info(product)")
            columns = {row[1]: row[2] for row in cursor.fetchall()}
            if columns.get("discount") == "TEXT":
                needs_recreation = True

    if not needs_recreation:
        return

    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"CSV file not found:\n{CSV_PATH}"
        )

    print("Creating SQLite database from CSV...")

    df = pd.read_csv(CSV_PATH)

    # Remove completely empty columns
    df = df.dropna(axis=1, how="all")

    # Remove duplicate column names
    df = df.loc[:, ~df.columns.duplicated()]

    # Clean column names
    df.columns = [
        str(col).strip().replace(" ", "_")
        for col in df.columns
    ]

    # Convert discount column from string with '%' to numeric integer
    if "discount" in df.columns:
        df["discount"] = (
            df["discount"]
            .astype(str)
            .str.replace("%", "", regex=False)
            .str.strip()
        )
        df["discount"] = pd.to_numeric(df["discount"], errors="coerce").fillna(0).astype(int)

    with sqlite3.connect(DB_PATH) as conn:

        df.to_sql(
            "product",
            conn,
            if_exists="replace",
            index=False
        )

    print(f"Database created successfully:")
    print(DB_PATH)

    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")


# ============================================================
# 5. GET DATABASE SCHEMA
# ============================================================

def get_database_schema():
    """
    Read the actual columns and distinct category values from the SQLite product table.
    """

    with sqlite3.connect(DB_PATH) as conn:

        cursor = conn.execute(
            "PRAGMA table_info(product)"
        )

        columns = cursor.fetchall()

        cat_cursor = conn.execute(
            "SELECT DISTINCT category FROM product WHERE category IS NOT NULL"
        )
        categories = [row[0] for row in cat_cursor.fetchall() if row[0]]

    if not columns:
        raise ValueError(
            "The 'product' table does not exist or has no columns."
        )

    schema_lines = []

    for column in columns:

        # PRAGMA table_info:
        # cid, name, type, notnull, default, pk

        name = column[1]
        data_type = column[2]

        schema_lines.append(
            f"- {name} ({data_type})"
        )

    schema_str = "\n".join(schema_lines)
    cat_str = ", ".join(categories)

    return f"{schema_str}\n\nAvailable Categories in Database:\n{cat_str}"


# ============================================================
# 6. SQL GENERATION PROMPT
# ============================================================

def build_sql_prompt():

    schema = get_database_schema()

    return f"""
You are an expert SQLite SQL query generator for an e-commerce
product database.

Your task is to convert the user's natural-language question
into ONE valid SQLite SELECT query.

DATABASE:

Table name:
product

Columns & Categories:
{schema}

IMPORTANT RULES:

1. Generate ONLY a SELECT query.
2. Never generate INSERT.
3. Never generate UPDATE.
4. Never generate DELETE.
5. Never generate DROP.
6. Never generate ALTER.
7. Never generate CREATE.
8. Never generate PRAGMA.
9. Never generate multiple SQL statements.
10. Use only the table and columns provided above.
11. Use SQLite syntax.
12. If the user asks for products, return useful product information.
13. Prefer LIMIT 10 unless the user explicitly requests another number.
14. If the user asks for "best", "top", "highest rated", or quality recommendations, ALWAYS order by rating DESC (i.e. ORDER BY rating DESC).
15. If the user asks for "popular", "trending", or "most reviewed", order by review_count DESC (i.e. ORDER BY review_count DESC).
16. If the user asks for "cheapest" or "lowest price", order by price ASC.
17. If the user asks for "most expensive" or "highest price", order by price DESC.
18. If the user asks for "highest discount", "max discount", "best discount", or "discounted products", order by discount DESC (i.e. ORDER BY discount DESC).
19. For category filtering, map user terms to one of the exact "Available Categories in Database" list above (e.g. map singular/plural variations like 'camera' to 'Cameras', 'laptop' to 'Laptops', 'shoe' to 'Shoes', 'phone' or 'mobile' to 'Mobiles').
20. When matching text (category, product_name, brand), use case-insensitive matching like LOWER(category) = 'cameras' or LOWER(category) LIKE '%camera%'.
21. Do not invent column names or category values.
22. Return the SQL inside <sql> and </sql>.

Example:

User:
Show me Nike products below 5000.

Response:

<sql>
SELECT *
FROM product
WHERE LOWER(brand) = 'nike'
AND price <= 5000
LIMIT 10;
</sql>

Remember:
ONLY return the SQL inside <sql>...</sql>.
"""


# ============================================================
# 7. GENERATE SQL USING GROQ
# ============================================================

def generate_sql_query(question):

    sql_prompt = build_sql_prompt()

    response = call_groq_with_retry(
        client,
        model=GROQ_MODEL,

        messages=[
            {
                "role": "system",
                "content": sql_prompt
            },
            {
                "role": "user",
                "content": question
            }
        ],

        temperature=0.0,

        max_tokens=1024
    )

    ans = response.choices[0].message.content.strip()
    return re.sub(r"<think>.*?</think>", "", ans, flags=re.DOTALL).strip()


# ============================================================
# 8. EXTRACT SQL FROM <sql> TAG
# ============================================================

def extract_sql(response):

    pattern = r"<sql>\s*(.*?)\s*</sql>"

    match = re.search(
        pattern,
        response,
        re.IGNORECASE | re.DOTALL
    )

    if match is None:
        return None

    sql_query = match.group(1).strip()

    # Remove markdown code fences if the model accidentally adds them
    sql_query = sql_query.replace("```sql", "")
    sql_query = sql_query.replace("```", "")

    return sql_query.strip()


# ============================================================
# 9. VALIDATE SQL
# ============================================================

def validate_sql(query):

    if not query:
        return False

    query = query.strip()

    # Must start with SELECT
    if not query.lower().startswith("select"):
        return False

    # Remove trailing semicolon
    query_without_semicolon = query.rstrip(";").strip()

    # Prevent multiple statements
    if ";" in query_without_semicolon:
        return False

    # Dangerous SQL keywords
    forbidden_keywords = [
        "insert",
        "update",
        "delete",
        "drop",
        "alter",
        "create",
        "replace",
        "attach",
        "detach",
        "pragma",
        "vacuum"
    ]

    query_lower = query.lower()

    for keyword in forbidden_keywords:

        pattern = rf"\b{keyword}\b"

        if re.search(pattern, query_lower):

            return False

    # Only allow product table
    tables = re.findall(
        r"\b(?:from|join)\s+([a-zA-Z_][a-zA-Z0-9_]*)",
        query,
        re.IGNORECASE
    )

    for table in tables:

        if table.lower() != "product":

            return False

    return True


# ============================================================
# 10. RUN SQL QUERY
# ============================================================

def run_query(query):

    if not validate_sql(query):

        raise ValueError(
            "Unsafe or invalid SQL query."
        )

    try:

        with sqlite3.connect(DB_PATH) as conn:

            df = pd.read_sql_query(
                query,
                conn
            )

        return df

    except Exception as e:

        print("SQL execution error:", e)

        return None


# ============================================================
# 11. DATA COMPREHENSION PROMPT
# ============================================================

COMPREHENSION_PROMPT = """
You are an expert e-commerce data analyst.

You will receive:

1. A user's question.
2. Product data retrieved from a database.

Your task is to answer the user's question using ONLY the
provided database results.

IMPORTANT RULES:

1. Never invent products.
2. Never invent prices.
3. Never invent ratings.
4. Never invent discounts.
5. Never invent review counts.
6. Never invent URLs.
7. Do not add information that is not present in the data.
8. If the provided data does not answer the question, say:
   "I don't know."
9. Keep the answer concise.
10. If multiple products are returned, use a numbered list.
11. Always append '%' to discount numbers (e.g., if discount is 33, write 'Discount: 33%').

For product recommendations, use this format when the
corresponding columns exist:

1. Product Name — ₹Price — Rating: X — Reviews: Y — Discount: Z%
2. Product Name — ₹Price — Rating: X — Reviews: Y — Discount: Z%

If a URL/product link exists in the data, include it.

Do not create fake links.
"""


# ============================================================
# 12. CONVERT DATABASE RESULTS INTO NATURAL LANGUAGE
# ============================================================

def data_comprehension(question, context):

    if not context:

        return "I don't know."

    user_message = f"""
QUESTION:
{question}

DATA:
{context}
"""

    response = call_groq_with_retry(
        client,
        model=GROQ_MODEL,

        messages=[
            {
                "role": "system",
                "content": COMPREHENSION_PROMPT
            },
            {
                "role": "user",
                "content": user_message
            }
        ],

        temperature=0.2,

        max_tokens=1024
    )

    ans = response.choices[0].message.content.strip()
    return re.sub(r"<think>.*?</think>", "", ans, flags=re.DOTALL).strip()


# ============================================================
# 13. MAIN SQL CHAIN
# ============================================================

def sql_chain(question):

    print("\n========================================")
    print("SQL CHAIN")
    print("========================================")

    print("User question:")
    print(question)

    # --------------------------------------------------------
    # Generate SQL
    # --------------------------------------------------------

    try:

        raw_response = generate_sql_query(question)

    except Exception as e:

        print("Groq SQL generation error:", e)

        return "Sorry, I could not generate the SQL query."

    print("\nRaw LLM response:")
    print(raw_response)

    # --------------------------------------------------------
    # Extract SQL
    # --------------------------------------------------------

    sql_query = extract_sql(raw_response)

    if sql_query is None:

        print("\nCould not extract SQL.")

        return "I don't know."

    print("\nGenerated SQL:")
    print(sql_query)

    # --------------------------------------------------------
    # Validate SQL
    # --------------------------------------------------------

    if not validate_sql(sql_query):

        print("\nSQL validation failed.")

        return "Sorry, I could not safely process that request."

    # --------------------------------------------------------
    # Execute SQL
    # --------------------------------------------------------

    result = run_query(sql_query)

    if result is None:

        return (
            "Sorry, there was a problem executing "
            "the database query."
        )

    # --------------------------------------------------------
    # Empty result
    # --------------------------------------------------------

    if result.empty:

        return (
            "I couldn't find any products matching "
            "your request."
        )

    # --------------------------------------------------------
    # Limit context sent to LLM
    # --------------------------------------------------------

    result_for_llm = result.head(10)

    context = result_for_llm.to_dict(
        orient="records"
    )

    print("\nDatabase results:")
    print(result_for_llm)

    # --------------------------------------------------------
    # Generate final answer
    # --------------------------------------------------------

    answer = data_comprehension(
        question,
        context
    )

    return answer


def sql_chain_with_context(question):
    try:
        raw_response = generate_sql_query(question)
    except Exception as e:
        return "Sorry, I could not generate the SQL query.", [], None

    sql_query = extract_sql(raw_response)
    if sql_query is None:
        return "I don't know.", [], None

    if not validate_sql(sql_query):
        return "Sorry, I could not safely process that request.", [], sql_query

    result = run_query(sql_query)
    if result is None:
        return "Sorry, there was a problem executing the database query.", [], sql_query

    if result.empty:
        return "I couldn't find any products matching your request.", [], sql_query

    result_for_llm = result.head(5)
    context = result.head(10).to_dict(orient="records")
    trimmed_context = [
        {k: v for k, v in row.items() if k in ['product_name', 'brand', 'price', 'rating', 'review_count', 'discount']}
        for row in result_for_llm.to_dict(orient="records")
    ]

    answer = data_comprehension(question, trimmed_context)
    return answer, context, sql_query



# ============================================================
# 14. TEST
# ============================================================

if __name__ == "__main__":

    print("\nInitializing database...")

    initialize_database()

    print("\nDatabase schema:")
    print(get_database_schema())

    question = (
        "show me best 10 cameras" 
        
        
    )

    answer = sql_chain(question)

    print("\n========================================")
    print("FINAL ANSWER")
    print("========================================")

    print(answer)