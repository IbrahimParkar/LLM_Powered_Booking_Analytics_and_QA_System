# Import necessary libraries
import pandas as pd
import numpy as np
import requests
import sqlite3
import warnings
from sentence_transformers import SentenceTransformer
import faiss
import pycountry
import re
from word2number import w2n
import os
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# Suppress future warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Set Hugging Face API details
API_KEY = os.getenv("HF_KEY")
API_ENDPOINT = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3"

# Load dataset
df = pd.read_csv(r"C:\Users\Ibrahim\Desktop\AI ML Task\hotel_bookings.csv")


# 3. Retrieval-Augmented Question Answering (RAG)
def generate_and_save_embeddings(df, index_file="booking_faiss.index"):

    # Handle missing values
    df.fillna({'children': 0, 'country': 'unknown', 'agent': 0, 'company': 0}, inplace=True)

    # Capitalize month names for consistency
    df['arrival_date_month'] = df['arrival_date_month'].str.capitalize()

    # Create a unique identifier for each record
    df['embedding_id'] = df.index.astype(str)

    # Function to map country codes to full country names
    def get_country_name(country_code):
        try:
            return pycountry.countries.get(alpha_3=country_code).name
        except AttributeError:
            return "Unknown" if country_code == "unknown" else country_code

    # Apply country name mapping
    df['country'] = df['country'].apply(get_country_name)

    # Special Cases mapped for consistency
    country_name_mapping = {
        "CN": "China", "Türkiye": "Turkey", "Korea, Republic of": "South Korea", "Iran, Islamic Republic of": "Iran",
        "Venezuela, Bolivarian Republic of": "Venezuela", "Taiwan, Province of China": "Taiwan",
        "Bolivia, Plurinational State of": "Bolivia",
        "Tanzania, United Republic of": "Tanzania", "Russian Federation": "Russia",
        "United States Minor Outlying Islands": "US Outlying Islands",
        "Syrian Arab Republic": "Syria", "Czechia": "Czech Republic", "Lao People's Democratic Republic": "Laos",
        "Macao": "Macau",
        "Palestinian Territory, Occupied": "Palestine", "Micronesia, Federated States of": "Micronesia",
        "Moldova, Republic of": "Moldova",
        "Saint Kitts and Nevis": "St. Kitts and Nevis", "Saint Lucia": "St. Lucia",
        "Saint Vincent and the Grenadines": "St. Vincent and the Grenadines"
    }

    # Apply the mapping during preprocessing
    df['country'] = df['country'].replace(country_name_mapping)

    # Create a unified text column for embeddings
    df['text'] = (
            df['hotel'] + " booking from " + df['country'] +
            " with revenue " + (df['adr'] * (df['stays_in_week_nights'] + df['stays_in_weekend_nights'])).round(
        2).astype(str) +
            " on " + df['arrival_date_month'] + " " + df['arrival_date_year'].astype(str) +
            ", Canceled: " + df['is_canceled'].astype(str)
    )

    # Extract necessary text for FAISS
    booking_texts = df["text"].tolist()

    # Initialize embedding model
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    # Generate embeddings
    embeddings = embedding_model.encode(booking_texts, convert_to_numpy=True)

    # Initialize FAISS index
    dimension = embeddings.shape[1]
    faiss_index = faiss.IndexFlatL2(dimension)

    # Add embeddings to FAISS
    faiss_index.add(embeddings)

    # Save FAISS index
    faiss.write_index(faiss_index, "booking_faiss.index")
    print(f"Embeddings and FAISS index saved to {index_file}.")


# Compute and Store Analytics in DB
def to_python_type(value):
    if isinstance(value, (np.int64, np.float64)):
        return float(value) if isinstance(value, np.float64) else int(value)
    return value


# Call the function after calculating your analytics
def to_python_type(value):
    if isinstance(value, (np.int64, np.float64)):
        return float(value) if isinstance(value, np.float64) else int(value)
    return value

def store_analytics(df):
    conn = sqlite3.connect("analytics.db")
    cursor = conn.cursor()

    # Delete existing tables if they exist
    cursor.execute("DROP TABLE IF EXISTS analytics")
    cursor.execute("DROP TABLE IF EXISTS revenue_by_year")
    cursor.execute("DROP TABLE IF EXISTS revenue_by_month")
    cursor.execute("DROP TABLE IF EXISTS cancellations_by_country")
    cursor.execute("DROP TABLE IF EXISTS bookings_by_country")
    cursor.execute("DROP TABLE IF EXISTS busiest_months")

    # Create tables
    cursor.execute('''
    CREATE TABLE analytics (
        metric TEXT PRIMARY KEY,
        value REAL
    )
    ''')

    cursor.execute('''
    CREATE TABLE revenue_by_year (
        year INTEGER PRIMARY KEY,
        revenue REAL
    )
    ''')

    cursor.execute('''
    CREATE TABLE revenue_by_month (
        month_year TEXT PRIMARY KEY,
        revenue REAL
    )
    ''')

    cursor.execute('''
    CREATE TABLE cancellations_by_country (
        country TEXT PRIMARY KEY,
        cancellation_count INTEGER
    )
    ''')

    cursor.execute('''
    CREATE TABLE bookings_by_country (
        country TEXT PRIMARY KEY,
        total_bookings INTEGER
    )
    ''')

    cursor.execute('''
    CREATE TABLE busiest_months (
        month_year TEXT PRIMARY KEY,
        booking_percentage REAL
    )
    ''')

    # Calculate general analytics (excluding canceled bookings)
    df_not_canceled = df[df["is_canceled"] == 0].copy()
    analytics = {
        "total_revenue": ((df.loc[df["is_canceled"] == 0, "adr"] * (df["stays_in_week_nights"] +
                       df["stays_in_weekend_nights"]) * (df["adults"] + df["children"] + df["babies"])).sum()),
        "cancellation_rate": df["is_canceled"].mean() * 100,
        "average_lead_time": df["lead_time"].mean(),
        "total_bookings": len(df),
        "average_daily_rate": df["adr"].mean(),
        "max_lead_time": df["lead_time"].max(),
    }

    analytics = {metric: to_python_type(value) for metric, value in analytics.items()}

    # Insert general analytics
    for metric, value in analytics.items():
        cursor.execute("INSERT INTO analytics (metric, value) VALUES (?, ?)", (metric, value))

    # Calculate revenue by year
    df_not_canceled.loc[:, "year"] = pd.to_datetime(df_not_canceled["arrival_date_year"].astype(str) + '-' + df_not_canceled["arrival_date_month"].str.capitalize() + '-01').dt.year
    revenue_by_year = df_not_canceled.groupby("year").apply(lambda x: (x["adr"] * (x["stays_in_week_nights"] + x["stays_in_weekend_nights"])).sum()).reset_index(name="revenue")

    for _, row in revenue_by_year.iterrows():
        cursor.execute("INSERT INTO revenue_by_year (year, revenue) VALUES (?, ?)", (to_python_type(row["year"]), to_python_type(row["revenue"])))

    # Calculate revenue by month
    df_not_canceled.loc[:, "month_year"] = pd.to_datetime(df_not_canceled["arrival_date_year"].astype(str) + '-' + df_not_canceled["arrival_date_month"].str.capitalize() + '-01').dt.strftime("%B %Y")
    revenue_by_month = df_not_canceled.groupby("month_year").apply(lambda x: (x["adr"] * (x["stays_in_week_nights"] + x["stays_in_weekend_nights"])).sum()).reset_index(name="revenue")

    for _, row in revenue_by_month.iterrows():
        cursor.execute("INSERT INTO revenue_by_month (month_year, revenue) VALUES (?, ?)", (row["month_year"], to_python_type(row["revenue"])))

    # Calculate cancellation count by country
    cancellations_by_country = df[df["is_canceled"] == 1]["country"].value_counts().reset_index()
    cancellations_by_country.columns = ["country", "cancellation_count"]

    for _, row in cancellations_by_country.iterrows():
        cursor.execute("INSERT INTO cancellations_by_country (country, cancellation_count) VALUES (?, ?)", (row["country"], to_python_type(row["cancellation_count"])))

    # Calculate total bookings by country
    bookings_by_country = df["country"].value_counts().reset_index()
    bookings_by_country.columns = ["country", "total_bookings"]

    for _, row in bookings_by_country.iterrows():
        cursor.execute("INSERT INTO bookings_by_country (country, total_bookings) VALUES (?, ?)", (row["country"], to_python_type(row["total_bookings"])))

    # Calculate most busy months by booking percentage
    monthly_bookings = df["arrival_date_month"].str.capitalize() + " " + df["arrival_date_year"].astype(str)
    busiest_months = monthly_bookings.value_counts(normalize=True).mul(100).reset_index()
    busiest_months.columns = ["month_year", "booking_percentage"]

    for _, row in busiest_months.iterrows():
        cursor.execute("INSERT INTO busiest_months (month_year, booking_percentage) VALUES (?, ?)", (row["month_year"], to_python_type(row["booking_percentage"])))

    conn.commit()
    conn.close()
    print("Analytics stored successfully.")

def query_llm(prompt):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 500,
            "temperature": 0.5,
            "top_p": 0.9
        }
    }
    response = requests.post(API_ENDPOINT, headers=headers, json=payload)

    if response.status_code == 200:
        return response.json()[0]["generated_text"]
    else:
        return f"Error: {response.status_code}, {response.json()}"
# Helper Functions


def extract_number_from_query(query):
    numbers = re.findall(r'\d+', query)
    if numbers:
        return int(numbers[0])

    try:
        words = re.findall(r'\b(one|two|three|four|five|six|seven|eight|nine|ten|'
                           r'eleven|twelve|thirteen|fourteen|fifteen|sixteen|'
                           r'seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|'
                           r'sixty|seventy|eighty|ninety|hundred|thousand)\b',
                           query, re.IGNORECASE)
        if words:
            return w2n.word_to_num(' '.join(words))
    except ValueError:
        pass

    return 5


def get_analytics():
    conn = sqlite3.connect("analytics.db")
    cursor = conn.cursor()
    cursor.execute("SELECT metric, value FROM analytics")
    analytics = {metric: value for metric, value in cursor.fetchall()}
    conn.close()
    return analytics


def get_top_cancellations(top_k):
    conn = sqlite3.connect("analytics.db")
    cursor = conn.cursor()

    query = f"""
    SELECT country, cancellation_count
    FROM cancellations_by_country
    ORDER BY cancellation_count DESC
    LIMIT {top_k}
    """

    cursor.execute(query)
    result = cursor.fetchall()
    conn.close()
    return result


def get_revenue_by_year(year):
    conn = sqlite3.connect("analytics.db")
    cursor = conn.cursor()

    query = """
    SELECT revenue
    FROM revenue_by_year
    WHERE year = ?
    """

    cursor.execute(query, (year,))
    result = cursor.fetchone()
    conn.close()
    return result[0] if result else None


def get_revenue_by_month(month, year):
    conn = sqlite3.connect("analytics.db")
    cursor = conn.cursor()

    month_year = f"{month} {year}"

    query = """
    SELECT revenue
    FROM revenue_by_month
    WHERE month_year = ?
    """

    cursor.execute(query, (month_year,))
    result = cursor.fetchone()
    conn.close()
    return result[0] if result else None


# Get total bookings by country
def get_bookings_by_country(country):
    conn = sqlite3.connect("analytics.db")
    cursor = conn.cursor()

    query = """
    SELECT total_bookings
    FROM bookings_by_country
    WHERE country = ?
    """

    cursor.execute(query, (country,))
    result = cursor.fetchone()
    conn.close()
    return result[0] if result else None


# Get top booked countries
def get_top_booked_countries(top_k):
    conn = sqlite3.connect("analytics.db")
    cursor = conn.cursor()

    query = f"""
    SELECT country, total_bookings
    FROM bookings_by_country
    ORDER BY total_bookings DESC
    LIMIT {top_k}
    """

    cursor.execute(query)
    result = cursor.fetchall()
    conn.close()
    return result


def search_bookings(query, top_k=5):
    try:
        # Encode the query to generate an embedding
        query_embedding = embedding_model.encode([query], convert_to_numpy=True)

        # Load FAISS index
        faiss_index = faiss.read_index("booking_faiss.index")

        # Perform FAISS search to find the closest matches
        distances, indices = faiss_index.search(query_embedding, top_k)

        # Check if any results were found
        if len(indices[0]) == 0 or all(idx == -1 for idx in indices[0]):
            return ["No relevant information found."]

        # Retrieve and return the corresponding booking texts from the DataFrame
        return [df.iloc[idx]['text'] for idx in indices[0] if idx < len(df)]

    except Exception as e:
        print(f"Error during FAISS search: {e}")
        return ["Error during search. Please try again."]


def get_busiest_months(top_k):
    conn = sqlite3.connect("analytics.db")
    cursor = conn.cursor()

    query = f"""
    SELECT month_year, booking_percentage
    FROM busiest_months
    ORDER BY booking_percentage DESC
    LIMIT {top_k}
    """

    cursor.execute(query)
    result = cursor.fetchall()
    conn.close()
    return result


def is_booking_related(query):
    """Check if the query is relevant to the booking analytics system."""
    keywords = [
        "revenue", "cancellation", "booking", "lead time", "hotel", "country", "busiest month",
        "most bookings", "location", "percentage", "average rate"
    ]
    return any(keyword in query.lower() for keyword in keywords)


# Initialize embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


def rag_pipeline(user_query):
    # Load FAISS index
    faiss_index = faiss.read_index("booking_faiss.index")

    print(f"User Query: {user_query}\n")

    analytics = get_analytics()

    year_match = re.search(r'(\d{4})', user_query)
    month_match = re.search(r'(January|February|March|April|May|June|July|August|September|October|November'
                            r'|December)', user_query, re.IGNORECASE)

    # Revenue-related queries
    if "revenue" in user_query.lower():
        if year_match and month_match:
            year = int(year_match.group(1))
            month = month_match.group(1).capitalize()
            revenue = get_revenue_by_month(month, year)
            return f"Response:\nThe total revenue for {month} {year} is {revenue:.2f}.\n" if revenue else \
                   f"Response:\nNo revenue data available for {month} {year}.\n"

        elif year_match:
            year = int(year_match.group(1))
            revenue = get_revenue_by_year(year)
            return f"Response:\nThe total revenue for {year} is {revenue:.2f}.\n" if revenue else \
                   f"Response:\nNo revenue data available for {year}.\n"

    # General analytics
    for metric, value in analytics.items():
        if metric.replace("_", " ") in user_query.lower():
            return f"Response:\nThe value of {metric.replace('_', ' ')} is {value:.2f}.\n"

    # Handle country-specific bookings query
    if "bookings from" in user_query.lower():
        country_match = re.search(r'from (\w+(?: \w+)*)', user_query, re.IGNORECASE)
        if country_match:
            country = country_match.group(1).title()
            bookings = get_bookings_by_country(country)
            return f"Response:\nTotal bookings from {country}: {bookings}.\n" if bookings else \
                   f"Response:\nNo bookings found from {country}.\n"

    # Handle most booked countries query
    if "most bookings" in user_query.lower():
        top_k = extract_number_from_query(user_query)
        most_booked = get_top_booked_countries(top_k)
        return "Response:\n" + "\n".join(f"{country}: {count} bookings" for country, count in most_booked) + "\n"

    if "busiest months" in user_query.lower():
        top_k = extract_number_from_query(user_query)
        busiest_months = get_busiest_months(top_k)
        return "Response:\n" + "\n".join(f"{month}: {percentage:.2f}%" for month, percentage in busiest_months) + "\n"

    # Cancellations
    if "cancellation" in user_query.lower():
        top_k = extract_number_from_query(user_query)
        cancellations = get_top_cancellations(top_k)
        if cancellations:
            return ("Response:\n" + "\n".join(f"{country}: {count} cancellations" for country, count in cancellations) +
                    "\n")

    # Default: FAISS search
    if is_booking_related(user_query):
        top_k = extract_number_from_query(user_query)
        search_results = search_bookings(user_query, top_k)
        if search_results:
            return "Response:\n" + "\n".join(search_results) + "\n"

    return "Response:\nQuery not recognized.\n"


# Flask app entry point
if __name__ == "__main__":
    # Generate and save embeddings
    generate_and_save_embeddings(df)

    # Store analytics in the database
    store_analytics(df)
