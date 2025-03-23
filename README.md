# LLM_Powered_Booking_Analytics_and_QA_System
Here's the updated **README** with the mention of downloading the **FAISS index** file from Google Drive:

---

# Booking Analytics System

## Overview
This project implements a booking analytics and QA system using Python. It includes data preprocessing, analytics, and reporting on various key booking metrics. The system provides insights on revenue trends, cancellation rates, geographical distribution of users, and more.

## Getting Started

### Simple Guide

1. **Clone or Download the Repository**:
   - Clone the repository using:
     ```bash
     git clone https://github.com/your-username/repository-name.git
     ```
   - Or download the repository as a ZIP file and extract it to your preferred location.

2. **Install Dependencies**:
   - Install the required packages by running the following command in the project directory:
     ```bash
     pip install -r requirements.txt
     ```

3. **Download the FAISS Index**:
   - Since the `booking_faiss.index` file is too large to upload to GitHub, you can download it from [this Google Drive link](https://drive.google.com/file/d/1ahh4ub5kUiZnmz2ZN6RP77yuOI1xBVoX/view?usp=sharing).
   - After downloading, place the `booking_faiss.index` file in the project directory.

4. **Run `app.py`**:
   - Start the Flask web server by running:
     ```bash
     python app.py
     ```

5. **Test the API**:
   - After running `app.py`, you can test the API using `curl` in your terminal.
   - For `/health` (GET), run:
     ```bash
     curl http://127.0.0.1:5000/health
     ```
   - For `/analytics` (POST), run:
     ```bash
     curl -X POST http://127.0.0.1:5000/analytics
     ```
   - For `/ask` (POST), run:
     ```bash
     curl -X POST http://localhost:5000/ask -H "Content-Type: application/json" -d "{\"query\": \"What is the total revenue for July 2017?\"}"
     ```

6. **Sample Queries**:
   - The `Sample Queries.txt` file contains **20 sample queries** that you can copy and paste to test the `/ask` endpoint.

---

### Detailed Guide

1. **Clone or Download the Repository**:
   - To get started, **clone** the repository using Git:
     ```bash
     git clone https://github.com/your-username/repository-name.git
     ```
   - Alternatively, you can **download** the repository as a ZIP file and extract it to your desired location.

2. **Install Dependencies**:
   - Once you have the repository on your local machine, navigate to the project directory in your terminal.
   - Install the necessary dependencies using the following command:
     ```bash
     pip install -r requirements.txt
     ```
   - This will install all the packages required to run the project.

3. **Run `Preprocessing_and_Analytics.py`**:
   - This script handles data preprocessing and analytics. It directly downloads the dataset (`hotel_bookings.csv`) and processes the data to generate insights.
   - It will create the following analytics reports:
     - Revenue trends over time
     - Cancellation rate as a percentage of total bookings
     - Geographical distribution of users doing the bookings
     - Booking lead time distribution
     - Additional analytics (e.g., average lead time, ADR, maximum lead time)

4. **Hugging Face API Key and Endpoint Setup**:
   - Before running `RAG.py`, you'll need to configure your **Hugging Face API key** and **API endpoint**.
   
   - **Generate Hugging Face API Key**:
     - Visit [Hugging Face](https://huggingface.co/).
     - Sign in or create an account.
     - Go to **Settings** > **Access Tokens** and create a new token.
     - Copy the generated API token.

   - **Set up the API Key and Endpoint in `RAG.py`**:
     - In the `RAG.py` file, set your Hugging Face API key and endpoint as shown below:
     ```python
     import os
     
     # Set your Hugging Face API key here
     api_key = "your_hugging_face_api_key_here"
     api_endpoint = "https://api-inference.huggingface.co/models/Mistral-7B-Instruct-v0.3"
     
     # Optionally set the environment variable if required
     os.environ["HF_API_KEY"] = api_key
     ```

5. **Run `RAG.py`**:
   - After setting up the API key and endpoint, run `RAG.py` to generate the embeddings and store them in `booking_faiss.index`.
   - This file will also create an SQLite database `analytics.db` containing various analytics data, including:
   
   **Database Name**: `analytics.db`
   
   **Tables and Their Content**:
   - **analytics**: Stores metrics like total revenue, cancellation rate, average lead time, etc.
   - **revenue_by_year**: Total revenue for each year (excluding canceled bookings).
   - **revenue_by_month**: Total revenue for each month-year combination.
   - **cancellations_by_country**: Number of canceled bookings by country.
   - **bookings_by_country**: Total number of bookings by country.
   - **busiest_months**: Percentage of bookings for each month-year combination.

6. **Run `app.py`**:
   - After generating the embeddings and database, run the `app.py` file to start the Flask web server:
     ```bash
     python app.py
     ```
   - This will start a local server at `http://127.0.0.1:5000/`.

   - The server will expose the following API endpoints:
     - `/analytics` (POST): For fetching analytics data.
     - `/ask` (POST): For querying the system using the RAG pipeline.
     - `/health` (GET): For checking the system's health status.

7. **Test the API**:
   - Once the Flask server is running, you can test the API using `curl` from your terminal:

     - **For `/health` (GET)**:
       ```bash
       curl http://127.0.0.1:5000/health
       ```

     - **For `/analytics` (POST)**:
       ```bash
       curl -X POST http://127.0.0.1:5000/analytics
       ```

     - **For `/ask` (POST)**:
       ```bash
       curl -X POST http://localhost:5000/ask -H "Content-Type: application/json" -d "{\"query\": \"What is the total revenue for July 2017?\"}"
       ```

8. **Sample Queries**:
   - The `Sample Queries.txt` file contains **20 sample queries** that you can use to test the `/ask` endpoint. Copy and paste the queries into the terminal to check if the system returns the correct responses.

---

## Files
- `hotel_bookings.csv`: The dataset used for analysis.
- `Preprocessing_and_Analytics.py`: Contains data preprocessing and analytics implementations.
- `RAG.py`: Handles the retrieval-augmented generation (RAG) system, embedding creation, and database generation.
- `app.py`: Implements a Flask API for accessing analytics and QA system through a local server.
- `Sample Queries.txt`: Contains **20 sample queries** along with their answers to test the system.
- `booking_faiss.index`: The FAISS index storing embeddings (download from [Google Drive](https://drive.google.com/file/d/1ahh4ub5kUiZnmz2ZN6RP77yuOI1xBVoX/view?usp=sharing)).
- `analytics.db`: The SQLite database containing analytics tables.
- `requirements.txt`: A list of required dependencies for the project.
