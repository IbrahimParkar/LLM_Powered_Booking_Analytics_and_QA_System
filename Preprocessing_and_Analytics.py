# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import warnings
# Suppress future warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Load the dataset
df = pd.read_csv(r"C:\Users\Ibrahim\Desktop\AI ML Task\hotel_bookings.csv")

# 1. Data Collection and preprocessing

# Display basic info about the dataset
print("Initial Dataset Info:")
print(df.info())
print("\nFirst 5 Rows:")
print(df.head())

# Display missing values before cleaning
print("Missing Values Before Cleaning:")
print(df.isnull().sum())

# Handle missing values
df['children'].fillna(0, inplace=True)  # Replace NaN with 0 in children
df['country'].fillna('unknown', inplace=True)  # Replace NaN with 'unknown' in country
df['agent'].fillna(0, inplace=True)  # Replace NaN with 0 in agent
df['company'].fillna(0, inplace=True)  # Replace NaN with 0 in company

# Verify if missing values are handled
print("Missing values after handling:")
print(df.isnull().sum())

# Ensure data consistency
# Convert reservation_status_date to datetime format
df['reservation_status_date'] = pd.to_datetime(df['reservation_status_date'], format='%d-%m-%y')

# Ensure lead_time, agent, and company are integers
df['lead_time'] = df['lead_time'].astype(int)
df['agent'] = df['agent'].astype(int)
df['company'] = df['company'].astype(int)

# Ensure adr (average daily rate) is a float (in case of inconsistencies)
df['adr'] = df['adr'].astype(float)

# Standardize text columns (strip whitespace and lowercase for consistency)
text_cols = ['hotel', 'arrival_date_month', 'meal', 'market_segment', 'distribution_channel',
             'reserved_room_type', 'assigned_room_type', 'deposit_type', 'customer_type',
             'reservation_status', 'country']

df[text_cols] = df[text_cols].apply(lambda x: x.str.strip().str.lower())

# Verify changes
print("Data types after ensuring consistency:")
print(df.dtypes)

# 2. Analytics & Reporting

# Revenue trends over time.


df['total_revenue'] = ((df.loc[df["is_canceled"] == 0, "adr"] * (df["stays_in_week_nights"] +
                        df["stays_in_weekend_nights"]) * (df["adults"] + df["children"] + df["babies"])).sum())
df['arrival_date_month'] = df['arrival_date_month'].str.lower()
month_order = ['january', 'february', 'march', 'april', 'may', 'june',
               'july', 'august', 'september', 'october', 'november', 'december']
df['arrival_date_month'] = pd.Categorical(df['arrival_date_month'], categories=month_order, ordered=True)
revenue_trend = df.groupby(['arrival_date_year', 'arrival_date_month'])['total_revenue'].sum().reset_index()

plt.figure(figsize=(12, 6))
sns.lineplot(data=revenue_trend, x='arrival_date_month', y='total_revenue', hue='arrival_date_year', marker='o')
plt.title('Revenue Trends Over Time')
plt.xlabel('Month')
plt.ylabel('Total Revenue')
plt.legend(title='Year')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Cancellation rate as percentage of total bookings
cancellation_rate = (df['is_canceled'].sum() / len(df)) * 100
print(f"Cancellation Rate: {cancellation_rate:.2f}%")

# Geographical distribution of users doing the bookings.
country_distribution = df['country'].value_counts().reset_index()
country_distribution.columns = ['country', 'bookings']

plt.figure(figsize=(12, 6))
sns.barplot(x='bookings', y='country', data=country_distribution.head(10), palette='viridis')
plt.title('Top 10 Countries by Number of Bookings')
plt.xlabel('Number of Bookings')
plt.ylabel('Country')
plt.tight_layout()
plt.show()

# Ensure all country codes are uppercase
df['country'] = df['country'].str.upper()

# Replace missing values with 'UNKNOWN'
df['country'].fillna('UNKNOWN', inplace=True)
geo_distribution = df['country'].value_counts().reset_index()
geo_distribution.columns = ['country', 'bookings']

fig = px.choropleth(geo_distribution,
                    locations='country',
                    locationmode='ISO-3',
                    color='bookings',
                    hover_name='country',
                    title='Geographical Distribution of Hotel Bookings',
                    color_continuous_scale=px.colors.sequential.Sunset,  # New color scheme
                    projection='natural earth')

fig.show()

# Booking Lead time distribution
plt.figure(figsize=(10, 6))
sns.histplot(df['lead_time'], bins=50, kde=True, color='royalblue')
plt.title('Booking Lead Time Distribution')
plt.xlabel('Lead Time (Days)')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# Average daily rate by hotel type
avg_adr_by_hotel = df.groupby('hotel')['adr'].mean().reset_index()
print("Average Daily Rate by Hotel Type:")
print(avg_adr_by_hotel)

plt.figure(figsize=(8, 5))
sns.barplot(x='hotel', y='adr', data=avg_adr_by_hotel, palette='muted')
plt.title('Average Daily Rate (ADR) by Hotel Type')
plt.xlabel('Hotel Type')
plt.ylabel('Average Daily Rate')
plt.tight_layout()
plt.show()

# Stay Duration Distribution
df['total_stay'] = df['stays_in_weekend_nights'] + df['stays_in_week_nights']

plt.figure(figsize=(10, 6))
sns.histplot(df['total_stay'], bins=30, kde=True, color='teal')
plt.title('Distribution of Total Stay Duration')
plt.xlabel('Total Stay (Nights)')
plt.ylabel('Frequency')
plt.xlim(0, 30)
plt.show()

# Most Busy Months

df['arrival_date_month'] = df['arrival_date_month'].str.lower()
month_order = ['january', 'february', 'march', 'april', 'may', 'june',
               'july', 'august', 'september', 'october', 'november', 'december']
df['arrival_date_month'] = pd.Categorical(df['arrival_date_month'], categories=month_order, ordered=True)
busy_months = df.groupby('arrival_date_month').size().reset_index(name='num_bookings')
busy_months = busy_months.sort_values('arrival_date_month')

plt.figure(figsize=(10, 10))
plt.pie(busy_months['num_bookings'], labels=busy_months['arrival_date_month'], autopct='%1.1f%%',
        startangle=140, colors=sns.color_palette('coolwarm', 12))
plt.title('Most Busy Months (Percentage of Bookings)')
plt.show()