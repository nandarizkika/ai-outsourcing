from google.cloud import bigquery
import json

client = bigquery.Client(project="ledger-fcc1e")

# Check the schema of query_history table
query = """
SELECT table_name, column_name, data_type
FROM `ledger-fcc1e.data_documentation.INFORMATION_SCHEMA.COLUMNS`
WHERE table_name = 'query_history'
"""

print("Schema of query_history table:")
result = client.query(query).result()
for row in result:
    print(f"  {row.column_name}: {row.data_type}")

# Get sample queries
print("\n\nSample queries from query_history:")
sample_query = """
SELECT * FROM `ledger-fcc1e.data_documentation.query_history`
LIMIT 5
"""
result = client.query(sample_query).result()
for row in result:
    print(f"\nQuery ID: {row.get('query_id', 'N/A')}")
    print(f"SQL: {row.get('sql', 'N/A')[:200]}...")
    print(f"Status: {row.get('status', 'N/A')}")
    print(f"Tables used: {row.get('tables_used', 'N/A')}")
