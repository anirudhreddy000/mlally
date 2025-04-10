import requests
import json
schema_url = "http://127.0.0.1:8005/placement_schema"
prediction_url = "http://127.0.0.1:8000/send_data"


schema_response = requests.get(schema_url)

if schema_response.status_code == 200:
    schema = schema_response.json()
    print("Schema:", json.dumps(schema, indent=2))
else:
    print(f"Error fetching schema: {schema_response.status_code}")
    print(schema_response.text)
    exit(1)

data = {}
for field, properties in schema.items():
    if properties["type"] == "number":
        value = input(f"Enter value for {field} : ")
        data[field] = float(value) if value else (3.5 if field == "cgpa" else 85)
    elif properties["type"] == "integer":
        value = input(f"Enter value for {field}: ")
        data[field] = int(value) if value else 13
    elif properties["type"] == "string":
        value = input(f"Enter value for {field} : ")
        data[field] = value if value else "example"
    elif properties["type"] == "boolean":
        value = input(f"Enter value for {field} : ").lower()
        data[field] = value == 'true'

print("Request data:", json.dumps(data, indent=2))

json_data = json.dumps(data)

headers = {"Content-Type": "application/json"}
response = requests.post(prediction_url, headers=headers, data=json_data)

print("Response status code:", response.status_code)
print("Response headers:", response.headers)
print("Response text:", response.text)

if response.status_code == 200:
    print(response.json())
else:
    print(f"Error: {response.status_code}")
    print(response.text)
