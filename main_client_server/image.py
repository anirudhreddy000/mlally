import requests
import zipfile

server_url = "http://127.0.0.1:8005/optimize_and_train/"

zip_file_path = r"G:\My Drive\my_desk_files\csv_files\train.zip"

try:
  with open(zip_file_path, 'rb') as f:
    zip_data = f.read()

  files = {'train_file': (zip_file_path, zip_data, 'application/zip')}

  response = requests.post(server_url, files=files)

  if response.status_code == 200:
    print("File uploaded and processed successfully!")
    print(response.json())  
  else:
    print(f"Error uploading file: {response.status_code}")
    print(f"Response: {response.text}")

except FileNotFoundError as e:
  print(f"Error: ZIP file not found at {zip_file_path}")
except Exception as e:
  print(f"An unexpected error occurred: {str(e)}")
