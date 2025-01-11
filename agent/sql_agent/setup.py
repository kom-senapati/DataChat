import requests

url = "https://storage.googleapis.com/benchmarks-artifacts/chinook/Chinook.db"

def main():
    response = requests.get(url)
    
    if response.status_code == 200:
        with open(r"data/data.db", "wb") as file:
            file.write(response.content)
        print("File downloaded and saved as data.db")
    else:
        print(f"Failed to download the file. Status code: {response.status_code}")
