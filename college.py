import requests
import zipfile
import io
import csv

url = "https://ed-public-download.app.cloud.gov/downloads/Most-Recent-Cohorts-Institution_04192023.zip"
output_filename = "Most-Recent-Cohorts-Institution_04192023.zip"

response = requests.get(url)

if response.status_code == 200:
    with open(output_filename, 'wb') as file:
        file.write(response.content)
    print(f"Downloaded {output_filename} successfully!")

    # Extract CSV files from the downloaded ZIP
    with zipfile.ZipFile(output_filename, 'r') as zip_ref:
        csv_filenames = [name for name in zip_ref.namelist() if name.lower().endswith('.csv')]
        for csv_filename in csv_filenames:
            with zip_ref.open(csv_filename) as csv_file:
                csv_content = csv_file.read().decode('utf-8')
                # Now you can process the CSV content using the csv module
                csv_reader = csv.reader(csv_content.splitlines())
                for row in csv_reader:
                    print(row)  # Replace this with your actual processing logic

else:
    print(f"Failed to download {output_filename}. Status code: {response.status_code}")
