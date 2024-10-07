import requests
import zipfile
import os
import csv

# URL of the ZIP file
url = "https://ed-public-download.app.cloud.gov/downloads/Most-Recent-Cohorts-Institution_04192023.zip"
output_filename = "Most-Recent-Cohorts-Institution_04192023.zip"

# Destination directory for saving extracted CSV files
extracted_directory = "csv_files"

response = requests.get(url)

if response.status_code == 200:
    # Create the directory if it doesn't exist
    os.makedirs(extracted_directory, exist_ok=True)

    # Save the downloaded ZIP file
    with open(output_filename, 'wb') as file:
        file.write(response.content)
    print(f"Downloaded {output_filename} successfully!")

    # Extract CSV files from the downloaded ZIP
    with zipfile.ZipFile(output_filename, 'r') as zip_ref:
        csv_filenames = [name for name in zip_ref.namelist() if name.lower().endswith('.csv')]
        for csv_filename in csv_filenames:
            with zip_ref.open(csv_filename) as csv_file:
                csv_content = csv_file.read().decode('utf-8')
                # Save the extracted CSV content to the specified directory
                extracted_csv_path = os.path.join(extracted_directory, csv_filename)
                with open(extracted_csv_path, 'w', encoding='utf-8') as extracted_csv_file:
                    extracted_csv_file.write(csv_content)
                print(f"Extracted and saved {csv_filename} successfully!")

                with open(extracted_csv_path, 'r', encoding='utf-8') as csv_file:
                    csv_reader = csv.DictReader(csv_file)
                    for row in csv_reader:
                        if row['STABBR'] == 'PA':
                            enrollment = row['UGDS']
                            if enrollment != 'NULL' and int(enrollment) < 5000:
                                sat = row['SAT_AVG']
                                if sat != 'NULL' and int(sat) > 1200:
                                    print(row['INSTNM'])

else:
    print(f"Failed to download {output_filename}. Status code: {response.status_code}")
