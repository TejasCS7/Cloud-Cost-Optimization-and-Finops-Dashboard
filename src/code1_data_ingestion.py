import csv
from datetime import datetime
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Service category mapping
SERVICE_CATEGORIES = {
    'Compute Engine': 'Compute',
    'Cloud Storage': 'Storage',
    'BigQuery': 'Analytics',
    # Add more mappings as needed
}

# Overprovisioning thresholds (configurable)
OVERPROVISIONING_THRESHOLDS = {
    'cpu_util': 30,
    'mem_util': 30,
    'cost': 100
}

def clean_cost_data(data, headers):
    """
    Clean and preprocess GCP billing data with robust error handling
    """
    # Initialize output data with new columns
    new_headers = headers + [
        'Usage Duration (Hours)', 'Cost Per Hour', 'Month', 'Day',
        'Service Category', 'Network Inbound Data (GB)', 'Network Outbound Data (GB)',
        'Cost per GB Transferred', 'Is Overprovisioned'
    ]
    processed_data = []

    for row in data:
        new_row = dict(row)  # Copy original row

        # Convert date columns to datetime with error handling
        try:
            start_date = datetime.strptime(row['Usage Start Date'], '%d-%m-%Y %H:%M')
            end_date = datetime.strptime(row['Usage End Date'], '%d-%m-%Y %H:%M')
        except Exception as e:
            logging.warning(f"Date conversion failed for row {row['Resource ID']} - {str(e)}. Using None.")
            start_date = None
            end_date = None

        # Calculate usage duration safely
        if start_date and end_date:
            duration_seconds = (end_date - start_date).total_seconds()
            usage_duration_hours = duration_seconds / 3600
        else:
            usage_duration_hours = 0
        new_row['Usage Duration (Hours)'] = usage_duration_hours

        # Calculate Cost Per Hour with division by zero protection
        unrounded_cost = float(row['Unrounded Cost ($)']) if row['Unrounded Cost ($)'] else 0
        cost_per_hour = unrounded_cost / usage_duration_hours if usage_duration_hours > 0 else 0
        new_row['Cost Per Hour'] = cost_per_hour

        # Extract time-based features
        if start_date:
            new_row['Month'] = start_date.strftime('%Y-%m')
            new_row['Day'] = start_date.strftime('%Y-%m-%d')
        else:
            new_row['Month'] = ''
            new_row['Day'] = ''

        # Clean Resource ID
        new_row['Resource ID'] = row['Resource ID'].strip()

        # Extract service category using mapping
        service_name = row['Service Name']
        new_row['Service Category'] = SERVICE_CATEGORIES.get(service_name, 'Unknown')

        # Normalize utilization metrics
        try:
            cpu_util = float(row['CPU Utilization (%)']) if row['CPU Utilization (%)'] else None
        except ValueError:
            cpu_util = None
        try:
            mem_util = float(row['Memory Utilization (%)']) if row['Memory Utilization (%)'] else None
        except ValueError:
            mem_util = None
        new_row['CPU Utilization (%)'] = cpu_util
        new_row['Memory Utilization (%)'] = mem_util

        # Convert network data to GB
        inbound_bytes = float(row['Network Inbound Data (Bytes)']) if row['Network Inbound Data (Bytes)'] else 0
        outbound_bytes = float(row['Network Outbound Data (Bytes)']) if row['Network Outbound Data (Bytes)'] else 0
        inbound_gb = inbound_bytes / (1024 ** 3)
        outbound_gb = outbound_bytes / (1024 ** 3)
        new_row['Network Inbound Data (GB)'] = inbound_gb
        new_row['Network Outbound Data (GB)'] = outbound_gb

        # Calculate Cost per GB Transferred with division by zero protection
        total_network_gb = inbound_gb + outbound_gb
        cost_per_gb = unrounded_cost / total_network_gb if total_network_gb > 0 else 0
        new_row['Cost per GB Transferred'] = cost_per_gb

        # Flag potentially over-provisioned resources
        if cpu_util is None or mem_util is None:
            new_row['Is Overprovisioned'] = 'Unknown'
        else:
            is_overprovisioned = (
                cpu_util < OVERPROVISIONING_THRESHOLDS['cpu_util'] and
                mem_util < OVERPROVISIONING_THRESHOLDS['mem_util'] and
                unrounded_cost > OVERPROVISIONING_THRESHOLDS['cost']
            )
            new_row['Is Overprovisioned'] = is_overprovisioned

        processed_data.append(new_row)

    return processed_data, new_headers

def ingest_billing_data(file_path):
    """
    Ingest GCP billing data from CSV file with validation
    """
    # Read CSV file
    try:
        with open(file_path, 'r') as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames
            data = [row for row in reader]
    except Exception as e:
        raise ValueError(f"Failed to read CSV file: {str(e)}")

    # Validate expected columns
    expected_columns = [
        'Resource ID', 'Service Name', 'Usage Quantity', 'Usage Unit',
        'Region/Zone', 'CPU Utilization (%)', 'Memory Utilization (%)',
        'Network Inbound Data (Bytes)', 'Network Outbound Data (Bytes)',
        'Usage Start Date', 'Usage End Date', 'Cost per Quantity ($)',
        'Unrounded Cost ($)', 'Rounded Cost ($)', 'Total Cost (INR)'
    ]
    missing_columns = [col for col in expected_columns if col not in headers]
    if missing_columns:
        raise ValueError(f"Missing expected columns: {missing_columns}")

    # Clean data
    clean_data, new_headers = clean_cost_data(data, headers)

    # Remove duplicates based on Resource ID and Usage Start Date
    seen = set()
    unique_data = []
    for row in clean_data:
        key = (row['Resource ID'], row['Usage Start Date'])
        if key not in seen:
            seen.add(key)
            unique_data.append(row)

    logging.info(f"Ingested {len(unique_data)} billing records")
    return unique_data, new_headers

def main(file_path='/Users/tejasg/Documents/MyProjects/Cloud-FinOps Dashboard/Cloud Cost Optimization & FinOps Dashboard/Data/gcp_billing_datasets.csv'):
    """
    Main function to ingest and preprocess data
    """
    if not os.path.exists(file_path):
        logging.error(f"File {file_path} not found")
        return

    try:
        # Ingest and clean data
        clean_data, headers = ingest_billing_data(file_path)
        logging.info(f"Data preprocessing complete. Records: {len(clean_data)}")

        # Save the processed data
        processed_path = '/Users/tejasg/Documents/MyProjects/Cloud-FinOps Dashboard/Cloud Cost Optimization & FinOps Dashboard/Data/processed_billing_data.csv'
        with open(processed_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(clean_data)
        logging.info(f"Processed data saved to {processed_path}")
    except Exception as e:
        logging.error(f"Error during data ingestion: {str(e)}")

if __name__ == "__main__":
    main()
