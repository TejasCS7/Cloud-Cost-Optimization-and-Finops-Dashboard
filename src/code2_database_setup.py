import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import csv
from datetime import datetime, timedelta
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_database(dbname, user, password, host='localhost', port='5432'):
    """
    Create PostgreSQL database if it doesn't exist
    """
    try:
        conn = psycopg2.connect(user=user, password=password, host=host, port=port)
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()
        
        cursor.execute("SELECT 1 FROM pg_catalog.pg_database WHERE datname = %s", (dbname,))
        exists = cursor.fetchone()
        
        if not exists:
            cursor.execute(f'CREATE DATABASE {dbname}')
            logging.info(f"Database {dbname} created successfully")
        else:
            logging.info(f"Database {dbname} already exists")
        
        cursor.close()
        conn.close()
    except Exception as e:
        logging.error(f"Error creating database: {str(e)}")
        raise

def setup_tables(conn):
    """
    Create necessary tables in the database
    """
    create_tables_sql = """
    -- Dimension tables
    CREATE TABLE IF NOT EXISTS dim_services (
        service_id SERIAL PRIMARY KEY,
        service_name VARCHAR(255) NOT NULL,
        service_category VARCHAR(100) NOT NULL,
        UNIQUE(service_name)
    );

    CREATE TABLE IF NOT EXISTS dim_regions (
        region_id SERIAL PRIMARY KEY,
        region_name VARCHAR(100) NOT NULL,
        UNIQUE(region_name)
    );

    CREATE TABLE IF NOT EXISTS dim_resources (
        resource_id VARCHAR(255) PRIMARY KEY,
        first_seen_date DATE NOT NULL,
        last_seen_date DATE NOT NULL
    );

    CREATE TABLE IF NOT EXISTS dim_dates (
        date_id DATE PRIMARY KEY,
        day INTEGER NOT NULL,
        month INTEGER NOT NULL,
        year INTEGER NOT NULL,
        quarter INTEGER NOT NULL,
        day_of_week INTEGER NOT NULL,
        month_name VARCHAR(20) NOT NULL
    );

    -- Fact table with nullable fields
    CREATE TABLE IF NOT EXISTS fact_billing (
        billing_id SERIAL PRIMARY KEY,
        resource_id VARCHAR(255) REFERENCES dim_resources(resource_id),
        service_id INTEGER REFERENCES dim_services(service_id),
        region_id INTEGER REFERENCES dim_regions(region_id),
        usage_start_date DATE REFERENCES dim_dates(date_id),
        usage_end_date DATE REFERENCES dim_dates(date_id),
        usage_quantity FLOAT NOT NULL,
        usage_unit VARCHAR(50) NOT NULL,
        cpu_utilization FLOAT,
        memory_utilization FLOAT,
        network_inbound_gb FLOAT,
        network_outbound_gb FLOAT,
        cost_per_unit FLOAT NOT NULL,
        unrounded_cost FLOAT NOT NULL,
        rounded_cost FLOAT NOT NULL,
        total_cost_inr FLOAT NOT NULL,
        usage_duration_hours FLOAT NOT NULL,
        cost_per_hour FLOAT,
        is_overprovisioned BOOLEAN,
        cost_per_gb_transferred FLOAT
    );

    -- Optimization recommendations
    CREATE TABLE IF NOT EXISTS optimization_recommendations (
        recommendation_id SERIAL PRIMARY KEY,
        resource_id VARCHAR(255) REFERENCES dim_resources(resource_id),
        recommendation_type VARCHAR(100) NOT NULL,
        recommendation_description TEXT NOT NULL,
        potential_savings FLOAT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        implemented BOOLEAN DEFAULT FALSE
    );

    -- Historical cost tracking
    CREATE TABLE IF NOT EXISTS historical_costs (
        history_id SERIAL PRIMARY KEY,
        month VARCHAR(7) NOT NULL,
        service_id INTEGER REFERENCES dim_services(service_id),
        region_id INTEGER REFERENCES dim_regions(region_id),
        total_cost FLOAT NOT NULL,
        average_daily_cost FLOAT NOT NULL,
        cost_trend FLOAT NOT NULL
    );
    """
    
    try:
        cursor = conn.cursor()
        cursor.execute(create_tables_sql)
        conn.commit()
        cursor.close()
        logging.info("Database tables created successfully")
    except Exception as e:
        conn.rollback()
        logging.error(f"Error creating database tables: {str(e)}")
        raise
    finally:
        if 'cursor' in locals():
            cursor.close()

def populate_date_dimension(conn, start_date, end_date):
    """
    Populate the date dimension table
    """
    try:
        cursor = conn.cursor()
        current_date = start_date
        while current_date <= end_date:
            quarter = (current_date.month - 1) // 3 + 1
            day_of_week = current_date.weekday()
            month_name = current_date.strftime('%B')
            
            cursor.execute("""
                INSERT INTO dim_dates (date_id, day, month, yearplural, day_of_week, month_name)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (date_id) DO NOTHING
            """, (
                current_date,
                current_date.day,
                current_date.month,
                current_date.year,
                quarter,
                day_of_week,
                month_name
            ))
            current_date += timedelta(days=1)
        
        conn.commit()
        cursor.close()
        logging.info(f"Date dimension populated with records from {start_date} to {end_date}")
    except Exception as e:
        conn.rollback()
        logging.error(f"Error populating date dimension: {str(e)}")
        raise
    finally:
        if 'cursor' in locals():
            cursor.close()

def load_data_to_database(conn, rows):
    """
    Load data from CSV rows into the database
    """
    try:
        cursor = conn.cursor()
        service_map = {}
        region_map = {}
        resource_dates = {}
        monthly_costs = {}
        row_count = 0
        
        for row in rows:
            resource_id = row.get('Resource ID', '')
            if not resource_id:
                logging.warning(f"Missing resource ID in row {row_count + 1}. Skipping row.")
                continue
            
            try:
                start_date = datetime.strptime(row.get('Usage Start Date', ''), '%d-%m-%Y %H:%M').date()
                end_date = datetime.strptime(row.get('Usage End Date', ''), '%d-%m-%Y %H:%M').date()
            except ValueError as e:
                logging.warning(f"Error parsing dates for resource {resource_id} in row {row_count + 1}: {str(e)}. Skipping row.")
                continue
            
            cursor.execute("""
                SELECT COUNT(*) FROM fact_billing 
                WHERE resource_id = %s AND usage_start_date = %s AND usage_end_date = %s
            """, (resource_id, start_date, end_date))
            if cursor.fetchone()[0] > 0:
                logging.info(f"Skipping duplicate entry for {resource_id} on {start_date}")
                continue
            
            if resource_id not in resource_dates:
                resource_dates[resource_id] = {'first': start_date, 'last': end_date}
            else:
                resource_dates[resource_id]['first'] = min(resource_dates[resource_id]['first'], start_date)
                resource_dates[resource_id]['last'] = max(resource_dates[resource_id]['last'], end_date)
            
            row_count += 1
            
            # Populate dim_services
            service_name = row.get('Service Name', '')
            service_category = row.get('Service Category', '')
            
            if not service_name:
                logging.warning(f"Missing service name in row {row_count}. Skipping row.")
                continue
            
            if service_name not in service_map:
                cursor.execute("""
                    INSERT INTO dim_services (service_name, service_category)
                    VALUES (%s, %s)
                    ON CONFLICT (service_name) DO NOTHING
                    RETURNING service_id
                """, (service_name, service_category))
                result = cursor.fetchone()
                if result:
                    service_map[service_name] = result[0]
                else:
                    cursor.execute("SELECT service_id FROM dim_services WHERE service_name = %s", (service_name,))
                    service_map[service_name] = cursor.fetchone()[0]
            
            # Populate dim_regions
            region_name = row.get('Region/Zone', '')
            
            if not region_name:
                logging.warning(f"Missing region name in row {row_count}. Skipping row.")
                continue
            
            if region_name not in region_map:
                cursor.execute("""
                    INSERT INTO dim_regions (region_name)
                    VALUES (%s)
                    ON CONFLICT (region_name) DO NOTHING
                    RETURNING region_id
                """, (region_name,))
                result = cursor.fetchone()
                if result:
                    region_map[region_name] = result[0]
                else:
                    cursor.execute("SELECT region_id FROM dim_regions WHERE region_name = %s", (region_name,))
                    region_map[region_name] = cursor.fetchone()[0]
            
            # Insert fact_billing
            try:
                usage_quantity = float(row.get('Usage Quantity', 0))
                usage_unit = row.get('Usage Unit', '')
                cpu_utilization = float(row.get('CPU Utilization (%)', 0)) if row.get('CPU Utilization (%)') else None
                memory_utilization = float(row.get('Memory Utilization (%)', 0)) if row.get('Memory Utilization (%)') else None
                network_inbound = float(row.get('Network Inbound Data (GB)', 0))
                network_outbound = float(row.get('Network Outbound Data (GB)', 0))
                cost_per_unit = float(row.get('Cost per Quantity ($)', 0))
                unrounded_cost = float(row.get('Unrounded Cost ($)', 0))
                rounded_cost = float(row.get('Rounded Cost ($)', 0))
                total_cost_inr = float(row.get('Total Cost (INR)', 0))
                usage_duration = float(row.get('Usage Duration (Hours)', 0))
                cost_per_hour = float(row.get('Cost Per Hour', 0)) if row.get('Cost Per Hour') else None
                is_overprovisioned_value = row.get('Is Overprovisioned', '')
                is_overprovisioned = None if is_overprovisioned_value == 'Unknown' else is_overprovisioned_value.lower() == 'true'
                cost_per_gb = float(row.get('Cost per GB Transferred', 0)) if row.get('Cost per GB Transferred') else None
                
                cursor.execute("""
                    INSERT INTO fact_billing (
                        resource_id, service_id, region_id, usage_start_date, usage_end_date,
                        usage_quantity, usage_unit, cpu_utilization, memory_utilization,
                        network_inbound_gb, network_outbound_gb, cost_per_unit, unrounded_cost,
                        rounded_cost, total_cost_inr, usage_duration_hours, cost_per_hour,
                        is_overprovisioned, cost_per_gb_transferred
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    resource_id, service_map[service_name], region_map[region_name], start_date, end_date,
                    usage_quantity, usage_unit, cpu_utilization, memory_utilization,
                    network_inbound, network_outbound, cost_per_unit, unrounded_cost,
                    rounded_cost, total_cost_inr, usage_duration, cost_per_hour,
                    is_overprovisioned, cost_per_gb
                ))
            except ValueError as e:
                logging.warning(f"Error converting numeric values for resource {resource_id} in row {row_count}: {str(e)}. Skipping row.")
                continue
            except KeyError as e:
                logging.warning(f"Missing field in row {row_count}: {str(e)}. Skipping row.")
                continue
            
            # Track historical costs
            month = row.get('Month', '')
            day = row.get('Day', '')
            
            if month and day:
                key = (month, service_name, region_name)
                if key not in monthly_costs:
                    monthly_costs[key] = {'cost': 0.0, 'days': set()}
                monthly_costs[key]['cost'] += unrounded_cost
                monthly_costs[key]['days'].add(day)
        
        # Populate dim_resources
        for resource_id, dates in resource_dates.items():
            cursor.execute("""
                INSERT INTO dim_resources (resource_id, first_seen_date, last_seen_date)
                VALUES (%s, %s, %s)
                ON CONFLICT (resource_id) DO UPDATE
                SET first_seen_date = LEAST(dim_resources.first_seen_date, EXCLUDED.first_seen_date),
                    last_seen_date = GREATEST(dim_resources.last_seen_date, EXCLUDED.last_seen_date)
            """, (resource_id, dates['first'], dates['last']))
        
        # Populate historical_costs with trend calculation
        sorted_monthly_costs = sorted(monthly_costs.items(), key=lambda x: x[0][0])
        prev_costs = {}
        prev_month = None
        
        for (month, service_name, region_name), data in sorted_monthly_costs:
            if prev_month and month != (datetime.strptime(prev_month, '%Y-%m') + timedelta(days=32)).strftime('%Y-%m'):
                logging.warning(f"Non-consecutive months detected: {prev_month} to {month}")
            prev_month = month
            
            avg_daily_cost = data['cost'] / len(data['days']) if len(data['days']) > 0 else 0.0
            service_region_key = (service_name, region_name)
            
            if service_region_key in prev_costs:
                prev_cost = prev_costs[service_region_key]
                cost_trend = ((data['cost'] - prev_cost) / prev_cost * 100) if prev_cost > 0 else 0.0
            else:
                cost_trend = 0.0
            
            cursor.execute("""
                INSERT INTO historical_costs (month, service_id, region_id, total_cost, average_daily_cost, cost_trend)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (month, service_map[service_name], region_map[region_name], data['cost'], avg_daily_cost, cost_trend))
            
            prev_costs[service_region_key] = data['cost']
        
        conn.commit()
        logging.info(f"Data loaded successfully to database. Processed {row_count} rows.")
    except Exception as e:
        conn.rollback()
        logging.error(f"Error loading data to database: {str(e)}")
        raise
    finally:
        if 'cursor' in locals():
            cursor.close()

def main():
    """
    Main function to set up database and load data
    """
    db_config = {
        'dbname': 'cloud_finops',
        'user': 'postgres',
        'password': '64823',
        'host': 'localhost',
        'port': '5432'
    }
    csv_file = '/Users/tejasg/Documents/MyProjects/Cloud-FinOps Dashboard/Cloud Cost Optimization & FinOps Dashboard/Data/processed_billing_data.csv'
    
    try:
        # Create database
        create_database(**db_config)
        
        # Connect to the database
        conn = psycopg2.connect(**db_config)
        
        # Setup tables
        setup_tables(conn)
        
        # Read CSV and determine date range
        rows = []
        min_date = None
        max_date = None
        
        if not os.path.exists(csv_file):
            logging.error(f"CSV file not found: {csv_file}")
            return
            
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
                try:
                    start_date = datetime.strptime(row.get('Usage Start Date', ''), '%d-%m-%Y %H:%M').date()
                    end_date = datetime.strptime(row.get('Usage End Date', ''), '%d-%m-%Y %H:%M').date()
                    min_date = min(min_date, start_date) if min_date else start_date
                    max_date = max(max_date, end_date) if max_date else end_date
                except ValueError as e:
                    logging.warning(f"Error parsing dates in row: {row.get('Resource ID', 'unknown')}. Skipping date range check for this row.")
                    continue
        
        if min_date is None or max_date is None:
            logging.error("Could not determine valid date range from CSV data.")
            return
        
        # Populate date dimension
        populate_date_dimension(conn, min_date, max_date)
        
        # Load data
        load_data_to_database(conn, rows)
        
        logging.info("Database setup and data loading completed successfully")
        
    except Exception as e:
        logging.error(f"Error during database operations: {str(e)}")
        return
    finally:
        if 'conn' in locals():
            conn.close()
            logging.info("Database connection closed")

if __name__ == "__main__":
    main()
