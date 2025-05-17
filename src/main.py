import os
import sys
import logging
from datetime import datetime
from code1_data_ingestion import main as ingest_data
from code2_database_setup import main as setup_database
from code3_cost_analysis import CloudCostOptimizer, main as analyze_costs
from code4_finops_dashboard import app

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('finops_workflow.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class FinOpsWorkflow:
    """Orchestrates the Cloud Cost Optimization & FinOps Dashboard workflow."""
    
    def __init__(self, input_file, db_config):
        self.input_file = input_file
        self.db_config = db_config
        self.processed_file = os.path.join(os.path.dirname(input_file), 'processed_billing_data.csv')
    
    def run_data_ingestion(self):
        """Step 1: Ingest and preprocess billing data."""
        logger.info("Starting data ingestion...")
        try:
            clean_data, headers = ingest_data(self.input_file)
            if clean_data is None:
                raise ValueError("Data ingestion failed - no data returned")
            logger.info(f"Data ingestion completed successfully. Processed {len(clean_data)} rows")
            return clean_data, headers
        except Exception as e:
            logger.error(f"Data ingestion error: {str(e)}", exc_info=True)
            raise
    
    def run_database_setup(self, csv_file):
        """Step 2: Set up database and load data."""
        logger.info("Starting database setup...")
        try:
            # Set environment variables for both CSV file and DB config
            os.environ['CSV_FILE_PATH'] = csv_file
            os.environ['DB_HOST'] = self.db_config['host']
            os.environ['DB_PORT'] = self.db_config['port']
            os.environ['DB_NAME'] = self.db_config['dbname']
            os.environ['DB_USER'] = self.db_config['user']
            os.environ['DB_PASSWORD'] = self.db_config['password']
            
            # Call setup_database with no parameters
            setup_database()
            logger.info("Database setup completed successfully.")
        except Exception as e:
            logger.error(f"Database setup error: {str(e)}", exc_info=True)
            raise
    
    def run_cost_analysis(self):
        """Step 3: Perform cost optimization analysis."""
        logger.info("Starting cost analysis...")
        try:
            optimizer = CloudCostOptimizer(self.db_config)
            
            # Initialize database tables for new features
            optimizer.setup_recommendation_tracking()
            optimizer._create_alerts_table()
            
            recommendations, summary = analyze_costs()
            logger.info(f"Generated {len(recommendations)} optimization recommendations")
            return recommendations, summary
        except Exception as e:
            logger.error(f"Cost analysis error: {str(e)}", exc_info=True)
            raise
    
    def run_dashboard(self):
        """Step 4: Launch the FinOps dashboard."""
        logger.info("Launching FinOps dashboard...")
        try:
            # Set the DB config for the dashboard
            app.config['DB_CONFIG'] = self.db_config
            app.run(host='0.0.0.0', port=8051, debug=False, use_reloader=False)
        except Exception as e:
            logger.error(f"Dashboard launch error: {str(e)}", exc_info=True)
            raise
    
    def execute(self):
        """Execute the full workflow."""
        start_time = datetime.now()
        logger.info("Starting Cloud FinOps workflow...")
        
        # Step 1: Data Ingestion
        clean_data, headers = self.run_data_ingestion()
        
        # Step 2: Database Setup with processed data
        self.run_database_setup(self.processed_file)
        
        # Step 3: Cost Analysis
        recommendations, summary = self.run_cost_analysis()
        
        # Log summary information
        if summary:
            logger.info(f"Potential savings: ${summary.get('potential_savings', 0):.2f}")
            logger.info(f"Cost trend: {summary.get('trend_percentage', 0):.1f}%")
        
        # Step 4: Dashboard (runs indefinitely until stopped)
        self.run_dashboard()
        
        end_time = datetime.now()
        logger.info(f"Workflow completed in {end_time - start_time}")

def main():
    # Configuration - should match all other files
    input_file = os.path.join('Data', 'gcp_billing_datasets.csv')
    db_config = {
        'dbname': 'cloud_finops',
        'user': 'postgres',
        'password': '64823',  # In production, use environment variables
        'host': 'localhost',
        'port': '5432'
    }
    
    # Ensure Data directory exists
    os.makedirs('Data', exist_ok=True)
    
    # Validate input file exists
    if not os.path.exists(input_file):
        logger.error(f"Input file not found: {input_file}")
        sys.exit(1)
    
    # Run workflow
    workflow = FinOpsWorkflow(input_file, db_config)
    try:
        workflow.execute()
    except Exception as e:
        logger.error(f"Fatal error in workflow execution: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
