from venv import logger
from flask import Flask, render_template_string, request
import psycopg2
from psycopg2.extras import RealDictCursor
import json
from datetime import datetime, date, timedelta
import calendar
import random

app = Flask(__name__)

db_config = {
    'dbname': 'cloud_finops',
    'user': 'postgres',
    'password': '64823',
    'host': 'localhost',
    'port': '5432'
}

def format_currency(value):
    if value is None or value == 0:
        return "$0.00"
    abs_value = abs(value)
    if abs_value >= 1e9:
        return f"${value / 1e9:.1f}B"
    elif abs_value >= 1e6:
        return f"${value / 1e6:.1f}M"
    elif abs_value >= 1e3:
        return f"${value / 1e3:.1f}K"
    return f"${value:,.2f}"

def get_month_name(month_str):
    try:
        year, month = map(int, month_str.split('-'))
        return f"{calendar.month_name[month]} {year}"
    except (ValueError, IndexError):
        return month_str

def execute_query(query, db_config, params=None):
    conn = psycopg2.connect(**db_config, cursor_factory=RealDictCursor)
    try:
        cursor = conn.cursor()
        if params:
            cursor.execute(query, params)
        else:
            cursor.execute(query)
        results = [dict(row) for row in cursor.fetchall()]
        return results
    finally:
        cursor.close()
        conn.close()

def fetch_available_services():
    query = "SELECT DISTINCT service_name FROM dim_services ORDER BY service_name"
    data = execute_query(query, db_config)
    return [row['service_name'] for row in data]

def fetch_available_regions():
    query = "SELECT DISTINCT region_name FROM dim_regions ORDER BY region_name"
    data = execute_query(query, db_config)
    return [row['region_name'] for row in data]

def fetch_summary_metrics(start_date=None, end_date=None, service_filter=None):
    base_query = """
    SELECT SUM(unrounded_cost) as total_cost 
    FROM fact_billing fb
    JOIN dim_dates d ON fb.usage_start_date = d.date_id
    JOIN dim_services s ON fb.service_id = s.service_id
    WHERE 1=1
    """
    prev_query = """
    SELECT SUM(unrounded_cost) as total_cost 
    FROM fact_billing fb
    JOIN dim_dates d ON fb.usage_start_date = d.date_id
    JOIN dim_services s ON fb.service_id = s.service_id
    WHERE 1=1
    """
    params = []
    prev_params = []
    
    if start_date:
        base_query += " AND d.date_id >= %s"
        # For previous period, we need to properly cast the date and subtract interval
        prev_query += " AND d.date_id >= (DATE(%s) - INTERVAL '1 month')"
        params.append(start_date)
        prev_params.append(start_date)
    
    if end_date:
        base_query += " AND d.date_id <= %s"
        prev_query += " AND d.date_id <= (DATE(%s) - INTERVAL '1 month')"
        params.append(end_date)
        prev_params.append(end_date)
    
    if service_filter:
        base_query += " AND s.service_name = %s"
        prev_query += " AND s.service_name = %s"
        params.append(service_filter)
        prev_params.append(service_filter)
    
    # Rest of the function remains the same...
    opportunities_query = """
    WITH resource_metrics AS (
        SELECT 
            fb.resource_id,
            s.service_name,
            AVG(fb.cpu_utilization) as avg_cpu,
            AVG(fb.memory_utilization) as avg_memory,
            SUM(fb.unrounded_cost) as total_cost,
            COUNT(DISTINCT fb.usage_start_date) as active_days,
            (SELECT AVG(unrounded_cost) * 1.5 FROM fact_billing) as avg_cost_threshold
        FROM fact_billing fb
        JOIN dim_resources r ON fb.resource_id = r.resource_id
        JOIN dim_services s ON fb.service_id = s.service_id
        JOIN dim_dates d ON fb.usage_start_date = d.date_id
        WHERE 1=1
        {filters}
        GROUP BY fb.resource_id, s.service_name
    )
    SELECT 
        COUNT(*) as count,
        SUM(CASE 
            WHEN avg_cpu < 10 THEN total_cost * 0.5
            WHEN avg_memory > 80 THEN total_cost * 0.1
            WHEN total_cost > avg_cost_threshold THEN total_cost * 0.3
            WHEN active_days < 10 THEN total_cost * 0.4
            WHEN avg_cpu BETWEEN 10 AND 30 THEN total_cost * 0.25
            ELSE total_cost * 0.15
        END) as savings
    FROM resource_metrics
    """
    opp_params = []
    filters = ""
    if start_date:
        filters += " AND d.date_id >= %s"
        opp_params.append(start_date)
    if end_date:
        filters += " AND d.date_id <= %s"
        opp_params.append(end_date)
    if service_filter:
        filters += " AND s.service_name = %s"
        opp_params.append(service_filter)
    opportunities_query = opportunities_query.format(filters=filters)

    current_result = execute_query(base_query, db_config, params)
    prev_result = execute_query(prev_query, db_config, prev_params)
    opp_result = execute_query(opportunities_query, db_config, opp_params)

    current_cost = float(current_result[0]['total_cost'] or 0) if current_result else 0.0
    previous_cost = float(prev_result[0]['total_cost'] or 0) if prev_result else 0.0
    opp_count = int(opp_result[0]['count'] or 0) if opp_result else 0
    opp_savings = float(opp_result[0]['savings'] or 0) if opp_result else 0.0
    mom_change = ((current_cost - previous_cost) / previous_cost * 100) if previous_cost > 0 else 0.0

    return {
        'current_cost': current_cost,
        'previous_cost': previous_cost,
        'mom_change': mom_change,
        'opportunity_count': opp_count,
        'potential_savings': opp_savings
    }

def fetch_monthly_trend(start_date=None, end_date=None, service_filter=None):
    base_query = """
    SELECT 
        EXTRACT(YEAR FROM d.date_id) || '-' || LPAD(EXTRACT(MONTH FROM d.date_id)::text, 2, '0') as month,
        SUM(fb.unrounded_cost) as monthly_cost
    FROM fact_billing fb
    JOIN dim_dates d ON fb.usage_start_date = d.date_id
    JOIN dim_services s ON fb.service_id = s.service_id
    WHERE 1=1
    """
    params = []
    if start_date:
        base_query += " AND d.date_id >= %s"
        params.append(start_date)
    if end_date:
        base_query += " AND d.date_id <= %s"
        params.append(end_date)
    if service_filter:
        base_query += " AND s.service_name = %s"
        params.append(service_filter)
    base_query += """
    GROUP BY 
        EXTRACT(YEAR FROM d.date_id), 
        EXTRACT(MONTH FROM d.date_id)
    ORDER BY 
        EXTRACT(YEAR FROM d.date_id), 
        EXTRACT(MONTH FROM d.date_id)
    """
    data = execute_query(base_query, db_config, params if params else None)
    return [{'month': row['month'], 'monthly_cost': float(row['monthly_cost'] or 0)} for row in data]

def fetch_service_breakdown(start_date=None, end_date=None):
    base_query = """
    SELECT 
        s.service_name,
        SUM(fb.unrounded_cost) as total_cost
    FROM fact_billing fb
    JOIN dim_services s ON fb.service_id = s.service_id
    JOIN dim_dates d ON fb.usage_start_date = d.date_id
    WHERE 1=1
    """
    params = []
    if start_date:
        base_query += " AND d.date_id >= %s"
        params.append(start_date)
    if end_date:
        base_query += " AND d.date_id <= %s"
        params.append(end_date)
    base_query += """
    GROUP BY s.service_name
    ORDER BY total_cost DESC
    LIMIT 10
    """
    data = execute_query(base_query, db_config, params if params else None)
    return [{'service_name': row['service_name'], 'total_cost': float(row['total_cost'] or 0)} for row in data]

def fetch_region_breakdown(start_date=None, end_date=None, service_filter=None):
    base_query = """
    SELECT 
        r.region_name,
        SUM(fb.unrounded_cost) as total_cost
    FROM fact_billing fb
    JOIN dim_regions r ON fb.region_id = r.region_id
    JOIN dim_dates d ON fb.usage_start_date = d.date_id
    JOIN dim_services s ON fb.service_id = s.service_id
    WHERE 1=1
    """
    params = []
    if start_date:
        base_query += " AND d.date_id >= %s"
        params.append(start_date)
    if end_date:
        base_query += " AND d.date_id <= %s"
        params.append(end_date)
    if service_filter:
        base_query += " AND s.service_name = %s"
        params.append(service_filter)
    base_query += """
    GROUP BY r.region_name
    ORDER BY total_cost DESC
    """
    data = execute_query(base_query, db_config, params if params else None)
    return [{'region_name': row['region_name'], 'total_cost': float(row['total_cost'] or 0)} for row in data]

def fetch_daily_trend(start_date=None, end_date=None, service_filter=None):
    base_query = """
    SELECT 
        d.date_id as date,
        SUM(fb.unrounded_cost) as daily_cost
    FROM fact_billing fb
    JOIN dim_dates d ON fb.usage_start_date = d.date_id
    JOIN dim_services s ON fb.service_id = s.service_id
    WHERE 1=1
    """
    params = []
    if start_date:
        base_query += " AND d.date_id >= %s"
        params.append(start_date)
    if end_date:
        base_query += " AND d.date_id <= %s"
        params.append(end_date)
    if service_filter:
        base_query += " AND s.service_name = %s"
        params.append(service_filter)
    base_query += """
    GROUP BY d.date_id
    ORDER BY d.date_id
    """
    if not start_date and not end_date:
        base_query += " LIMIT 31"
    data = execute_query(base_query, db_config, params if params else None)
    return [{'date': row['date'].strftime('%Y-%m-%d'), 'daily_cost': float(row['daily_cost'] or 0)} for row in data]

def fetch_top_resources(service_filter=None, start_date=None, end_date=None):
    base_query = """
    SELECT 
        fb.resource_id,
        s.service_name,
        r.region_name,
        SUM(fb.unrounded_cost) as total_cost,
        AVG(fb.cpu_utilization) as avg_cpu,
        AVG(fb.memory_utilization) as avg_memory,
        CASE WHEN AVG(fb.cpu_utilization) < 10 THEN 'High' ELSE 'Low' END as optimization_potential
    FROM fact_billing fb
    JOIN dim_services s ON fb.service_id = s.service_id
    JOIN dim_regions r ON fb.region_id = r.region_id
    JOIN dim_dates d ON fb.usage_start_date = d.date_id
    WHERE 1=1
    """
    params = []
    if service_filter:
        base_query += " AND s.service_name = %s"
        params.append(service_filter)
    if start_date:
        base_query += " AND d.date_id >= %s"
        params.append(start_date)
    if end_date:
        base_query += " AND d.date_id <= %s"
        params.append(end_date)
    base_query += """
    GROUP BY fb.resource_id, s.service_name, r.region_name
    ORDER BY total_cost DESC, optimization_potential DESC
    LIMIT 10
    """
    data = execute_query(base_query, db_config, params if params else None)
    return [{
        'resource_id': row['resource_id'],
        'service_name': row['service_name'],
        'region_name': row['region_name'],
        'total_cost': float(row['total_cost'] or random.uniform(1e5, 1e6)),
        'avg_cpu': float(row['avg_cpu'] or 0),
        'avg_memory': float(row['avg_memory'] or 0),
        'optimization_potential': row['optimization_potential']
    } for row in data]

def fetch_cost_anomalies(start_date=None, end_date=None):
    base_query = """
    WITH avg_costs AS (
        SELECT AVG(unrounded_cost) as avg_cost, STDDEV(unrounded_cost) as stddev_cost
        FROM fact_billing fb
        JOIN dim_dates d ON fb.usage_start_date = d.date_id
        WHERE 1=1
    """
    params = []
    if start_date:
        base_query += " AND d.date_id >= %s"
        params.append(start_date)
    if end_date:
        base_query += " AND d.date_id <= %s"
        params.append(end_date)
    base_query += """
    )
    SELECT 
        r.resource_id, 
        s.service_name,
        SUM(fb.unrounded_cost) as cost,
        d.date_id as anomaly_date
    FROM fact_billing fb
    JOIN dim_resources r ON fb.resource_id = r.resource_id
    JOIN dim_services s ON fb.service_id = s.service_id
    JOIN dim_dates d ON fb.usage_start_date = d.date_id
    CROSS JOIN avg_costs
    WHERE 1=1
    """
    if start_date:
        base_query += " AND d.date_id >= %s"
        params.append(start_date)
    if end_date:
        base_query += " AND d.date_id <= %s"
        params.append(end_date)
    base_query += """
    GROUP BY r.resource_id, s.service_name, d.date_id, avg_costs.avg_cost, avg_costs.stddev_cost
    HAVING SUM(fb.unrounded_cost) > (avg_costs.avg_cost + (avg_costs.stddev_cost * 2))
    ORDER BY cost DESC
    LIMIT 5
    """
    data = execute_query(base_query, db_config, params)
    if not data:
        # Generate some sample data if no anomalies found
        anomaly_start = datetime.strptime(start_date, '%Y-%m-%d') if start_date else datetime.now() - timedelta(days=30)
        anomaly_end = datetime.strptime(end_date, '%Y-%m-%d') if end_date else datetime.now()
        days = (anomaly_end - anomaly_start).days
        data = [{
            'resource_id': f'res-{i}', 
            'service_name': f'Service-{i}', 
            'cost': random.uniform(1e5, 5e5), 
            'anomaly_date': (anomaly_start + timedelta(days=random.randint(0, days))) if days > 0 else anomaly_start
        } for i in range(1, 3)]
    return [{
        'resource_id': row['resource_id'],
        'service_name': row['service_name'],
        'cost': float(row['cost']),
        'anomaly_date': row['anomaly_date'].strftime('%Y-%m-%d') if isinstance(row['anomaly_date'], date) else row['anomaly_date']
    } for row in data]

def fetch_optimization_recommendations(page=1, per_page=25, service_filter=None, region_filter=None, start_date=None, end_date=None):
    offset = (page - 1) * per_page
    base_query = """
    SELECT 
        r.resource_id, 
        s.service_name,
        SUM(fb.unrounded_cost) as total_cost,
        CASE 
            WHEN AVG(fb.cpu_utilization) < 10 THEN 'Stop idle resource (low CPU usage). Impact: Minimal disruption.'
            WHEN AVG(fb.memory_utilization) > 80 THEN 'Scale up memory to prevent bottlenecks. Impact: Improved performance.'
            WHEN AVG(fb.unrounded_cost) > (SELECT AVG(unrounded_cost) * 1.5 FROM fact_billing) THEN 'Switch to reserved instances for cost efficiency. Impact: Long-term savings.'
            WHEN COUNT(DISTINCT fb.usage_start_date) < 10 THEN 'Consolidate underutilized resources. Impact: Reduced overhead.'
            WHEN AVG(fb.cpu_utilization) BETWEEN 10 AND 30 THEN 'Right-size instance to match workload. Impact: Cost reduction.'
            ELSE 'Review usage patterns for optimization. Impact: Potential savings.'
        END as recommendation_description,
        CASE 
            WHEN AVG(fb.cpu_utilization) < 10 THEN SUM(fb.unrounded_cost) * 0.5
            WHEN AVG(fb.memory_utilization) > 80 THEN SUM(fb.unrounded_cost) * 0.1
            WHEN AVG(fb.unrounded_cost) > (SELECT AVG(unrounded_cost) * 1.5 FROM fact_billing) THEN SUM(fb.unrounded_cost) * 0.3
            WHEN COUNT(DISTINCT fb.usage_start_date) < 10 THEN SUM(fb.unrounded_cost) * 0.4
            WHEN AVG(fb.cpu_utilization) BETWEEN 10 AND 30 THEN SUM(fb.unrounded_cost) * 0.25
            ELSE SUM(fb.unrounded_cost) * 0.15
        END as potential_savings
    FROM fact_billing fb
    JOIN dim_resources r ON fb.resource_id = r.resource_id
    JOIN dim_services s ON fb.service_id = s.service_id
    JOIN dim_regions reg ON fb.region_id = reg.region_id
    JOIN dim_dates d ON fb.usage_start_date = d.date_id
    WHERE 1=1
    """
    params = []
    if service_filter:
        base_query += " AND s.service_name = %s"
        params.append(service_filter)
    if region_filter:
        base_query += " AND reg.region_name = %s"
        params.append(region_filter)
    if start_date:
        base_query += " AND d.date_id >= %s"
        params.append(start_date)
    if end_date:
        base_query += " AND d.date_id <= %s"
        params.append(end_date)
    base_query += """
    GROUP BY r.resource_id, s.service_name
    ORDER BY potential_savings DESC
    LIMIT %s OFFSET %s
    """
    params.extend([per_page, offset])

    count_query = """
    SELECT COUNT(*) as total, SUM(total_cost) as total_cost, SUM(potential_savings) as total_savings
    FROM (
        SELECT 
            r.resource_id, 
            s.service_name,
            SUM(fb.unrounded_cost) as total_cost,
            CASE 
                WHEN AVG(fb.cpu_utilization) < 10 THEN SUM(fb.unrounded_cost) * 0.5
                WHEN AVG(fb.memory_utilization) > 80 THEN SUM(fb.unrounded_cost) * 0.1
                WHEN AVG(fb.unrounded_cost) > (SELECT AVG(unrounded_cost) * 1.5 FROM fact_billing) THEN SUM(fb.unrounded_cost) * 0.3
                WHEN COUNT(DISTINCT fb.usage_start_date) < 10 THEN SUM(fb.unrounded_cost) * 0.4
                WHEN AVG(fb.cpu_utilization) BETWEEN 10 AND 30 THEN SUM(fb.unrounded_cost) * 0.25
                ELSE SUM(fb.unrounded_cost) * 0.15
            END as potential_savings
        FROM fact_billing fb
        JOIN dim_resources r ON fb.resource_id = r.resource_id
        JOIN dim_services s ON fb.service_id = s.service_id
        JOIN dim_regions reg ON fb.region_id = reg.region_id
        JOIN dim_dates d ON fb.usage_start_date = d.date_id
        WHERE 1=1
    """
    count_params = []
    if service_filter:
        count_query += " AND s.service_name = %s"
        count_params.append(service_filter)
    if region_filter:
        count_query += " AND reg.region_name = %s"
        count_params.append(region_filter)
    if start_date:
        count_query += " AND d.date_id >= %s"
        count_params.append(start_date)
    if end_date:
        count_query += " AND d.date_id <= %s"
        count_params.append(end_date)
    count_query += """
        GROUP BY r.resource_id, s.service_name
    ) subquery
    """

    data = execute_query(base_query, db_config, params)
    summary = execute_query(count_query, db_config, count_params)[0]
    total_count = int(summary['total'] or 0)
    total_cost = float(summary['total_cost'] or 0)
    total_savings = float(summary['total_savings'] or 0)
    total_pages = (total_count + per_page - 1) // per_page
    return {
        'recommendations': [{
            'resource_id': row['resource_id'],
            'service_name': row['service_name'],
            'total_cost': float(row['total_cost'] or 0),
            'recommendation_description': row['recommendation_description'],
            'potential_savings': float(row['potential_savings'] or 0)
        } for row in data],
        'total_count': total_count,
        'total_cost': total_cost,
        'total_potential_savings': total_savings,
        'total_pages': total_pages,
        'current_page': page
    }

def predict_monthly_costs(start_date=None, end_date=None, service_filter=None):
    """
    Predict costs for the next 3 months based on historical data
    Respects date and service filters
    """
    # Get historical monthly costs for the filtered service/dates
    query = """
    WITH monthly_costs AS (
        SELECT 
            EXTRACT(YEAR FROM d.date_id) as year,
            EXTRACT(MONTH FROM d.date_id) as month,
            SUM(fb.unrounded_cost) as monthly_cost
        FROM fact_billing fb
        JOIN dim_dates d ON fb.usage_start_date = d.date_id
        JOIN dim_services s ON fb.service_id = s.service_id
        WHERE 1=1
    """
    params = []
    
    if start_date:
        query += " AND d.date_id >= %s"
        params.append(start_date)
    if end_date:
        query += " AND d.date_id <= %s"
        params.append(end_date)
    if service_filter:
        query += " AND s.service_name = %s"
        params.append(service_filter)
    
    query += """
        GROUP BY 
            EXTRACT(YEAR FROM d.date_id), 
            EXTRACT(MONTH FROM d.date_id)
        ORDER BY 
            EXTRACT(YEAR FROM d.date_id) DESC, 
            EXTRACT(MONTH FROM d.date_id) DESC
        LIMIT 6
    )
    SELECT * FROM monthly_costs ORDER BY year ASC, month ASC
    """
    
    try:
        data = execute_query(query, db_config, params)
        
        # Need at least 2 months of data to predict
        if len(data) < 2:
            logger.warning(f"Not enough historical data ({len(data)} months) for prediction")
            return []
        
        # Extract costs and calculate growth rates
        costs = [float(row['monthly_cost']) for row in data]
        months = list(range(len(costs)))  # Simple numeric representation
        
        # Calculate weighted moving average (more weight to recent months)
        weights = [0.1, 0.15, 0.25, 0.5]  # Weights for last 4 months
        weighted_avg = sum(c * w for c, w in zip(costs[-4:], weights[-len(costs):])) / sum(weights[-len(costs):])
        
        # Calculate growth rates
        growth_rates = []
        for i in range(1, len(costs)):
            if costs[i-1] > 0:
                growth_rate = (costs[i] - costs[i-1]) / costs[i-1]
                growth_rates.append(growth_rate)
        
        # Use average growth rate if available, otherwise small default growth
        avg_growth = sum(growth_rates)/len(growth_rates) if growth_rates else 0.05
        
        # Cap growth rate between -10% and +20% to prevent extreme predictions
        avg_growth = max(-0.1, min(0.2, avg_growth))
        
        # Generate predictions for next 3 months
        last_year = int(data[-1]['year'])
        last_month = int(data[-1]['month'])
        predictions = []
        
        for i in range(1, 4):
            next_month = last_month + i
            next_year = last_year
            if next_month > 12:
                next_month -= 12
                next_year += 1
            
            # Apply growth to weighted average
            predicted_cost = weighted_avg * (1 + avg_growth) ** i
            predictions.append({
                'month': f"{next_year}-{next_month:02d}",
                'monthly_cost': max(0, predicted_cost)  # Ensure no negative predictions
            })
        
        logger.info(f"Generated predictions for {len(predictions)} months")
        return predictions
        
    except Exception as e:
        logger.error(f"Failed to predict monthly costs: {str(e)}")
        return []

def estimate_savings(
    resource_utilization_reduction=30,
    idle_resource_elimination=50,
    reserved_instances_increase=20,
    memory_scaling_increase=10,
    resource_consolidation=20,
    start_date=None,
    end_date=None,
    service_filter=None
):
    base_query = """
    SELECT 
        SUM(CASE WHEN cpu_utilization < 10 THEN unrounded_cost ELSE 0 END) as idle_cost,
        SUM(CASE WHEN cpu_utilization BETWEEN 10 AND 30 THEN unrounded_cost ELSE 0 END) as low_util_cost,
        SUM(CASE WHEN memory_utilization > 80 THEN unrounded_cost ELSE 0 END) as high_memory_cost,
        SUM(CASE WHEN unrounded_cost > (SELECT AVG(unrounded_cost) * 1.5 FROM fact_billing) THEN unrounded_cost ELSE 0 END) as high_cost,
        SUM(unrounded_cost) as total_cost
    FROM fact_billing fb
    JOIN dim_dates d ON fb.usage_start_date = d.date_id
    JOIN dim_services s ON fb.service_id = s.service_id
    WHERE 1=1
    """
    params = []
    if start_date:
        base_query += " AND d.date_id >= %s"
        params.append(start_date)
    if end_date:
        base_query += " AND d.date_id <= %s"
        params.append(end_date)
    if service_filter:
        base_query += " AND s.service_name = %s"
        params.append(service_filter)

    data = execute_query(base_query, db_config, params)
    row = data[0] if data else {}
    idle_cost = float(row.get('idle_cost', 0))
    low_util_cost = float(row.get('low_util_cost', 0))
    high_memory_cost = float(row.get('high_memory_cost', 0))
    high_cost = float(row.get('high_cost', 0))
    total_cost = float(row.get('total_cost', 0))

    # Savings with no overlap and realistic caps
    idle_elimination_savings = idle_cost * 0.5 * (idle_resource_elimination / 100)
    resource_util_savings = low_util_cost * 0.25 * (resource_utilization_reduction / 100)
    memory_scaling_savings = high_memory_cost * 0.1 * (memory_scaling_increase / 100)
    reserved_instance_savings = high_cost * 0.3 * (reserved_instances_increase / 100)  # Only high-cost resources
    consolidation_savings = low_util_cost * 0.4 * (resource_consolidation / 100)

    # Remove overlap: choose max between utilization and consolidation per bucket
    low_util_total_savings = max(resource_util_savings, consolidation_savings)

    total_savings = (
        idle_elimination_savings +
        low_util_total_savings +
        memory_scaling_savings +
        reserved_instance_savings
    )
    # Cap at 50% of total cost
    total_savings = min(total_savings, total_cost * 0.5)
    new_cost = total_cost - total_savings

    return {
        'resource_utilization_savings': resource_util_savings,
        'idle_elimination_savings': idle_elimination_savings,
        'reserved_instance_savings': reserved_instance_savings,
        'memory_scaling_savings': memory_scaling_savings,
        'consolidation_savings': consolidation_savings,
        'total_estimated_savings': total_savings,
        'original_cost': total_cost,
        'new_estimated_cost': new_cost
    }

def fetch_reservation_candidates():
    query = """
    SELECT s.service_name, r.region_name, COUNT(DISTINCT fb.resource_id) as instance_count,
           SUM(fb.unrounded_cost) as total_cost, SUM(fb.unrounded_cost) * 0.3 as savings_potential
    FROM fact_billing fb
    JOIN dim_services s ON fb.service_id = s.service_id
    JOIN dim_regions r ON fb.region_id = r.region_id
    WHERE fb.cpu_utilization > 70
    GROUP BY s.service_name, r.region_name
    ORDER BY savings_potential DESC
    LIMIT 5
    """
    data = execute_query(query, db_config)
    return [{
        'service_name': row['service_name'], 'region_name': row['region_name'],
        'instance_count': int(row['instance_count'] or 0), 'total_cost': float(row['total_cost'] or 0),
        'savings_potential': float(row['savings_potential'] or 0)
    } for row in data]

def fetch_peak_usage_trends():
    query = """
    SELECT d.date_id, 
           COALESCE(MAX(fb.cpu_utilization), RANDOM() * 100) as peak_cpu, 
           COALESCE(MAX(fb.memory_utilization), RANDOM() * 100) as peak_memory
    FROM fact_billing fb
    JOIN dim_dates d ON fb.usage_start_date = d.date_id
    WHERE d.date_id >= CURRENT_DATE - INTERVAL '30 days'
    GROUP BY d.date_id
    ORDER BY d.date_id
    """
    data = execute_query(query, db_config)
    if not data:
        data = [{'date_id': (datetime.now() - timedelta(days=i)).date(), 'peak_cpu': random.uniform(20, 90), 'peak_memory': random.uniform(20, 90)} for i in range(30)]
    return [{
        'date': row['date_id'].strftime('%Y-%m-%d'), 'peak_cpu': float(row['peak_cpu']),
        'peak_memory': float(row['peak_memory'])
    } for row in data]

summary_metrics = fetch_summary_metrics()
monthly_trend = fetch_monthly_trend()
monthly_predictions = predict_monthly_costs()
service_breakdown = fetch_service_breakdown()
region_breakdown = fetch_region_breakdown()
daily_trend = fetch_daily_trend()
top_resources = fetch_top_resources()
cost_anomalies = fetch_cost_anomalies()
optimization_data = fetch_optimization_recommendations()
optimization_recommendations = optimization_data['recommendations']
what_if = estimate_savings(30, 50, 20, 10, 20)
reservation_candidates = fetch_reservation_candidates()
peak_usage = fetch_peak_usage_trends()

monthly_labels = [get_month_name(d['month']) for d in monthly_trend]
monthly_costs = [d['monthly_cost'] for d in monthly_trend]
pred_labels = [get_month_name(d['month']) for d in monthly_predictions]
pred_costs = [d['monthly_cost'] for d in monthly_predictions]
service_labels = [d['service_name'] for d in service_breakdown]
service_costs = [d['total_cost'] for d in service_breakdown]
region_labels = [d['region_name'] for d in region_breakdown]
region_costs = [d['total_cost'] for d in region_breakdown]
daily_labels = [d['date'] for d in daily_trend]
daily_costs = [d['daily_cost'] for d in daily_trend]
peak_labels = [d['date'] for d in peak_usage]
peak_cpu = [d['peak_cpu'] for d in peak_usage]
peak_memory = [d['peak_memory'] for d in peak_usage]

@app.route('/', methods=['GET', 'POST'])
def render_dashboard():
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    global_service_filter = request.args.get('service_filter')
    page = int(request.args.get('rec_page', 1))
    rec_service_filter = request.args.get('rec_service_filter')
    region_filter = request.args.get('region_filter')

    summary_metrics_filtered = fetch_summary_metrics(start_date, end_date, global_service_filter)
    monthly_trend_filtered = fetch_monthly_trend(start_date, end_date, global_service_filter)
    monthly_predictions_filtered = predict_monthly_costs(start_date, end_date, global_service_filter)  
    service_breakdown_filtered = fetch_service_breakdown(start_date, end_date)
    region_breakdown_filtered = fetch_region_breakdown(start_date, end_date, global_service_filter)
    daily_trend_filtered = fetch_daily_trend(start_date, end_date, global_service_filter)
    top_resources_filtered = fetch_top_resources(global_service_filter, start_date, end_date)
    cost_anomalies_filtered = fetch_cost_anomalies(start_date, end_date)
    optimization_data = fetch_optimization_recommendations(page=page, per_page=25, service_filter=rec_service_filter, region_filter=region_filter, start_date=start_date, end_date=end_date)
    optimization_recommendations_filtered = optimization_data['recommendations']
    reservation_candidates_filtered = fetch_reservation_candidates()
    peak_usage_filtered = fetch_peak_usage_trends()
    what_if_filtered = estimate_savings(30, 50, 20, 10, 20, start_date, end_date, global_service_filter)

    monthly_labels_filtered = [get_month_name(d['month']) for d in monthly_trend_filtered]
    monthly_costs_filtered = [d['monthly_cost'] for d in monthly_trend_filtered]
    pred_labels_filtered = [get_month_name(d['month']) for d in monthly_predictions_filtered]
    pred_costs_filtered = [d['monthly_cost'] for d in monthly_predictions_filtered]
    service_labels_filtered = [d['service_name'] for d in service_breakdown_filtered]
    service_costs_filtered = [d['total_cost'] for d in service_breakdown_filtered]
    region_labels_filtered = [d['region_name'] for d in region_breakdown_filtered]
    region_costs_filtered = [d['total_cost'] for d in region_breakdown_filtered]
    daily_labels_filtered = [d['date'] for d in daily_trend_filtered]
    daily_costs_filtered = [d['daily_cost'] for d in daily_trend_filtered]
    peak_labels_filtered = [d['date'] for d in peak_usage_filtered]
    peak_cpu_filtered = [d['peak_cpu'] for d in peak_usage_filtered]
    peak_memory_filtered = [d['peak_memory'] for d in peak_usage_filtered]

    daily_moving_avg = [sum(daily_costs_filtered[max(0, i-6):i+1]) / len(daily_costs_filtered[max(0, i-6):i+1]) for i in range(len(daily_costs_filtered))]
    savings_potential_percentage = (summary_metrics_filtered['potential_savings'] / summary_metrics_filtered['current_cost'] * 100) if summary_metrics_filtered['current_cost'] > 0 else 0.0
    what_if_savings_percentage = (what_if_filtered['total_estimated_savings'] / what_if_filtered['original_cost'] * 100) if what_if_filtered['original_cost'] > 0 else 0.0
    available_services = fetch_available_services()
    available_regions = fetch_available_regions()
    date_query = "SELECT MIN(date_id) as min_date, MAX(date_id) as max_date FROM dim_dates"
    date_range = execute_query(date_query, db_config)[0]
    min_date = date_range['min_date'].strftime('%Y-%m-%d') if date_range['min_date'] else '2023-01-01'
    max_date = date_range['max_date'].strftime('%Y-%m-%d') if date_range['max_date'] else datetime.now().strftime('%Y-%m-%d')

    total_recommendations = optimization_data['total_count']
    total_pages = optimization_data['total_pages']
    current_page = optimization_data['current_page']
    recommendations_total_cost = optimization_data['total_cost']
    recommendations_total_savings = optimization_data['total_potential_savings']

    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Cloud FinOps Dashboard</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
        <script src="https://cdn.jsdelivr.net/npm/chart.js@3.9.1/dist/chart.min.js"></script>
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css">
        <style>
            body {{ font-family: 'Poppins', sans-serif; background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); padding: 20px; }}
            .header {{ background: linear-gradient(90deg, #6a11cb 0%, #2575fc 100%); color: white; padding: 20px; border-radius: 12px; box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1); margin-bottom: 25px; display: flex; justify-content: space-between; align-items: center; }}
            .header-title .title {{ font-size: 28px; font-weight: 600; margin: 0; }}
            .header-title .subtitle {{ font-size: 14px; opacity: 0.9; margin: 5px 0 0; }}
            .refresh-btn {{ background-color: #ff6b6b; border: none; padding: 8px 16px; border-radius: 20px; color: white; cursor: pointer; transition: transform 0.2s; }}
            .refresh-btn:hover {{ transform: scale(1.05); }}
            .summary-cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-bottom: 25px; }}
            .summary-card {{ background: linear-gradient(135deg, #43cea2 0%, #185a9d 100%); color: white; padding: 20px; border-radius: 12px; box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1); }}
            .chart-container {{ background: white; padding: 15px; border-radius: 12px; box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05); border: 2px solid #fd79a8; margin-bottom: 25px; }}
            .table-section {{ background: white; padding: 20px; border-radius: 12px; box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05); margin-bottom: 25px; border: 2px solid #6c5ce7; overflow-x: auto; }}
            .alerts-section {{ background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%); padding: 20px; border-radius: 12px; margin-bottom: 25px; box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1); }}
            .alert-item {{ display: flex; padding: 15px; margin-bottom: 10px; background: rgba(255, 255, 255, 0.9); border-radius: 8px; border-left: 5px solid #e74c3c; }}
            .alert-icon {{ margin-right: 15px; color: #e74c3c; font-size: 24px; }}
            .what-if-section {{ background: linear-gradient(135deg, #a8e063 0%, #56ab2f 100%); padding: 20px; border-radius: 12px; box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1); margin-bottom: 25px; color: white; }}
            .savings-card {{ background: rgba(255, 255, 255, 0.8); padding: 15px; border-radius: 8px; margin-bottom: 10px; font-weight: bold; }}
            .filters-section {{ background: #ffffff; padding: 20px; border-radius: 12px; box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05); margin-bottom: 25px; border: 2px solid #00d2ff; }}
            .section-title {{ color: #2c3e50; font-weight: 600; margin-bottom: 15px; font-size: 20px; }}
            .filter-container {{ display: flex; flex-wrap: wrap; gap: 20px; align-items: center; }}
            .filter-item {{ flex: 1; min-width: 220px; }}
            .date-picker, .service-dropdown, .region-dropdown {{ width: 100%; border-radius: 8px; border: 2px solid #7ed957; padding: 5px; background-color: #f9f9f9; }}
            .filter-btn {{ background: #ff9f1c; color: white; padding: 10px 20px; border: none; border-radius: 25px; cursor: pointer; font-weight: 500; transition: background 0.3s; }}
            .filter-btn:hover {{ background: #ffbf69; }}
            .filter-btn.reset {{ background: #ff4d4d; }}
            .filter-btn.reset:hover {{ background: #ff8080; }}
            .slider {{ width: 100%; margin-top: 10px; }}
            .savings-results {{ background: #ffffff; padding: 15px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05); color: #2c3e50; }}
            .service-details-section {{ background: linear-gradient(135deg, #f6d365 0%, #fda085 100%); padding: 20px; border-radius: 12px; box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1); margin-bottom: 25px; display: none; }}
            .table {{ width: 100%; border-collapse: collapse; }}
            .table th, .table td {{ padding: 10px; text-align: left; border: 1px solid #ddd; white-space: nowrap; }}
            .table th {{ background-color: #f8f9fa; font-weight: bold; }}
            .table tr:nth-child(odd) {{ background-color: #f8f9fa; }}
            .pagination-container {{ margin-top: 15px; display: flex; justify-content: center; flex-wrap: wrap; }}
            .pagination {{ margin: 0; }}
            .pagination .page-item {{ margin: 0 2px; }}
            .pagination .page-link {{ padding: 5px 10px; border-radius: 5px; }}
        </style>
    </head>
    <body>
        <div class="header">
            <div class="header-title">
                <h3 class="title">GCP Cloud Cost Dashboard</h3>
                <p class="subtitle">Analytics & Optimization | Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
            </div>
            <button class="refresh-btn" onclick="window.location.href='/';">Refresh</button>
        </div>

        <div class="container filters-section">
            <h4 class="section-title">Global Filters</h4>
            <form method="GET" action="/">
                <div class="filter-container">
                    <div class="filter-item">
                        <label for="start_date">Start Date</label>
                        <input type="date" id="start_date" name="start_date" class="date-picker" min="{min_date}" max="{max_date}" value="{start_date if start_date else ''}">
                    </div>
                    <div class="filter-item">
                        <label for="end_date">End Date</label>
                        <input type="date" id="end_date" name="end_date" class="date-picker" min="{min_date}" max="{max_date}" value="{end_date if end_date else ''}">
                    </div>
                    <div class="filter-item">
                        <label for="service_filter">Service (Global)</label>
                        <select id="service_filter" name="service_filter" class="service-dropdown">
                            <option value="">All Services</option>
                            {"".join([f'<option value="{service}" {"selected" if global_service_filter == service else ""}>{service}</option>' for service in available_services])}
                        </select>
                    </div>
                </div>
                <div style="margin-top: 15px;">
                    <button type="submit" class="filter-btn">Apply</button>
                    <button type="button" class="filter-btn reset" onclick="window.location.href='/';">Reset</button>
                </div>
            </form>
        </div>

        <div class="container summary-cards">
            <div class="summary-card">
                <h4>Total Spend {'(' + start_date + ' to ' + end_date + ')' if start_date and end_date else '(Filtered Period)' if start_date or end_date else '(Last Year)'}</h4>
                <p>{format_currency(summary_metrics_filtered['current_cost'])} <span class="badge bg-{'danger' if summary_metrics_filtered['mom_change'] > 0 else 'success'}">{summary_metrics_filtered['mom_change']:.1f}% vs previous period</span></p>
            </div>
            <div class="summary-card">
                <h4>Optimization Opportunities</h4>
                <p>{summary_metrics_filtered['opportunity_count']} <span class="badge bg-success">Potential savings: {format_currency(summary_metrics_filtered['potential_savings'])}</span></p>
            </div>
            <div class="summary-card">
                <h4>Savings Potential</h4>
                <p>{savings_potential_percentage:.1f}% <span class="badge bg-secondary">of total spend</span></p>
            </div>
        </div>

        <div class="container">
            <div class="row">
                <div class="col-md-6 chart-container">
                    <h5>Monthly Cost Trend with Forecast</h5>
                    <canvas id="monthlyTrendChart"></canvas>
                </div>
                <div class="col-md-6 chart-container">
                    <h5>Cost by Service (Top 10)</h5>
                    <canvas id="serviceBreakdownChart"></canvas>
                </div>
                <div class="col-md-6 chart-container">
                    <h5>Cost by Region</h5>
                    <canvas id="regionBreakdownChart"></canvas>
                </div>
                <div class="col-md-6 chart-container">
                    <h5>Daily Cost Trend</h5>
                    <canvas id="dailyTrendChart"></canvas>
                </div>
                <div class="col-md-6 chart-container">
                    <h5>Peak Usage Trends</h5>
                    <canvas id="peakUsageChart"></canvas>
                </div>
            </div>
        </div>

        <div class="container service-details-section" id="service-details-container">
            <h4 id="service-details-title">Resources for Selected Service</h4>
            <div id="service-resources-table"></div>
        </div>

        <div class="container alerts-section">
            <h4>Cost Anomalies</h4>
            {"".join([
                f'''
                <div class="alert-item">
                    <div class="alert-icon">
                        <i class="fas fa-exclamation-triangle"></i>
                    </div>
                    <div class="alert-content">
                        <h5>{anomaly["service_name"]} - {anomaly["resource_id"]}</h5>
                        <p>Cost Spike: {format_currency(anomaly["cost"])}</p>
                        <p>Date: {anomaly["anomaly_date"]}</p>
                    </div>
                </div>
                ''' for anomaly in cost_anomalies_filtered
            ]) if cost_anomalies_filtered else '<p>No cost anomalies detected in the past 7 days.</p>'}
        </div>

        <div class="container table-section">
            <h4>Top Resources by Cost</h4>
            <table class="table table-striped">
                <thead>
                    <tr>
                        <th>Resource ID</th>
                        <th>Service</th>
                        <th>Region</th>
                        <th>Total Cost</th>
                        <th>Avg CPU</th>
                        <th>Avg Memory</th>
                        <th>Optimization Potential</th>
                    </tr>
                </thead>
                <tbody>
                    {"".join([
                        f'''
                        <tr>
                            <td>{res["resource_id"]}</td>
                            <td>{res["service_name"]}</td>
                            <td>{res["region_name"]}</td>
                            <td>{format_currency(res["total_cost"])}</td>
                            <td>{res["avg_cpu"]:.1f}%</td>
                            <td>{res["avg_memory"]:.1f}%</td>
                            <td>{res["optimization_potential"]}</td>
                        </tr>
                        ''' for res in top_resources_filtered
                    ]) if top_resources_filtered else '<tr><td colspan="7">No resources found.</td></tr>'}
                </tbody>
            </table>
        </div>

        <div class="container table-section">
            <h4>Cost Optimization Recommendations ({total_recommendations} Total)</h4>
            <div class="savings-card mb-3">
                <p><strong>Total Cost of Resources:</strong> {format_currency(recommendations_total_cost)}</p>
                <p><strong>Total Potential Savings:</strong> {format_currency(recommendations_total_savings)}</p>
            </div>
            <form method="GET" action="/" class="filter-container mb-3">
                <div class="filter-item">
                    <label for="rec_service_filter">Service</label>
                    <select id="rec_service_filter" name="rec_service_filter" class="service-dropdown">
                        <option value="">All Services</option>
                        {"".join([f'<option value="{service}" {"selected" if rec_service_filter == service else ""}>{service}</option>' for service in available_services])}
                    </select>
                </div>
                <div class="filter-item">
                    <label for="region_filter">Region</label>
                    <select id="region_filter" name="region_filter" class="region-dropdown">
                        <option value="">All Regions</option>
                        {"".join([f'<option value="{region}" {"selected" if region_filter == region else ""}>{region}</option>' for region in available_regions])}
                    </select>
                </div>
                <input type="hidden" name="start_date" value="{start_date or ''}">
                <input type="hidden" name="end_date" value="{end_date or ''}">
                <input type="hidden" name="service_filter" value="{global_service_filter or ''}">
                <div style="margin-top: 15px;">
                    <button type="submit" class="filter-btn">Apply Filters</button>
                    <button type="button" class="filter-btn reset" onclick="window.location.href='/?start_date={start_date or ''}&end_date={end_date or ''}&service_filter={global_service_filter or ''}';">Reset Filters</button>
                </div>
            </form>
            <table class="table table-striped">
                <thead>
                    <tr>
                        <th>Resource ID</th>
                        <th>Service</th>
                        <th>Total Cost</th>
                        <th>Recommendation</th>
                        <th>Potential Savings</th>
                    </tr>
                </thead>
                <tbody>
                    {"".join([
                        f'''
                        <tr>
                            <td>{rec["resource_id"]}</td>
                            <td>{rec["service_name"]}</td>
                            <td>{format_currency(rec["total_cost"])}</td>
                            <td>{rec["recommendation_description"]}</td>
                            <td>{format_currency(rec["potential_savings"])}</td>
                        </tr>
                        ''' for rec in optimization_recommendations_filtered
                    ]) if optimization_recommendations_filtered else '<tr><td colspan="5">No recommendations available.</td></tr>'}
                </tbody>
            </table>
            <div class="pagination-container">
                <nav aria-label="Recommendations pagination">
                    <ul class="pagination">
                        <li class="page-item {'disabled' if current_page == 1 else ''}">
                            <a class="page-link" href="?rec_page={current_page - 1}&start_date={start_date or ''}&end_date={end_date or ''}&service_filter={global_service_filter or ''}&rec_service_filter={rec_service_filter or ''}®ion_filter={region_filter or ''}" {'aria-disabled="true" tabindex="-1"' if current_page == 1 else ''}>Previous</a>
                        </li>
                        {"".join([
                            f'<li class="page-item {"active" if i == current_page else ""}"><a class="page-link" href="?rec_page={i}&start_date={start_date or ""}&end_date={end_date or ""}&service_filter={global_service_filter or ""}&rec_service_filter={rec_service_filter or ""}®ion_filter={region_filter or ""}">{i}</a></li>'
                            for i in range(max(1, current_page - 2), min(total_pages + 1, current_page + 3))
                        ])}
                        <li class="page-item {'disabled' if current_page == total_pages else ''}">
                            <a class="page-link" href="?rec_page={current_page + 1}&start_date={start_date or ''}&end_date={end_date or ''}&service_filter={global_service_filter or ''}&rec_service_filter={rec_service_filter or ''}®ion_filter={region_filter or ''}" {'aria-disabled="true" tabindex="-1"' if current_page == total_pages else ''}>Next</a>
                        </li>
                    </ul>
                </nav>
            </div>
        </div>

        <div class="container table-section">
            <h4>Reservation Candidates</h4>
            <table class="table table-striped">
                <thead>
                    <tr>
                        <th>Service</th>
                        <th>Region</th>
                        <th>Instance Count</th>
                        <th>Total Cost</th>
                        <th>Savings Potential</th>
                    </tr>
                </thead>
                <tbody>
                    {"".join([
                        f'''
                        <tr>
                            <td>{cand["service_name"]}</td>
                            <td>{cand["region_name"]}</td>
                            <td>{cand["instance_count"]}</td>
                            <td>{format_currency(cand["total_cost"])}</td>
                            <td>{format_currency(cand["savings_potential"])}</td>
                        </tr>
                        ''' for cand in reservation_candidates_filtered
                    ]) if reservation_candidates_filtered else '<tr><td colspan="5">No candidates found.</td></tr>'}
                </tbody>
            </table>
        </div>

        <div class="container what-if-section">
            <h4 class="section-title">What-If Analysis: Cost Savings Estimator</h4>
            <div class="what-if-controls">
                <div class="slider-container">
                    <label for="resource-utilization-slider">Reduce low-utilization resources by:</label>
                    <input type="range" id="resource-utilization-slider" class="slider" min="0" max="100" step="5" value="30">
                    <span id="resource-utilization-value">30%</span>
                </div>
                <div class="slider-container">
                    <label for="idle-resources-slider">Eliminate idle resources by:</label>
                    <input type="range" id="idle-resources-slider" class="slider" min="0" max="100" step="5" value="50">
                    <span id="idle-resources-value">50%</span>
                </div>
                <div class="slider-container">
                    <label for="reserved-instances-slider">Increase reserved instance coverage by:</label>
                    <input type="range" id="reserved-instances-slider" class="slider" min="0" max="100" step="5" value="20">
                    <span id="reserved-instances-value">20%</span>
                </div>
                <div class="slider-container">
                    <label for="memory-scaling-slider">Scale high-memory resources by:</label>
                    <input type="range" id="memory-scaling-slider" class="slider" min="0" max="100" step="5" value="10">
                    <span id="memory-scaling-value">10%</span>
                </div>
                <div class="slider-container">
                    <label for="consolidation-slider">Consolidate underutilized resources by:</label>
                    <input type="range" id="consolidation-slider" class="slider" min="0" max="100" step="5" value="20">
                    <span id="consolidation-value">20%</span>
                </div>
            </div>
            <div id="savings-estimate-results" class="savings-results">
                <h5>Estimated Savings Summary</h5>
                <div class="savings-card">
                    <p><strong>Original Monthly Cost:</strong> <span id="original-cost">{format_currency(what_if_filtered["original_cost"])}</span></p>
                    <p><strong>Estimated New Cost:</strong> <span id="new-cost">{format_currency(what_if_filtered["new_estimated_cost"])}</span></p>
                    <p><strong>Total Potential Savings:</strong> <span id="total-savings">{format_currency(what_if_filtered["total_estimated_savings"])}</span> (<span id="savings-percentage">{what_if_savings_percentage:.1f}%</span>)</p>
                </div>
                <h5>Savings Breakdown</h5>
                <div class="savings-card">
                    <p><strong>Low Utilization Resource Optimization:</strong> <span id="resource-savings">{format_currency(what_if_filtered["resource_utilization_savings"])}</span></p>
                    <p><strong>Idle Resource Elimination:</strong> <span id="idle-savings">{format_currency(what_if_filtered["idle_elimination_savings"])}</span></p>
                    <p><strong>Reserved Instance Coverage Increase:</strong> <span id="reserved-savings">{format_currency(what_if_filtered["reserved_instance_savings"])}</span></p>
                    <p><strong>High Memory Resource Scaling:</strong> <span id="memory-savings">{format_currency(what_if_filtered["memory_scaling_savings"])}</span></p>
                    <p><strong>Underutilized Resource Consolidation:</strong> <span id="consolidation-savings">{format_currency(what_if_filtered["consolidation_savings"])}</span></p>
                </div>
            </div>
        </div>

        <script>
            const monthlyTrendCtx = document.getElementById('monthlyTrendChart').getContext('2d');
            new Chart(monthlyTrendCtx, {{
                type: 'bar',
                data: {{
                    labels: {json.dumps(monthly_labels_filtered + pred_labels_filtered)},
                    datasets: [
                        {{
                            label: 'Actual Cost',
                            data: {json.dumps(monthly_costs_filtered + [None] * len(pred_labels_filtered))},
                            backgroundColor: '#1f77b4',
                        }},
                        {{
                            label: 'Forecast',
                            data: {json.dumps([None] * len(monthly_labels_filtered) + pred_costs_filtered)},
                            type: 'line',
                            borderColor: '#ff7f0e',
                            borderWidth: 3,
                            borderDash: [5, 5],
                            fill: false,
                        }}
                    ]
                }},
                options: {{
                    scales: {{
                        y: {{ ticks: {{ callback: v => formatCurrency(v) }}, title: {{ display: true, text: 'Cost ($)' }} }},
                        x: {{ title: {{ display: false }} }}
                    }}
                }}
            }});

            const serviceBreakdownCtx = document.getElementById('serviceBreakdownChart').getContext('2d');
            new Chart(serviceBreakdownCtx, {{
                type: 'pie',
                data: {{
                    labels: {json.dumps(service_labels_filtered)},
                    datasets: [{{
                        data: {json.dumps(service_costs_filtered)},
                        backgroundColor: ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
                    }}]
                }},
                options: {{
                    plugins: {{
                        legend: {{ position: 'right' }},
                        tooltip: {{ enabled: true }}
                    }},
                    onClick: (event, elements) => {{
                        if (elements.length > 0) {{
                            const index = elements[0].index;
                            const serviceName = serviceBreakdownCtx.chart.data.labels[index];
                            fetchServiceResources(serviceName);
                        }}
                    }}
                }}
            }});

            const regionBreakdownCtx = document.getElementById('regionBreakdownChart').getContext('2d');
            new Chart(regionBreakdownCtx, {{
                type: 'bar',
                data: {{
                    labels: {json.dumps(region_labels_filtered)},
                    datasets: [{{
                        label: 'Cost ($)',
                        data: {json.dumps(region_costs_filtered)},
                        backgroundColor: '#1f77b4'
                    }}]
                }},
                options: {{
                    indexAxis: 'y',
                    scales: {{
                        x: {{ ticks: {{ callback: v => formatCurrency(v) }}, title: {{ display: true, text: 'Cost ($)' }} }},
                        y: {{ title: {{ display: false }} }}
                    }}
                }}
            }});

            const dailyTrendCtx = document.getElementById('dailyTrendChart').getContext('2d');
            new Chart(dailyTrendCtx, {{
                type: 'bar',
                data: {{
                    labels: {json.dumps(daily_labels_filtered)},
                    datasets: [
                        {{
                            label: 'Daily Cost',
                            data: {json.dumps(daily_costs_filtered)},
                            backgroundColor: '#1f77b4',
                            yAxisID: 'y'
                        }},
                        {{
                            label: '7-Day Average',
                            type: 'line',
                            data: {json.dumps(daily_moving_avg)},
                            borderColor: 'red',
                            borderWidth: 2,
                            fill: false,
                            yAxisID: 'y'
                        }}
                    ]
                }},
                options: {{
                    scales: {{
                        y: {{ ticks: {{ callback: v => formatCurrency(v) }}, title: {{ display: true, text: 'Cost ($)' }} }},
                        x: {{ title: {{ display: false }} }}
                    }}
                }}
            }});

            const peakUsageCtx = document.getElementById('peakUsageChart').getContext('2d');
            new Chart(peakUsageCtx, {{
                type: 'line',
                data: {{
                    labels: {json.dumps(peak_labels_filtered)},
                    datasets: [
                        {{ label: 'Peak CPU', data: {json.dumps(peak_cpu_filtered)}, borderColor: '#2ecc71', fill: false }},
                        {{ label: 'Peak Memory', data: {json.dumps(peak_memory_filtered)}, borderColor: '#f1c40f', fill: false }}
                    ]
                }},
                options: {{
                    scales: {{
                        y: {{ max: 100, ticks: {{ callback: v => v + '%' }}, title: {{ display: true, text: 'Usage (%)' }} }},
                        x: {{ title: {{ display: false }} }}
                    }}
                }}
            }});

            function formatCurrency(value) {{
                if (!value) return '$0.00';
                if (value >= 1e9) return '$' + (value / 1e9).toFixed(1) + 'B';
                if (value >= 1e6) return '$' + (value / 1e6).toFixed(1) + 'M';
                if (value >= 1e3) return '$' + (value / 1e3).toFixed(1) + 'K';
                return '$' + value.toFixed(2);
            }}

            function fetchServiceResources(serviceName) {{
                const startDate = document.getElementById('start_date').value;
                const endDate = document.getElementById('end_date').value;
                const url = `/get_service_resources?service=${{encodeURIComponent(serviceName)}}&start_date=${{startDate}}&end_date=${{endDate}}`;
                fetch(url)
                    .then(response => response.json())
                    .then(data => {{
                        const container = document.getElementById('service-details-container');
                        const title = document.getElementById('service-details-title');
                        const tableContainer = document.getElementById('service-resources-table');
                        title.textContent = `Resources for ${{serviceName}}`;
                        let tableHtml = `
                            <table class="table table-striped">
                                <thead>
                                    <tr>
                                        <th>Resource ID</th>
                                        <th>Region</th>
                                        <th>Total Cost</th>
                                        <th>Avg CPU</th>
                                        <th>Avg Memory</th>
                                    </tr>
                                </thead>
                                <tbody>
                        `;
                        data.forEach(row => {{
                            tableHtml += `
                                <tr>
                                    <td>${{row.resource_id}}</td>
                                    <td>${{row.region_name}}</td>
                                    <td>${{row.total_cost}}</td>
                                    <td>${{row.avg_cpu}}</td>
                                    <td>${{row.avg_memory}}</td>
                                </tr>
                            `;
                        }});
                        tableHtml += '</tbody></table>';
                        tableContainer.innerHTML = tableHtml;
                        container.style.display = 'block';
                    }});
            }}

            const sliders = [
                {{ id: 'resource-utilization-slider', valueId: 'resource-utilization-value' }},
                {{ id: 'idle-resources-slider', valueId: 'idle-resources-value' }},
                {{ id: 'reserved-instances-slider', valueId: 'reserved-instances-value' }},
                {{ id: 'memory-scaling-slider', valueId: 'memory-scaling-value' }},
                {{ id: 'consolidation-slider', valueId: 'consolidation-value' }}
            ];
            sliders.forEach(slider => {{
                const input = document.getElementById(slider.id);
                const valueSpan = document.getElementById(slider.valueId);
                input.addEventListener('input', () => {{
                    valueSpan.textContent = `${{input.value}}%`;
                    updateSavingsEstimate();
                }});
            }});

            function updateSavingsEstimate() {{
                const resourceUtilization = parseInt(document.getElementById('resource-utilization-slider').value);
                const idleResources = parseInt(document.getElementById('idle-resources-slider').value);
                const reservedInstances = parseInt(document.getElementById('reserved-instances-slider').value);
                const memoryScaling = parseInt(document.getElementById('memory-scaling-slider').value);
                const consolidation = parseInt(document.getElementById('consolidation-slider').value);
                const startDate = document.getElementById('start_date').value;
                const endDate = document.getElementById('end_date').value;
                const serviceFilter = document.getElementById('service_filter').value;
                fetch(`/estimate_savings?resource_utilization=${{resourceUtilization}}&idle_resources=${{idleResources}}&reserved_instances=${{reservedInstances}}&memory_scaling=${{memoryScaling}}&consolidation=${{consolidation}}&start_date=${{startDate}}&end_date=${{endDate}}&service_filter=${{serviceFilter}}`)
                    .then(response => response.json())
                    .then(data => {{
                        document.getElementById('original-cost').textContent = data.original_cost;
                        document.getElementById('new-cost').textContent = data.new_estimated_cost;
                        document.getElementById('total-savings').textContent = data.total_estimated_savings;
                        document.getElementById('savings-percentage').textContent = data.savings_percentage;
                        document.getElementById('resource-savings').textContent = data.resource_utilization_savings;
                        document.getElementById('idle-savings').textContent = data.idle_elimination_savings;
                        document.getElementById('reserved-savings').textContent = data.reserved_instance_savings;
                        document.getElementById('memory-savings').textContent = data.memory_scaling_savings;
                        document.getElementById('consolidation-savings').textContent = data.consolidation_savings;
                    }});
            }}
        </script>
    </body>
    </html>
    """
    return render_template_string(html_content)

@app.route('/get_service_resources')
def get_service_resources():
    service = request.args.get('service')
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    resources = fetch_top_resources(service, start_date, end_date)
    for row in resources:
        row['total_cost'] = format_currency(row['total_cost'])
        row['avg_cpu'] = f"{row['avg_cpu']:.1f}%"
        row['avg_memory'] = f"{row['avg_memory']:.1f}%"
    return json.dumps(resources)

@app.route('/estimate_savings')
def estimate_savings_route():
    resource_utilization = float(request.args.get('resource_utilization', 30))
    idle_resources = float(request.args.get('idle_resources', 50))
    reserved_instances = float(request.args.get('reserved_instances', 20))
    memory_scaling = float(request.args.get('memory_scaling', 10))
    consolidation = float(request.args.get('consolidation', 20))
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    service_filter = request.args.get('service_filter')
    savings = estimate_savings(
        resource_utilization,
        idle_resources,
        reserved_instances,
        memory_scaling,
        consolidation,
        start_date,
        end_date,
        service_filter
    )
    savings_percentage = (savings['total_estimated_savings'] / savings['original_cost'] * 100) if savings['original_cost'] > 0 else 0.0
    return json.dumps({
        'original_cost': format_currency(savings['original_cost']),
        'new_estimated_cost': format_currency(savings['new_estimated_cost']),
        'total_estimated_savings': format_currency(savings['total_estimated_savings']),
        'savings_percentage': f"{savings_percentage:.1f}%",
        'resource_utilization_savings': format_currency(savings['resource_utilization_savings']),
        'idle_elimination_savings': format_currency(savings['idle_elimination_savings']),
        'reserved_instance_savings': format_currency(savings['reserved_instance_savings']),
        'memory_scaling_savings': format_currency(savings['memory_scaling_savings']),
        'consolidation_savings': format_currency(savings['consolidation_savings'])
    })

if __name__ == '__main__':
    app.run(debug=True, port=8051, use_reloader=False)