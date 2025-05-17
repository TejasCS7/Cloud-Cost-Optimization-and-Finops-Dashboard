import json
import psycopg2
from psycopg2.extras import RealDictCursor
from datetime import datetime, timedelta
import logging
import math
import statistics

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('cost_optimization.log'), logging.StreamHandler()]
)
logger = logging.getLogger('cost_optimizer')

class CloudCostOptimizer:
    def __init__(self, db_config):
        """
        Initialize the CloudCostOptimizer with database configuration
        """
        self.db_config = db_config
        logger.info("Cost optimizer initialized")
        
        # Dynamic thresholds for idle detection
        self._initialize_dynamic_thresholds()
    
    def _initialize_dynamic_thresholds(self):
        """
        Initialize dynamic thresholds for idle detection based on dataset statistics
        """
        try:
            query = """
            SELECT cpu_utilization, memory_utilization
            FROM fact_billing
            WHERE cpu_utilization IS NOT NULL AND memory_utilization IS NOT NULL
            """
            data = self.execute_query(query)
            
            if data:
                cpu_values = [float(row['cpu_utilization']) for row in data]
                memory_values = [float(row['memory_utilization']) for row in data]
                
                cpu_mean = statistics.mean(cpu_values) if cpu_values else 15
                cpu_stdev = statistics.stdev(cpu_values) if len(cpu_values) > 1 else 5
                memory_mean = statistics.mean(memory_values) if memory_values else 15
                memory_stdev = statistics.stdev(memory_values) if len(memory_values) > 1 else 5
                
                self.idle_cpu_threshold = max(5, cpu_mean - cpu_stdev)
                self.idle_memory_threshold = max(5, memory_mean - memory_stdev)
                
                logger.info(f"Dynamic thresholds initialized: CPU {self.idle_cpu_threshold:.2f}%, Memory {self.idle_memory_threshold:.2f}%")
            else:
                self.idle_cpu_threshold = 15
                self.idle_memory_threshold = 15
                logger.warning("No utilization data found, using default thresholds")
        except Exception as e:
            logger.error(f"Failed to initialize dynamic thresholds: {str(e)}")
            self.idle_cpu_threshold = 15
            self.idle_memory_threshold = 15
    
    def execute_query(self, query, params=None):
        """
        Execute SQL query and return results as a list of dictionaries
        """
        conn = None
        try:
            conn = psycopg2.connect(**self.db_config, cursor_factory=RealDictCursor)
            cursor = conn.cursor()
            cursor.execute(query, params)
            results = cursor.fetchall()
            return [dict(row) for row in results]
        except Exception as e:
            logger.error(f"Query execution failed: {str(e)}")
            raise
        finally:
            if conn:
                cursor.close()
                conn.close()
    
    def get_resource_utilization(self):
        """
        Get resource utilization metrics from the database
        """
        query = """
        SELECT 
            r.resource_id,
            s.service_name,
            reg.region_name,
            fb.cpu_utilization,
            fb.memory_utilization,
            fb.unrounded_cost,
            fb.usage_duration_hours,
            fb.usage_start_date
        FROM fact_billing fb
        JOIN dim_resources r ON fb.resource_id = r.resource_id
        JOIN dim_services s ON fb.service_id = s.service_id
        JOIN dim_regions reg ON fb.region_id = reg.region_id
        """
        
        data = self.execute_query(query)
        
        utilization = {}
        for row in data:
            key = (row['resource_id'], row['service_name'], row['region_name'])
            if key not in utilization:
                utilization[key] = {
                    'cpu': [], 'memory': [], 'cost': 0, 'hours': 0, 'days': set()
                }
            if row['cpu_utilization'] is not None:
                utilization[key]['cpu'].append(float(row['cpu_utilization']))
            if row['memory_utilization'] is not None:
                utilization[key]['memory'].append(float(row['memory_utilization']))
            utilization[key]['cost'] += float(row['unrounded_cost'])
            utilization[key]['hours'] += float(row['usage_duration_hours'])
            utilization[key]['days'].add(row['usage_start_date'].isoformat())
        
        result = []
        for (resource_id, service_name, region_name), metrics in utilization.items():
            avg_cpu = sum(metrics['cpu']) / len(metrics['cpu']) if metrics['cpu'] else 0
            avg_memory = sum(metrics['memory']) / len(metrics['memory']) if metrics['memory'] else 0
            result.append({
                'resource_id': resource_id,
                'service_name': service_name,
                'region_name': region_name,
                'avg_cpu': avg_cpu,
                'avg_memory': avg_memory,
                'total_cost': metrics['cost'],
                'total_hours': metrics['hours'],
                'active_days': len(metrics['days'])
            })
        
        logger.info(f"Retrieved utilization data for {len(result)} resources")
        return sorted(result, key=lambda x: x['total_cost'], reverse=True)
    
    def identify_idle_resources(self, threshold=None):
        cpu_threshold = self.idle_cpu_threshold if threshold is None else threshold
        memory_threshold = self.idle_memory_threshold if threshold is None else threshold
        utilization_data = self.get_resource_utilization()
        idle_resources = []
        total_potential_savings = 0
        
        for resource in utilization_data:
            if (resource['avg_cpu'] < cpu_threshold and 
                resource['avg_memory'] < memory_threshold and 
                resource['total_cost'] > 100):
                potential_savings = min(resource['total_cost'] * 0.8, resource['total_cost'])
                resource['potential_savings'] = potential_savings
                idle_resources.append(resource)
                total_potential_savings += potential_savings
        
        logger.info(f"Identified {len(idle_resources)} idle resources with potential savings of ${total_potential_savings:.2f}")
        return idle_resources
    
    def identify_rightsizing_opportunities(self):
        """
        Identify resources that could be downsized
        """
        query = """
        SELECT 
            r.resource_id,
            s.service_name,
            reg.region_name,
            fb.cpu_utilization,
            fb.memory_utilization,
            fb.unrounded_cost
        FROM fact_billing fb
        JOIN dim_resources r ON fb.resource_id = r.resource_id
        JOIN dim_services s ON fb.service_id = s.service_id
        JOIN dim_regions reg ON fb.region_id = reg.region_id
        """
        
        data = self.execute_query(query)
        utilization = {}
        for row in data:
            key = (row['resource_id'], row['service_name'], row['region_name'])
            if key not in utilization:
                utilization[key] = {'cpu': [], 'memory': [], 'cost': 0}
            if row['cpu_utilization'] is not None:
                utilization[key]['cpu'].append(float(row['cpu_utilization']))
            if row['memory_utilization'] is not None:
                utilization[key]['memory'].append(float(row['memory_utilization']))
            utilization[key]['cost'] += float(row['unrounded_cost'])
        
        rightsizing = []
        for (resource_id, service_name, region_name), metrics in utilization.items():
            max_cpu = max(metrics['cpu']) if metrics['cpu'] else 0
            max_memory = max(metrics['memory']) if metrics['memory'] else 0
            total_cost = metrics['cost']
            if max_cpu < 50 and max_memory < 60:
                potential_savings = min(total_cost * 0.4, total_cost)
                rightsizing.append({
                    'resource_id': resource_id,
                    'service_name': service_name,
                    'region_name': region_name,
                    'max_cpu': max_cpu,
                    'max_memory': max_memory,
                    'total_cost': total_cost,
                    'potential_savings': potential_savings,
                    'recommendation': "Rightsize this resource"
                })
        total_savings = sum(r['potential_savings'] for r in rightsizing)
        logger.info(f"Identified {len(rightsizing)} rightsizing opportunities with potential savings of ${total_savings:.2f}")
        return rightsizing
    
    def analyze_regional_cost_distribution(self):
        """
        Analyze cost distribution across regions
        """
        query = """
        SELECT 
            reg.region_name,
            fb.unrounded_cost,
            fb.resource_id
        FROM fact_billing fb
        JOIN dim_regions reg ON fb.region_id = reg.region_id
        """
        
        data = self.execute_query(query)
        
        regions = {}
        for row in data:
            region = row['region_name']
            if region not in regions:
                regions[region] = {'cost': 0, 'resources': set()}
            regions[region]['cost'] += float(row['unrounded_cost'])
            regions[region]['resources'].add(row['resource_id'])
        
        result = [{'region_name': k, 'total_cost': v['cost'], 'resource_count': len(v['resources'])} 
                 for k, v in regions.items()]
        total_cost = sum(r['total_cost'] for r in result)
        
        for region in result:
            region['cost_percentage'] = (region['total_cost'] / total_cost) * 100 if total_cost > 0 else 0
        
        logger.info(f"Analyzed cost distribution across {len(result)} regions")
        return sorted(result, key=lambda x: x['total_cost'], reverse=True)
    
    def find_cost_anomalies(self):
        """
        Detect anomalous spending patterns
        """
        query = """
        SELECT 
            d.date_id as date,
            s.service_name,
            fb.unrounded_cost
        FROM fact_billing fb
        JOIN dim_services s ON fb.service_id = s.service_id
        JOIN dim_dates d ON fb.usage_start_date = d.date_id
        """
        
        data = self.execute_query(query)
        
        daily_costs = {}
        for row in data:
            key = (row['date'].isoformat(), row['service_name'])
            if key not in daily_costs:
                daily_costs[key] = 0
            daily_costs[key] += float(row['unrounded_cost'])
        
        service_stats = {}
        for (date, service), cost in daily_costs.items():
            if service not in service_stats:
                service_stats[service] = []
            service_stats[service].append(cost)
        
        for service in service_stats:
            costs = service_stats[service]
            n = len(costs)
            if n > 0:
                mean = sum(costs) / n
                variance = sum((x - mean) ** 2 for x in costs) / n if n > 1 else 0
                stddev = math.sqrt(variance)
                service_stats[service] = {'avg_cost': mean, 'stddev_cost': stddev, 'costs': costs}
            else:
                service_stats[service] = {'avg_cost': 0, 'stddev_cost': 0, 'costs': []}
        
        anomalies = []
        for (date, service), daily_cost in daily_costs.items():
            stats = service_stats[service]
            z_score = (daily_cost - stats['avg_cost']) / stats['stddev_cost'] if stats['stddev_cost'] > 0 else 0
            if abs(z_score) > 2 and daily_cost > 10:
                anomalies.append({
                    'date': date,
                    'service_name': service,
                    'daily_cost': daily_cost,
                    'avg_cost': stats['avg_cost'],
                    'stddev_cost': stats['stddev_cost'],
                    'z_score': z_score
                })
        
        logger.info(f"Detected {len(anomalies)} cost anomalies")
        return sorted(anomalies, key=lambda x: abs(x['z_score']), reverse=True)
    
    def generate_optimization_recommendations(self):
        """
        Generate optimization recommendations
        """
        recommendations = []
        utilization_data = self.get_resource_utilization()
        avg_cost_query = "SELECT AVG(unrounded_cost) * 1.5 AS avg_cost FROM fact_billing"
        avg_cost_result = self.execute_query(avg_cost_query)
        avg_cost_threshold = float(avg_cost_result[0]['avg_cost'] or 0)

        for resource in utilization_data:
            total_cost = resource['total_cost']
            avg_cpu = resource['avg_cpu']
            avg_memory = resource['avg_memory']
            active_days = resource['active_days']
            
            if avg_cpu < 10:
                savings = total_cost * 0.5
                desc = f"Stop idle resource (low CPU usage: {avg_cpu:.1f}%). Impact: Minimal disruption."
            elif avg_memory > 80:
                savings = total_cost * 0.1
                desc = f"Scale up memory to prevent bottlenecks (Memory: {avg_memory:.1f}%). Impact: Improved performance."
            elif total_cost > avg_cost_threshold:
                savings = total_cost * 0.3
                desc = "Switch to reserved instances for cost efficiency. Impact: Long-term savings."
            elif active_days < 10:
                savings = total_cost * 0.4
                desc = "Consolidate underutilized resources. Impact: Reduced overhead."
            elif 10 <= avg_cpu <= 30:
                savings = total_cost * 0.25
                desc = f"Right-size instance to match workload (CPU: {avg_cpu:.1f}%). Impact: Cost reduction."
            else:
                savings = total_cost * 0.15
                desc = "Review usage patterns for optimization. Impact: Potential savings."

            if savings >= 10:
                recommendations.append({
                    'resource_id': resource['resource_id'],
                    'recommendation_type': 'Cost Optimization',
                    'recommendation_description': desc,
                    'potential_savings': savings
                })

        self.save_recommendations(recommendations)
        recommendations.sort(key=lambda x: x['potential_savings'], reverse=True)
        logger.info(f"Generated {len(recommendations)} optimization recommendations")
        return recommendations
    
    def save_recommendations(self, recommendations):
        """
        Save recommendations to database
        """
        if not recommendations:
            return
        
        resource_recommendations = [r for r in recommendations if r['resource_id'] != 'N/A']
        if not resource_recommendations:
            return
        
        conn = None
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            
            for rec in resource_recommendations:
                cursor.execute("""
                    INSERT INTO optimization_recommendations (resource_id, recommendation_type, recommendation_description, potential_savings)
                    VALUES (%s, %s, %s, %s)
                """, (
                    rec['resource_id'],
                    rec['recommendation_type'],
                    rec['recommendation_description'],
                    rec['potential_savings']
                ))
            
            conn.commit()
            logger.info(f"Saved {len(resource_recommendations)} recommendations to database")
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Failed to save recommendations: {str(e)}")
            raise
        finally:
            if conn:
                conn.close()
    
    def calculate_total_savings_potential(self):
        """
        Calculate total potential savings
        """
        recommendations = self.generate_optimization_recommendations()
        total_savings = sum(rec['potential_savings'] for rec in recommendations)
        
        query = "SELECT SUM(unrounded_cost) as total_cost FROM fact_billing"
        result = self.execute_query(query)
        total_cost = float(result[0]['total_cost']) if result and result[0]['total_cost'] is not None else 0
        
        savings_percentage = (total_savings / total_cost) * 100 if total_cost > 0 else 0
        
        result = {
            'total_cost': total_cost,
            'potential_savings': total_savings,
            'savings_percentage': savings_percentage,
            'recommendation_count': len(recommendations)
        }
        
        logger.info(f"Calculated total savings potential: ${total_savings:.2f} ({savings_percentage:.1f}%)")
        return result
    
    def generate_executive_summary(self):
        """
        Generate executive summary
        """
        savings_data = self.calculate_total_savings_potential()
        
        query = """
        SELECT 
            s.service_name,
            fb.unrounded_cost,
            fb.resource_id
        FROM fact_billing fb
        JOIN dim_services s ON fb.service_id = s.service_id
        """
        data = self.execute_query(query)
        
        services = {}
        for row in data:
            service = row['service_name']
            if service not in services:
                services[service] = {'cost': 0, 'resources': set()}
            services[service]['cost'] += float(row['unrounded_cost'])
            services[service]['resources'].add(row['resource_id'])
        
        top_services = [
            {'service_name': k, 'total_cost': v['cost'], 'resource_count': len(v['resources'])}
            for k, v in sorted(services.items(), key=lambda x: x[1]['cost'], reverse=True)[:5]
        ]
        
        query = """
        SELECT 
            fb.unrounded_cost,
            fb.usage_start_date
        FROM fact_billing fb
        JOIN dim_dates d ON fb.usage_start_date = d.date_id
        """
        data = self.execute_query(query)
        
        monthly_costs = {}
        for row in data:
            month = row['usage_start_date'].strftime('%Y-%m')
            if month not in monthly_costs:
                monthly_costs[month] = 0
            monthly_costs[month] += float(row['unrounded_cost'])
        
        monthly_trend = [{'month': k, 'monthly_cost': v} 
                        for k, v in sorted(monthly_costs.items())]
        
        summary = {
            'total_cost': savings_data['total_cost'],
            'potential_savings': savings_data['potential_savings'],
            'savings_percentage': savings_data['savings_percentage'],
            'recommendation_count': savings_data['recommendation_count'],
            'top_services': top_services,
            'monthly_trend': monthly_trend
        }
        
        logger.info("Generated executive summary")
        return summary
    
    def predict_next_month_cost(self):
        """
        Predict next month's costs using linear regression
        """
        query = """
        SELECT 
            TO_CHAR(fb.usage_start_date, 'YYYY-MM') as month,
            SUM(fb.unrounded_cost) as monthly_cost
        FROM fact_billing fb
        GROUP BY TO_CHAR(fb.usage_start_date, 'YYYY-MM')
        ORDER BY month
        """
        
        try:
            data = self.execute_query(query)
            
            if len(data) < 2:
                logger.warning("Not enough historical data for prediction")
                return None
            
            months = []
            costs = []
            
            for i, row in enumerate(data):
                months.append(i+1)
                costs.append(float(row['monthly_cost']))
            
            n = len(months)
            sum_x = sum(months)
            sum_y = sum(costs)
            sum_xy = sum(x*y for x, y in zip(months, costs))
            sum_xx = sum(x*x for x in months)
            
            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x)
            intercept = (sum_y - slope * sum_x) / n
            
            next_month = n + 1
            predicted_cost = slope * next_month + intercept
            
            average_cost = sum_y / n
            last_cost = costs[-1]
            last_month_name = data[-1]['month']
            trend_percentage = ((predicted_cost / last_cost) - 1) * 100
            
            month_over_month_changes = []
            for i in range(1, len(costs)):
                if costs[i-1] > 0:
                    month_over_month_changes.append((costs[i] / costs[i-1]) - 1)
            
            avg_growth_rate = sum(month_over_month_changes) / len(month_over_month_changes) if month_over_month_changes else 0
            
            result = {
                'last_month': last_month_name,
                'last_month_cost': last_cost,
                'predicted_next_month_cost': max(0, predicted_cost),
                'trend_percentage': trend_percentage,
                'average_monthly_growth_rate': avg_growth_rate * 100,
                'prediction_confidence': 'medium' if len(data) >= 3 else 'low'
            }
            
            logger.warning("Predictive model uses simple linear regression, which may not capture complex patterns")
            logger.info(f"Predicted next month's cost: ${result['predicted_next_month_cost']:.2f} ({result['trend_percentage']:.1f}% change)")
            return result
            
        except Exception as e:
            logger.error(f"Failed to predict costs: {str(e)}")
            return None
    
    def benchmark_cost_comparison(self):
        """
        Compare costs against industry benchmarks
        """
        industry_benchmarks = {
            'Cloud Compute': {'cost_per_cpu_hour': 0.03, 'cost_per_memory_gb_hour': 0.004},
            'Cloud Storage': {'cost_per_gb_month': 0.02},
            'Cloud Database': {'cost_per_instance_hour': 0.12},
            'Cloud Dataproc': {'cost_per_request': 0.0000035},
            'Cloud Networking': {'cost_per_gb_transfer': 0.08},
            'Default': {'avg_utilization_threshold': 60, 'idle_threshold': 15}
        }
        logger.warning("Using static industry benchmarks, which may not reflect current standards")
        
        try:
            query = """
            SELECT 
                s.service_name,
                AVG(fb.cpu_utilization) as avg_cpu,
                AVG(fb.memory_utilization) as avg_memory,
                SUM(fb.unrounded_cost) as total_cost,
                SUM(fb.usage_quantity) as total_usage,
                MAX(fb.usage_unit) as usage_unit
            FROM fact_billing fb
            JOIN dim_services s ON fb.service_id = s.service_id
            GROUP BY s.service_name
            """
            data = self.execute_query(query)
            
            benchmark_results = []
            for service in data:
                service_name = service['service_name']
                if 'Compute' in service_name:
                    benchmark_category = 'Cloud Compute'
                elif 'Storage' in service_name:
                    benchmark_category = 'Cloud Storage'
                elif 'Database' in service_name:
                    benchmark_category = 'Cloud Database'
                elif 'Dataproc' in service_name:
                    benchmark_category = 'Cloud Dataproc'
                elif 'Networking' in service_name:
                    benchmark_category = 'Cloud Networking'
                else:
                    benchmark_category = 'Default'
                
                benchmark = industry_benchmarks[benchmark_category]
                
                if 'Compute' in service_name:
                    usage_qty = float(service['total_usage']) if service['total_usage'] else 0
                    total_cost = float(service['total_cost']) if service['total_cost'] else 0
                    if usage_qty > 0:
                        actual_cost_per_unit = total_cost / usage_qty
                        benchmark_cost = benchmark['cost_per_cpu_hour']
                        cost_ratio = actual_cost_per_unit / benchmark_cost if benchmark_cost > 0 else 0
                        benchmark_results.append({
                            'service_name': service_name,
                            'metric': 'Cost per CPU hour',
                            'your_cost': actual_cost_per_unit,
                            'benchmark_cost': benchmark_cost,
                            'ratio_to_benchmark': cost_ratio,
                            'assessment': 'Above benchmark' if cost_ratio > 1.1 else 
                                        'Below benchmark' if cost_ratio < 0.9 else 'At benchmark',
                            'potential_savings': max(0, total_cost - (total_cost / cost_ratio)) if cost_ratio > 1 else 0
                        })
                elif 'Storage' in service_name:
                    usage_qty = float(service['total_usage']) if service['total_usage'] else 0
                    total_cost = float(service['total_cost']) if service['total_cost'] else 0
                    if usage_qty > 0:
                        actual_cost_per_unit = total_cost / usage_qty
                        benchmark_cost = benchmark['cost_per_gb_month']
                        cost_ratio = actual_cost_per_unit / benchmark_cost if benchmark_cost > 0 else 0
                        benchmark_results.append({
                            'service_name': service_name,
                            'metric': 'Cost per GB per month',
                            'your_cost': actual_cost_per_unit,
                            'benchmark_cost': benchmark_cost,
                            'ratio_to_benchmark': cost_ratio,
                            'assessment': 'Above benchmark' if cost_ratio > 1.1 else 
                                        'Below benchmark' if cost_ratio < 0.9 else 'At benchmark',
                            'potential_savings': max(0, total_cost - (total_cost / cost_ratio)) if cost_ratio > 1 else 0
                        })
                else:
                    avg_cpu = float(service['avg_cpu']) if service['avg_cpu'] is not None else 0
                    threshold = benchmark.get('avg_utilization_threshold', 60)
                    benchmark_results.append({
                        'service_name': service_name,
                        'metric': 'CPU Utilization',
                        'your_value': avg_cpu,
                        'benchmark_value': threshold,
                        'ratio_to_benchmark': avg_cpu / threshold if threshold > 0 else 0,
                        'assessment': 'Below optimal' if avg_cpu < threshold * 0.7 else 
                                    'Optimal' if avg_cpu <= threshold * 1.2 else 'Over-utilized',
                        'potential_savings': float(service['total_cost']) * 0.3 if avg_cpu < threshold * 0.7 else 0
                    })
            
            total_benchmark_savings = sum(b['potential_savings'] for b in benchmark_results if 'potential_savings' in b)
            logger.info(f"Completed benchmark comparison with {len(benchmark_results)} metrics, potential savings: ${total_benchmark_savings:.2f}")
            return {
                'benchmark_comparisons': benchmark_results,
                'total_potential_savings': total_benchmark_savings,
                'benchmark_date': datetime.now().strftime('%Y-%m-%d')
            }
        except Exception as e:
            logger.error(f"Failed to perform benchmark comparison: {str(e)}")
            return None
    
    def setup_cost_alerts(self):
        """
        Create threshold-based alerts for cost spikes
        """
        try:
            self._create_alerts_table()
            
            query = """
            SELECT 
                s.service_name,
                d.date_id::date as usage_date,
                SUM(fb.unrounded_cost) as daily_cost
            FROM fact_billing fb
            JOIN dim_services s ON fb.service_id = s.service_id
            JOIN dim_dates d ON fb.usage_start_date = d.date_id
            GROUP BY s.service_name, d.date_id::date
            ORDER BY s.service_name, d.date_id::date
            """
            
            data = self.execute_query(query)
            
            service_costs = {}
            for row in data:
                service = row['service_name']
                if service not in service_costs:
                    service_costs[service] = []
                service_costs[service].append({
                    'date': row['usage_date'],
                    'cost': float(row['daily_cost'])
                })
            
            alerts = []
            for service, costs in service_costs.items():
                if len(costs) < 7:
                    continue
                    
                daily_costs = [c['cost'] for c in costs]
                avg_cost = sum(daily_costs) / len(daily_costs)
                variance = sum((x - avg_cost) ** 2 for x in daily_costs) / len(daily_costs)
                stddev = math.sqrt(variance)
                
                regular_threshold = avg_cost + (2 * stddev)
                high_threshold = avg_cost + (3 * stddev)
                critical_threshold = avg_cost + (4 * stddev)
                
                alerts.append({
                    'service_name': service,
                    'avg_daily_cost': avg_cost,
                    'standard_deviation': stddev,
                    'medium_alert_threshold': regular_threshold,
                    'high_alert_threshold': high_threshold,
                    'critical_alert_threshold': critical_threshold,
                    'created_at': datetime.now()
                })
            
            self._save_alerts(alerts)
            
            logger.info(f"Set up {len(alerts)} cost alerts based on historical patterns")
            return alerts
            
        except Exception as e:
            logger.error(f"Failed to set up cost alerts: {str(e)}")
            return []
    
    def _create_alerts_table(self):
        """
        Create the cost_alerts table
        """
        conn = None
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    AND table_name = 'cost_alerts'
                );
            """)
            
            table_exists = cursor.fetchone()[0]
            
            if not table_exists:
                cursor.execute("""
                    CREATE TABLE cost_alerts (
                        alert_id SERIAL PRIMARY KEY,
                        service_name VARCHAR(100) NOT NULL,
                        avg_daily_cost NUMERIC(12,4) NOT NULL,
                        standard_deviation NUMERIC(12,4) NOT NULL,
                        medium_alert_threshold NUMERIC(12,4) NOT NULL,
                        high_alert_threshold NUMERIC(12,4) NOT NULL,
                        critical_alert_threshold NUMERIC(12,4) NOT NULL,
                        created_at TIMESTAMP NOT NULL,
                        active BOOLEAN DEFAULT TRUE
                    );
                """)
                
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS cost_alert_violations (
                        violation_id SERIAL PRIMARY KEY,
                        alert_id INTEGER REFERENCES cost_alerts(alert_id),
                        service_name VARCHAR(100) NOT NULL,
                        violation_date DATE NOT NULL,
                        actual_cost NUMERIC(12,4) NOT NULL,
                        expected_cost NUMERIC(12,4) NOT NULL,
                        threshold_exceeded VARCHAR(20) NOT NULL,
                        percentage_over NUMERIC(8,2) NOT NULL,
                        created_at TIMESTAMP NOT NULL,
                        resolved BOOLEAN DEFAULT FALSE
                    );
                """)
                
                conn.commit()
                logger.info("Created cost_alerts and cost_alert_violations tables")
            
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Failed to create alerts table: {str(e)}")
            raise
        finally:
            if conn:
                conn.close()
    
    def _save_alerts(self, alerts):
        """
        Save cost alerts to the database
        """
        if not alerts:
            return
            
        conn = None
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            
            cursor.execute("UPDATE cost_alerts SET active = FALSE WHERE active = TRUE;")
            
            for alert in alerts:
                cursor.execute("""
                    INSERT INTO cost_alerts 
                    (service_name, avg_daily_cost, standard_deviation, 
                     medium_alert_threshold, high_alert_threshold, critical_alert_threshold, 
                     created_at, active)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, TRUE)
                """, (
                    alert['service_name'], 
                    alert['avg_daily_cost'],
                    alert['standard_deviation'],
                    alert['medium_alert_threshold'],
                    alert['high_alert_threshold'],
                    alert['critical_alert_threshold'],
                    alert['created_at']
                ))
            
            conn.commit()
            logger.info(f"Saved {len(alerts)} cost alerts to database")
            
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Failed to save cost alerts: {str(e)}")
            raise
        finally:
            if conn:
                conn.close()
    
    def check_for_cost_violations(self, check_date=None):
        """
        Check for cost alert violations
        """
        if check_date is None:
            query = "SELECT MAX(usage_start_date) as latest_date FROM fact_billing"
            result = self.execute_query(query)
            if not result or not result[0]['latest_date']:
                logger.warning("No cost data found for violation checks")
                return []
            check_date = result[0]['latest_date']
        
        try:
            query = "SELECT * FROM cost_alerts WHERE active = TRUE"
            alerts = self.execute_query(query)
            
            if not alerts:
                logger.warning("No active cost alerts found")
                return []
            
            query = """
            SELECT 
                s.service_name,
                SUM(fb.unrounded_cost) as daily_cost
            FROM fact_billing fb
            JOIN dim_services s ON fb.service_id = s.service_id
            WHERE fb.usage_start_date = %s
            GROUP BY s.service_name
            """
            cost_data = self.execute_query(query, (check_date,))
            
            violations = []
            for daily_cost in cost_data:
                service = daily_cost['service_name']
                actual_cost = float(daily_cost['daily_cost'])
                
                matching_alerts = [a for a in alerts if a['service_name'] == service]
                if not matching_alerts:
                    continue
                
                alert = matching_alerts[0]
                expected_cost = float(alert['avg_daily_cost'])
                
                threshold_exceeded = None
                if actual_cost >= float(alert['critical_alert_threshold']):
                    threshold_exceeded = 'CRITICAL'
                elif actual_cost >= float(alert['high_alert_threshold']):
                    threshold_exceeded = 'HIGH'
                elif actual_cost >= float(alert['medium_alert_threshold']):
                    threshold_exceeded = 'MEDIUM'
                
                if threshold_exceeded:
                    percent_over = ((actual_cost / expected_cost) - 1) * 100 if expected_cost > 0 else 100
                    violations.append({
                        'alert_id': alert['alert_id'],
                        'service_name': service,
                        'violation_date': check_date,
                        'actual_cost': actual_cost,
                        'expected_cost': expected_cost,
                        'threshold_exceeded': threshold_exceeded,
                        'percentage_over': percent_over,
                        'created_at': datetime.now()
                    })
            
            if violations:
                self._save_violations(violations)
                
            logger.info(f"Checked for cost violations on {check_date}, found {len(violations)}")
            return violations
            
        except Exception as e:
            logger.error(f"Failed to check for cost violations: {str(e)}")
            return []
    
    def _save_violations(self, violations):
        """
        Save cost alert violations to the database
        """
        if not violations:
            return
            
        conn = None
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            
            for violation in violations:
                cursor.execute("""
                    INSERT INTO cost_alert_violations 
                    (alert_id, service_name, violation_date, actual_cost, expected_cost, 
                     threshold_exceeded, percentage_over, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    violation['alert_id'],
                    violation['service_name'],
                    violation['violation_date'],
                    violation['actual_cost'],
                    violation['expected_cost'],
                    violation['threshold_exceeded'],
                    violation['percentage_over'],
                    violation['created_at']
                ))
            
            conn.commit()
            logger.info(f"Saved {len(violations)} cost alert violations to database")
            
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Failed to save cost violations: {str(e)}")
            raise
        finally:
            if conn:
                conn.close()
    
    def setup_recommendation_tracking(self):
        """
        Set up tables to track recommendation implementation
        """
        conn = None
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    AND table_name = 'recommendation_tracking'
                );
            """)
            
            table_exists = cursor.fetchone()[0]
            
            if not table_exists:
                cursor.execute("""
                    CREATE TABLE recommendation_tracking (
                        tracking_id SERIAL PRIMARY KEY,
                        recommendation_id INTEGER REFERENCES optimization_recommendations(recommendation_id),
                        status VARCHAR(20) NOT NULL DEFAULT 'PENDING',
                        implemented_date DATE,
                        expected_savings NUMERIC(12,4),
                        actual_savings NUMERIC(12,4),
                        savings_accuracy NUMERIC(8,2),
                        notes TEXT,
                        last_updated TIMESTAMP NOT NULL
                    );
                """)
                
                conn.commit()
                logger.info("Created recommendation_tracking table")
                
                cursor.execute("""
                    INSERT INTO recommendation_tracking
                    (recommendation_id, status, expected_savings, last_updated)
                    SELECT recommendation_id, 'PENDING', potential_savings, NOW()
                    FROM optimization_recommendations
                    WHERE recommendation_id NOT IN (
                        SELECT recommendation_id FROM recommendation_tracking
                        WHERE recommendation_id IS NOT NULL
                    )
                """)
                
                conn.commit()
                logger.info("Initialized tracking for existing recommendations")
            
            return True
            
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Failed to set up recommendation tracking: {str(e)}")
            return False
        finally:
            if conn:
                conn.close()
    
    def update_recommendation_status(self, recommendation_id, status, implemented_date=None, notes=None):
        """
        Update recommendation status
        """
        valid_statuses = ['PENDING', 'IMPLEMENTED', 'REJECTED', 'SCHEDULED', 'IN_PROGRESS']
        if status not in valid_statuses:
            logger.error(f"Invalid recommendation status: {status}")
            return False
            
        conn = None
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT potential_savings
                FROM optimization_recommendations
                WHERE recommendation_id = %s
            """, (recommendation_id,))
            
            rec_data = cursor.fetchone()
            if not rec_data:
                logger.error(f"Recommendation ID {recommendation_id} not found")
                return False
                
            expected_savings = rec_data[0]
            
            if status == 'IMPLEMENTED':
                cursor.execute("""
                    UPDATE recommendation_tracking
                    SET status = %s, 
                        implemented_date = %s,
                        expected_savings = %s,
                        notes = %s,
                        last_updated = NOW()
                    WHERE recommendation_id = %s
                """, (status, implemented_date or datetime.now().date(), expected_savings, notes, recommendation_id))
            else:
                cursor.execute("""
                    UPDATE recommendation_tracking
                    SET status = %s, 
                        notes = %s,
                        last_updated = NOW()
                    WHERE recommendation_id = %s
                """, (status, notes, recommendation_id))
            
            conn.commit()
            logger.info(f"Updated recommendation {recommendation_id} status to {status}")
            return True
            
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Failed to update recommendation status: {str(e)}")
            return False
        finally:
            if conn:
                conn.close()
    
    def calculate_recommendation_roi(self, days_since_implementation=30):
        """
        Calculate savings and ROI for implemented recommendations
        """
        try:
            self.setup_recommendation_tracking()
            
            query = """
            SELECT 
                r.recommendation_id,
                r.resource_id,
                r.recommendation_type,
                t.implemented_date,
                t.expected_savings
            FROM optimization_recommendations r
            JOIN recommendation_tracking t ON r.recommendation_id = t.recommendation_id
            WHERE t.status = 'IMPLEMENTED'
                AND t.implemented_date IS NOT NULL
                AND t.implemented_date <= CURRENT_DATE - INTERVAL '%s days'
            """
            
            implemented_recs = self.execute_query(query, (days_since_implementation,))
            
            if not implemented_recs:
                logger.info("No implemented recommendations found to calculate ROI")
                return []
                
            results = []
            for rec in implemented_recs:
                if rec['resource_id'] == 'N/A':
                    continue
                    
                before_query = """
                SELECT AVG(daily_cost) as avg_cost
                FROM (
                    SELECT 
                        usage_start_date::date as day,
                        SUM(unrounded_cost) as daily_cost
                    FROM fact_billing
                    WHERE resource_id = %s
                        AND usage_start_date < %s
                        AND usage_start_date >= %s - INTERVAL '30 days'
                    GROUP BY usage_start_date::date
                ) as daily_costs
                """
                
                before_data = self.execute_query(before_query, (
                    rec['resource_id'], 
                    rec['implemented_date'],
                    rec['implemented_date']
                ))
                
                before_avg = float(before_data[0]['avg_cost']) if before_data and before_data[0]['avg_cost'] else 0
                
                after_query = """
                SELECT AVG(daily_cost) as avg_cost
                FROM (
                    SELECT 
                        usage_start_date::date as day,
                        SUM(unrounded_cost) as daily_cost
                    FROM fact_billing
                    WHERE resource_id = %s
                        AND usage_start_date >= %s
                        AND usage_start_date <= %s + INTERVAL '%s days'
                    GROUP BY usage_start_date::date
                ) as daily_costs
                """
                
                after_data = self.execute_query(after_query, (
                    rec['resource_id'], 
                    rec['implemented_date'],
                    rec['implemented_date'],
                    days_since_implementation
                ))
                
                after_avg = float(after_data[0]['avg_cost']) if after_data and after_data[0]['avg_cost'] else 0
                
                daily_savings = max(0, before_avg - after_avg)
                actual_savings = daily_savings * days_since_implementation
                expected_savings = float(rec['expected_savings']) if rec['expected_savings'] else 0
                
                accuracy = (actual_savings / expected_savings * 100) if expected_savings > 0 else 0
                
                result = {
                    'recommendation_id': rec['recommendation_id'],
                    'resource_id': rec['resource_id'],
                    'recommendation_type': rec['recommendation_type'],
                    'implemented_date': rec['implemented_date'],
                    'expected_savings': expected_savings,
                    'actual_savings': actual_savings,
                    'accuracy': accuracy,
                    'roi_period_days': days_since_implementation
                }
                
                results.append(result)
                
                conn = None
                try:
                    conn = psycopg2.connect(**self.db_config)
                    cursor = conn.cursor()
                    
                    cursor.execute("""
                        UPDATE recommendation_tracking
                        SET actual_savings = %s,
                            savings_accuracy = %s,
                            last_updated = NOW()
                        WHERE recommendation_id = %s
                    """, (actual_savings, accuracy, rec['recommendation_id']))
                    
                    conn.commit()
                except Exception as e:
                    if conn:
                        conn.rollback()
                    logger.error(f"Failed to update actual savings: {str(e)}")
                finally:
                    if conn:
                        conn.close()
            
            logger.info(f"Calculated ROI for {len(results)} implemented recommendations")
            return results
            
        except Exception as e:
            logger.error(f"Failed to calculate recommendation ROI: {str(e)}")
            return []
    
    def generate_implementation_impact_report(self):
        """
        Generate report on implemented recommendations
        """
        try:
            roi_data = self.calculate_recommendation_roi()
            
            query = """
            SELECT 
                status,
                COUNT(*) as count,
                SUM(expected_savings) as total_expected_savings
            FROM recommendation_tracking
            GROUP BY status
            """
            
            status_data = self.execute_query(query)
            
            status_summary = {}
            for row in status_data:
                status_summary[row['status']] = {
                    'count': row['count'],
                    'expected_savings': float(row['total_expected_savings']) if row['total_expected_savings'] else 0
                }
                
            implemented_count = status_summary.get('IMPLEMENTED', {}).get('count', 0)
            total_count = sum(s.get('count', 0) for s in status_summary.values())
            implemented_percentage = (implemented_count / total_count) * 100 if total_count > 0 else 0
            
            total_expected = sum(s.get('expected_savings', 0) for s in status_summary.values())
            total_actual = sum(r['actual_savings'] for r in roi_data)
            savings_realization = (total_actual / total_expected) * 100 if total_expected > 0 else 0
            
            report = {
                'status_summary': status_summary,
                'recommendation_count': total_count,
                'implemented_count': implemented_count,
                'implemented_percentage': implemented_percentage,
                'total_expected_savings': total_expected,
                'total_actual_savings': total_actual,
                'savings_realization_percentage': savings_realization,
                'implementation_details': roi_data,
                'report_date': datetime.now().strftime('%Y-%m-%d')
            }
            
            logger.info("Generated implementation impact report")
            return report
            
        except Exception as e:
            logger.error(f"Failed to generate implementation impact report: {str(e)}")
            return {}

def main():
    """
    Main function to run the cost optimization analysis
    """
    db_config = {
        'dbname': 'cloud_finops',
        'user': 'postgres',
        'password': '64823',
        'host': 'localhost',
        'port': '5432'
    }
    
    try:
        optimizer = CloudCostOptimizer(db_config)
        
        optimizer.setup_recommendation_tracking()
        optimizer._create_alerts_table()
        
        recommendations = optimizer.generate_optimization_recommendations()
        logger.info(f"Generated {len(recommendations)} cost optimization recommendations")
        
        savings_data = optimizer.calculate_total_savings_potential()
        logger.info(f"Potential savings: ${savings_data['potential_savings']:.2f} ({savings_data['savings_percentage']:.1f}%)")
        
        logger.info("=== New Features ===")
        
        next_month_prediction = optimizer.predict_next_month_cost()
        if next_month_prediction:
            logger.info(f"Predicted next month cost: ${next_month_prediction['predicted_next_month_cost']:.2f}")
            logger.info(f"Trend: {next_month_prediction['trend_percentage']:.1f}% change from previous month")
        
        logger.info(f"Using dynamic thresholds for idle detection - CPU: {optimizer.idle_cpu_threshold:.1f}%, Memory: {optimizer.idle_memory_threshold:.1f}%")
        
        benchmark_data = optimizer.benchmark_cost_comparison()
        if benchmark_data:
            logger.info(f"Benchmark comparison completed, identified ${benchmark_data['total_potential_savings']:.2f} in potential savings")
            
        alerts = optimizer.setup_cost_alerts()
        logger.info(f"Set up {len(alerts)} cost alerts based on historical patterns")
        
        violations = optimizer.check_for_cost_violations()
        logger.info(f"Found {len(violations)} cost alert violations")
        
        impact_report = optimizer.generate_implementation_impact_report()
        if impact_report:
            implemented = impact_report.get('implemented_count', 0)
            total = impact_report.get('recommendation_count', 0)
            logger.info(f"Implementation status: {implemented}/{total} recommendations implemented")
            logger.info(f"Actual savings: ${impact_report.get('total_actual_savings', 0):.2f}")
        
        summary = optimizer.generate_executive_summary()
        
        enhanced_summary = {
            **summary,
            'cost_prediction': next_month_prediction,
            'benchmark_data': benchmark_data,
            'alert_violations': len(violations),
            'implementation_impact': impact_report
        }
        
        with open('enhanced_executive_summary.json', 'w') as f:
            json.dump(enhanced_summary, f, indent=2)
        
        logger.info("Enhanced executive summary saved to enhanced_executive_summary.json")
        return recommendations, enhanced_summary
        
    except Exception as e:
        logger.error(f"Error during optimization analysis: {str(e)}")
        return None, None

if __name__ == "__main__":
    main()
