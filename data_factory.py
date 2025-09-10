"""
THE HUGO BOSS DATA FACTORY
This script generates hyper-realistic IT operations data
that mimics what Hugo Boss would actually see
"""

import pandas as pd
import numpy as np
from faker import Faker
from datetime import datetime, timedelta
import random
import json

fake = Faker('de_DE')  # German locale for authenticity

class HugoBossDataFactory:
    """
    This is our factory. Instead of manufacturing clothes,
    we're manufacturing data that looks EXACTLY like what 
    Hugo Boss IT department would see every day.
    """
    
    def __init__(self):
        # Real Hugo Boss locations
        self.locations = {
            'HQ': 'Metzingen Headquarters',
            'DC1': 'Stuttgart Distribution Center',
            'DC2': 'Filderstadt Logistics Hub',
            'R1': 'Munich Flagship Store',
            'R2': 'Berlin Kurfürstendamm Store',
            'R3': 'Hamburg Neuer Wall Store',
            'O1': 'Frankfurt Office',
            'O2': 'Düsseldorf Showroom'
        }
        
        # Real departments in a fashion company
        self.departments = [
            'Creative Design', 'Pattern Making', 'Product Development',
            'Supply Chain', 'Retail Operations', 'E-Commerce',
            'Finance', 'HR', 'Marketing', 'Visual Merchandising'
        ]
        
        # Real IT issues that happen in fashion retail
        self.issue_templates = {
            'VDI_FREEZE': {
                'description': 'VDI session frozen while working on {software}',
                'software': ['Adobe Illustrator', 'CLO 3D', 'Browzwear', 'PLM System'],
                'category': 'VDI',
                'base_priority': 'High',
                'resolution_pattern': 'Increased VDI memory allocation from 8GB to 16GB'
            },
            'SAP_ERROR': {
                'description': 'Cannot access SAP Fashion Management - Error {code}',
                'code': ['RFC_ERROR_SYSTEM_FAILURE', 'TIME_OUT', 'NO_AUTHORIZATION'],
                'category': 'ERP',
                'base_priority': 'Critical',
                'resolution_pattern': 'Restarted SAP connector service, cleared user buffer'
            },
            'POS_DOWN': {
                'description': 'POS terminal {terminal} offline - customers waiting',
                'terminal': ['POS-001', 'POS-002', 'POS-003', 'MOBILE-POS-01'],
                'category': 'Retail Systems',
                'base_priority': 'Critical',
                'resolution_pattern': 'Rebooted terminal, restored network connectivity'
            },
            'DESIGN_SLOW': {
                'description': 'Extreme lag in {software} when opening large design files',
                'software': ['Photoshop', 'InDesign', 'Kaledo', 'Optitex'],
                'category': 'Performance',
                'base_priority': 'Medium',
                'resolution_pattern': 'Optimized GPU settings, cleared cache'
            },
            'EMAIL_SYNC': {
                'description': 'Outlook not syncing - missing emails from {source}',
                'source': ['suppliers', 'retail stores', 'management', 'customers'],
                'category': 'Communication',
                'base_priority': 'Medium',
                'resolution_pattern': 'Recreated Outlook profile, fixed autodiscover'
            }
        }
    
    def generate_realistic_ticket(self, ticket_id, base_time):
        """
        Generate a single, ultra-realistic ticket
        """
        # Pick a random issue template
        template_key = random.choice(list(self.issue_templates.keys()))
        template = self.issue_templates[template_key]
        
        # Generate realistic description
        description = template['description']
        for key, values in template.items():
            if key in ['software', 'code', 'terminal', 'source']:
                description = description.replace(f'{{{key}}}', random.choice(values))
        
        # Time patterns (fashion retail is busiest 10am-7pm)
        hour = base_time.hour
        if 10 <= hour <= 19:  # Peak hours
            priority_boost = random.choice([True, True, False])  # 66% chance of higher priority
        else:
            priority_boost = False
        
        # Determine priority
        priority = template['base_priority']
        if priority_boost and priority == 'Medium':
            priority = 'High'
        
        # Resolution time based on priority (realistic SLA)
        if priority == 'Critical':
            resolution_hours = np.random.gamma(1.5, 0.5)  # Most resolved in 0.5-2 hours
        elif priority == 'High':
            resolution_hours = np.random.gamma(2, 1)  # 1-4 hours
        else:
            resolution_hours = np.random.gamma(4, 2)  # 2-12 hours
        
        # User details (German names for authenticity)
        user_name = fake.name()
        user_email = user_name.lower().replace(' ', '.') + '@hugoboss.com'
        
        return {
            'ticket_id': f'HBIT-{ticket_id}',
            'created_at': base_time,
            'created_by': user_name,
            'email': user_email,
            'location': random.choice(list(self.locations.values())),
            'department': random.choice(self.departments),
            'category': template['category'],
            'priority': priority,
            'description': description,
            'resolution_time_hours': max(0.25, resolution_hours),  # Minimum 15 minutes
            'resolved_at': base_time + timedelta(hours=resolution_hours),
            'resolution_notes': template['resolution_pattern'],
            'satisfaction_score': np.random.choice([3, 4, 4, 5, 5], p=[0.1, 0.2, 0.3, 0.3, 0.1])
        }
    
    def generate_vdi_metrics(self, timestamp):
        """
        Generate realistic VDI performance metrics
        """
        hour = timestamp.hour
        day = timestamp.weekday()
        
        # Fashion industry patterns
        if day in [5, 6]:  # Weekend
            base_load = 20
        elif hour < 7 or hour > 20:  # Off hours
            base_load = 30
        elif 9 <= hour <= 11 or 14 <= hour <= 16:  # Design review times
            base_load = 85
        else:
            base_load = 60
        
        # Add realistic noise
        cpu = base_load + np.random.normal(0, 10)
        memory = base_load + np.random.normal(5, 8)
        
        # Occasional spikes (5% chance)
        if random.random() < 0.05:
            cpu = min(100, cpu + 30)
            memory = min(100, memory + 25)
        
        return {
            'timestamp': timestamp,
            'server': f'VDI-PROD-{random.randint(1,5):02d}',
            'cpu_usage': max(5, min(100, cpu)),
            'memory_usage': max(10, min(100, memory)),
            'gpu_usage': max(0, min(100, np.random.beta(2, 5) * 100)) if hour in range(9, 18) else 0,
            'active_sessions': int(base_load * 2 + np.random.normal(0, 5)),
            'disk_iops': int(base_load * 10 + np.random.normal(0, 50)),
            'network_mbps': base_load * 0.5 + np.random.normal(0, 2),
            'location': random.choice(list(self.locations.keys()))
        }
    
    def manufacture_data(self):
        """
        The main factory production line
        """
        print("🏭 HUGO BOSS Data Factory Starting Production...")
        print("=" * 50)
        
        # Generate 90 days of tickets
        tickets = []
        start_date = datetime.now() - timedelta(days=90)
        
        for i in range(2000):
            # Spread tickets across 90 days with realistic patterns
            days_offset = random.random() ** 2 * 90  # More recent tickets
            hours_offset = np.random.normal(14, 4)  # Peak around 2pm
            
            ticket_time = start_date + timedelta(days=days_offset, hours=hours_offset)
            ticket = self.generate_realistic_ticket(10000 + i, ticket_time)
            tickets.append(ticket)
            
            if i % 100 == 0:
                print(f"📋 Generated {i} tickets...")
        
        tickets_df = pd.DataFrame(tickets)
        
        # Generate VDI metrics (every 5 minutes for last 7 days)
        vdi_metrics = []
        current_time = datetime.now()
        
        for minutes_ago in range(0, 7*24*60, 5):  # Every 5 minutes
            timestamp = current_time - timedelta(minutes=minutes_ago)
            metric = self.generate_vdi_metrics(timestamp)
            vdi_metrics.append(metric)
            
            if len(vdi_metrics) % 500 == 0:
                print(f"📊 Generated {len(vdi_metrics)} VDI metrics...")
        
        vdi_df = pd.DataFrame(vdi_metrics)
        
        # Save to CSV
        tickets_df.to_csv('hugo_boss_tickets.csv', index=False)
        vdi_df.to_csv('hugo_boss_vdi_metrics.csv', index=False)
        
        print("=" * 50)
        print("✅ Data Manufacturing Complete!")
        print(f"📁 Created: hugo_boss_tickets.csv ({len(tickets_df)} records)")
        print(f"📁 Created: hugo_boss_vdi_metrics.csv ({len(vdi_df)} records)")
        
        return tickets_df, vdi_df

# Run the factory
if __name__ == "__main__":
    factory = HugoBossDataFactory()
    tickets, vdi = factory.manufacture_data()
    
    # Show sample data
    print("\n📋 Sample Ticket:")
    print(tickets.iloc[0].to_dict())
    
    print("\n📊 Sample VDI Metric:")
    print(vdi.iloc[0].to_dict())