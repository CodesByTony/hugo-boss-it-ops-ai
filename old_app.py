"""
HUGO BOSS IT OPERATIONS INTELLIGENCE PLATFORM
All-in-one version that generates its own data and models
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import random
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="Hugo Boss IT Intelligence",
    page_icon="👔",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: #000000;
        color: white;
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Data Generation Functions
@st.cache_data
def generate_synthetic_data():
    """Generate Hugo Boss IT data on the fly"""
    
    # Ticket data
    tickets = []
    locations = ['Metzingen HQ', 'Stuttgart Office', 'Munich Store', 'Berlin Store', 'Hamburg Office']
    departments = ['Design', 'IT', 'Sales', 'Marketing', 'Finance', 'Supply Chain', 'HR']
    categories = ['VDI', 'Application', 'Email', 'Network', 'Hardware']
    priorities = ['Low', 'Medium', 'High', 'Critical']
    
    for i in range(500):  # Reduced for faster loading
        created = datetime.now() - timedelta(days=random.randint(0, 30), 
                                            hours=random.randint(0, 23))
        
        priority = random.choice(priorities)
        if priority == 'Critical':
            resolution_hours = random.uniform(0.5, 2)
        elif priority == 'High':
            resolution_hours = random.uniform(2, 6)
        elif priority == 'Medium':
            resolution_hours = random.uniform(4, 12)
        else:
            resolution_hours = random.uniform(8, 24)
        
        tickets.append({
            'ticket_id': f'HB-{10000+i}',
            'created_at': created,
            'location': random.choice(locations),
            'department': random.choice(departments),
            'category': random.choice(categories),
            'priority': priority,
            'description': f"Issue with {random.choice(categories)} system",
            'resolution_time_hours': resolution_hours,
            'resolved_at': created + timedelta(hours=resolution_hours),
            'satisfaction_score': random.choice([3, 4, 4, 5, 5])
        })
    
    tickets_df = pd.DataFrame(tickets)
    
    # VDI metrics
    vdi_metrics = []
    for i in range(200):  # Reduced for faster loading
        timestamp = datetime.now() - timedelta(minutes=i*5)
        
        hour = timestamp.hour
        if 8 <= hour <= 18:  # Business hours
            cpu_base = 60
            memory_base = 65
        else:
            cpu_base = 30
            memory_base = 35
        
        vdi_metrics.append({
            'timestamp': timestamp,
            'server': f'VDI-{random.randint(1,5)}',
            'cpu_usage': min(100, max(0, cpu_base + random.normalvariate(0, 15))),
            'memory_usage': min(100, max(0, memory_base + random.normalvariate(0, 12))),
            'active_sessions': random.randint(20, 150),
            'health_score': random.uniform(70, 100)
        })
    
    vdi_df = pd.DataFrame(vdi_metrics)
    
    return tickets_df, vdi_df

# Simple AI Models
class SimpleTicketAI:
    def __init__(self):
        self.priority_map = {
            'VDI': 'High',
            'Application': 'Medium',
            'Email': 'Low',
            'Network': 'Critical',
            'Hardware': 'High'
        }
    
    def predict(self, category, description):
        # Simple rule-based prediction
        priority = self.priority_map.get(category, 'Medium')
        
        # Check for urgent keywords
        urgent_words = ['urgent', 'critical', 'down', 'broken', 'emergency']
        if any(word in description.lower() for word in urgent_words):
            priority = 'Critical'
        
        # Estimate resolution time
        resolution_times = {
            'Critical': random.uniform(0.5, 2),
            'High': random.uniform(2, 6),
            'Medium': random.uniform(4, 12),
            'Low': random.uniform(8, 24)
        }
        
        return {
            'predicted_priority': priority,
            'estimated_hours': resolution_times[priority],
            'confidence': random.uniform(75, 95)
        }

class SimpleVDIMonitor:
    def analyze(self, cpu, memory):
        risk_score = 0
        issues = []
        recommendations = []
        
        if cpu > 90:
            risk_score += 50
            issues.append(f"CPU Critical: {cpu:.1f}%")
            recommendations.append("Migrate sessions immediately")
        elif cpu > 75:
            risk_score += 25
            issues.append(f"CPU Warning: {cpu:.1f}%")
            recommendations.append("Monitor closely")
        
        if memory > 85:
            risk_score += 30
            issues.append(f"Memory High: {memory:.1f}%")
            recommendations.append("Clear cache and restart services")
        
        status = "CRITICAL" if risk_score > 60 else "WARNING" if risk_score > 30 else "HEALTHY"
        
        return {
            'status': status,
            'risk_score': risk_score,
            'issues': issues,
            'recommendations': recommendations
        }

# Load data and models
tickets_df, vdi_df = generate_synthetic_data()
ticket_ai = SimpleTicketAI()
vdi_monitor = SimpleVDIMonitor()

# Header
st.markdown("""
<div class="main-header">
    <h1 style='margin:0;'>👔 HUGO BOSS</h1>
    <h2 style='margin:0; font-weight: 300;'>IT Operations Intelligence Platform</h2>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### Hugo Boss IT Ops")
    module = st.radio(
        "Navigation",
        ["🏠 Dashboard", "🎫 Tickets", "💻 VDI", "🤖 AI Predictions"]
    )
    
    st.markdown("---")
    st.success("✅ All Systems Operational")
    st.metric("Active Users", random.randint(245, 267))

# Main Content
if module == "🏠 Dashboard":
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        recent_tickets = len(tickets_df[tickets_df['created_at'] > datetime.now() - timedelta(days=7)])
        st.metric("Weekly Tickets", recent_tickets)
    
    with col2:
        avg_resolution = tickets_df['resolution_time_hours'].mean()
        st.metric("Avg Resolution", f"{avg_resolution:.1f}h")
    
    with col3:
        critical = len(tickets_df[tickets_df['priority'] == 'Critical'])
        st.metric("Critical Issues", critical)
    
    with col4:
        latest_cpu = vdi_df.iloc[0]['cpu_usage']
        st.metric("Current VDI Load", f"{latest_cpu:.0f}%")
    
    st.markdown("---")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Ticket Trends")
        daily = tickets_df.groupby(tickets_df['created_at'].dt.date).size().reset_index()
        daily.columns = ['Date', 'Count']
        fig = px.line(daily.tail(14), x='Date', y='Count', markers=True)
        fig.update_traces(line_color='#000000')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🎯 Priority Distribution")
        priority_dist = tickets_df['priority'].value_counts()
        fig = px.pie(values=priority_dist.values, names=priority_dist.index,
                    color_discrete_map={'Critical':'red', 'High':'orange', 
                                       'Medium':'yellow', 'Low':'green'})
        st.plotly_chart(fig, use_container_width=True)
    
    # Recent Activity
    st.subheader("📝 Recent Tickets")
    recent = tickets_df.head(5)[['ticket_id', 'priority', 'category', 'location', 'created_at']]
    st.dataframe(recent, use_container_width=True)

elif module == "🎫 Tickets":
    
    st.header("🎫 Intelligent Ticket Management")
    
    tab1, tab2 = st.tabs(["Create New", "Analytics"])
    
    with tab1:
        st.subheader("Create Ticket with AI Prediction")
        
        col1, col2 = st.columns(2)
        
        with col1:
            category = st.selectbox("Category", ['VDI', 'Application', 'Email', 'Network', 'Hardware'])
            location = st.selectbox("Location", ['Metzingen HQ', 'Munich Store', 'Berlin Store'])
        
        with col2:
            description = st.text_area("Description", 
                                      "VDI session frozen, urgent fix needed")
        
        if st.button("🤖 Get AI Prediction", type="primary"):
            prediction = ticket_ai.predict(category, description)
            
            st.success("✅ AI Analysis Complete!")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Priority", prediction['predicted_priority'])
            with col2:
                st.metric("Est. Resolution", f"{prediction['estimated_hours']:.1f}h")
            with col3:
                st.metric("Confidence", f"{prediction['confidence']:.1f}%")
            
            st.info("""
            **AI Recommendations:**
            - Assign to VDI Support Team
            - Check server resources
            - Review similar tickets #HB-10453, #HB-10421
            """)
    
    with tab2:
        st.subheader("Ticket Analytics")
        
        # Category distribution
        cat_dist = tickets_df['category'].value_counts()
        fig = px.bar(x=cat_dist.index, y=cat_dist.values,
                    labels={'x': 'Category', 'y': 'Count'})
        fig.update_traces(marker_color='#000000')
        st.plotly_chart(fig, use_container_width=True)

elif module == "💻 VDI":
    
    st.header("💻 VDI Monitoring")
    
    # Get latest metrics
    latest = vdi_df.iloc[0]
    analysis = vdi_monitor.analyze(latest['cpu_usage'], latest['memory_usage'])
    
    # Status
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if analysis['status'] == 'CRITICAL':
            st.error(f"Status: {analysis['status']}")
        elif analysis['status'] == 'WARNING':
            st.warning(f"Status: {analysis['status']}")
        else:
            st.success(f"Status: {analysis['status']}")
    
    with col2:
        st.metric("Risk Score", f"{analysis['risk_score']}/100")
    
    with col3:
        st.metric("Active Sessions", latest['active_sessions'])
    
    # Issues and Recommendations
    if analysis['issues']:
        st.markdown("### ⚠️ Issues Detected")
        for issue in analysis['issues']:
            st.error(issue)
    
    if analysis['recommendations']:
        st.markdown("### 💡 Recommendations")
        for rec in analysis['recommendations']:
            st.info(rec)
    
    # Performance Chart
    st.subheader("📈 Performance Trend")
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=vdi_df['timestamp'], y=vdi_df['cpu_usage'],
                            name='CPU %', line=dict(color='red')))
    fig.add_trace(go.Scatter(x=vdi_df['timestamp'], y=vdi_df['memory_usage'],
                            name='Memory %', line=dict(color='blue')))
    fig.add_hline(y=80, line_dash="dash", annotation_text="Warning Threshold")
    
    st.plotly_chart(fig, use_container_width=True)

elif module == "🤖 AI Predictions":
    
    st.header("🤖 AI Predictions")
    
    st.subheader("Next 24 Hours Forecast")
    
    # Generate predictions
    predictions = []
    for hour in range(24):
        time = datetime.now() + timedelta(hours=hour)
        
        # Predict based on hour
        if 8 <= time.hour <= 18:
            tickets = random.randint(5, 15)
            vdi_load = random.uniform(60, 85)
        else:
            tickets = random.randint(0, 5)
            vdi_load = random.uniform(20, 40)
        
        predictions.append({
            'Time': time.strftime('%H:00'),
            'Predicted Tickets': tickets,
            'VDI Load %': vdi_load
        })
    
    pred_df = pd.DataFrame(predictions)
    
    # Chart
    fig = go.Figure()
    fig.add_trace(go.Bar(x=pred_df['Time'], y=pred_df['Predicted Tickets'],
                        name='Tickets', marker_color='black'))
    fig.add_trace(go.Scatter(x=pred_df['Time'], y=pred_df['VDI Load %'],
                            name='VDI Load', yaxis='y2', line=dict(color='red')))
    
    fig.update_layout(
        yaxis=dict(title='Tickets'),
        yaxis2=dict(title='VDI Load %', overlaying='y', side='right')
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Key Insights
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("""
        **📈 Peak Hours**
        - 14:00-16:00: Highest ticket volume
        - 10:00-11:00: VDI load peaks
        """)
    
    with col2:
        st.warning("""
        **⚠️ Risk Windows**
        - 15:00: Potential VDI overload
        - Recommend: Pre-scale resources
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    Hugo Boss IT Operations Intelligence | Built with AI | © 2024
</div>
""", unsafe_allow_html=True)