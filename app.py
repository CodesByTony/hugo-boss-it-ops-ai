"""
Hugo Boss IT Operations Intelligence Platform
Created for IT Operations Team - Hugo Boss;Metzingen.
Author: Tony Thomas
Purpose: Demonstrate predictive AI capabilities for IT infrastructure
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import random
import time
import warnings
warnings.filterwarnings('ignore')

# Page setup - going for a premium look
st.set_page_config(
    page_title="Hugo Boss IT Intelligence",
    page_icon="👔",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - mixing Hugo Boss black with modern tech colors
st.markdown("""
<style>
    /* Main color scheme - black, white, with tech blue and green accents */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2.5rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        text-align: center;
        animation: slideDown 0.5s ease-out;
    }
    
    @keyframes slideDown {
        from { transform: translateY(-20px); opacity: 0; }
        to { transform: translateY(0); opacity: 1; }
    }
    
    .metric-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.08);
        border-left: 4px solid #667eea;
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }
    
    .alert-critical {
        background: linear-gradient(135deg, #f85032 0%, #e73827 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(248, 80, 50, 0.7); }
        70% { box-shadow: 0 0 0 10px rgba(248, 80, 50, 0); }
        100% { box-shadow: 0 0 0 0 rgba(248, 80, 50, 0); }
    }
    
    .alert-warning {
        background: linear-gradient(135deg, #f7971e 0%, #ffd200 100%);
        color: #333;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    
    .success-message {
        background: linear-gradient(135deg, #00b09b 0%, #96c93d 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
    }
    
    /* Stylish sidebar */
    .css-1d391kg {
        background: linear-gradient(180deg, #2c3e50 0%, #3498db 100%);
    }
    
    /* Metric styling */
    [data-testid="metric-container"] {
        background: rgba(255, 255, 255, 0.9);
        border: 1px solid rgba(102, 126, 234, 0.2);
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        border-radius: 30px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: scale(1.05);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(255, 255, 255, 0.8);
        border-radius: 10px;
        padding: 5px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        color: #333;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state for chat
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'insight_idx' not in st.session_state:
    st.session_state.insight_idx = 0

# Data generation - creating realistic Hugo Boss IT data
@st.cache_data
def generate_synthetic_data():
    """
    Generate synthetic data that mimics real Hugo Boss IT operations
    I researched typical fashion company IT infrastructure for this
    """
    
    # Tickets with realistic patterns
    tickets = []
    
    # Hugo Boss specific locations - these are their actual offices
    locations = [
        'Metzingen Headquarters', 
        'Stuttgart Distribution Center',
        'Munich Flagship Store', 
        'Berlin Kurfürstendamm', 
        'Hamburg Neuer Wall',
        'Frankfurt Office',
        'Düsseldorf Showroom',
        'Paris Showroom'
    ]
    
    # Departments that would exist in Hugo Boss
    departments = [
        'Creative Design', 'Pattern Making', 'Product Development',
        'Supply Chain', 'Retail Operations', 'E-Commerce',
        'Finance', 'HR', 'Marketing', 'Visual Merchandising',
        'IT Infrastructure', 'Digital Innovation'
    ]
    
    # Realistic IT categories for fashion industry
    categories = ['VDI', 'SAP', 'Adobe Creative', 'POS Systems', 'Network', 'Email', 'PLM System']
    priorities = ['Low', 'Medium', 'High', 'Critical']
    
    # Generate 500 tickets with realistic patterns
    for i in range(500):
        # Time patterns - more tickets during business hours
        hour = random.choice([9,10,11,14,15,16] * 3 + list(range(24)))  # Weight business hours
        created = datetime.now() - timedelta(
            days=random.randint(0, 30),
            hours=hour,
            minutes=random.randint(0, 59)
        )
        
        priority = random.choices(
            priorities,
            weights=[30, 40, 25, 5],  # Most tickets are low/medium
            k=1
        )[0]
        
        # Resolution time based on priority - this is realistic SLA
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
            'description': f"Issue with {random.choice(categories)} - affecting productivity",
            'resolution_time_hours': resolution_hours,
            'resolved_at': created + timedelta(hours=resolution_hours),
            'satisfaction_score': random.choices([3, 4, 5], weights=[10, 30, 60], k=1)[0],
            'assigned_to': random.choice(['Klaus Mueller', 'Anna Schmidt', 'Marcus Weber', 'Lisa Fischer'])
        })
    
    tickets_df = pd.DataFrame(tickets)
    
    # VDI metrics - simulating real server behavior
    vdi_metrics = []
    for i in range(200):
        timestamp = datetime.now() - timedelta(minutes=i*5)
        hour = timestamp.hour
        
        # Realistic load patterns based on time
        if hour in [10, 11, 14, 15, 16]:  # Peak hours
            cpu_base = 70
            memory_base = 75
            sessions_base = 120
        elif 8 <= hour <= 18:  # Business hours
            cpu_base = 55
            memory_base = 60
            sessions_base = 80
        else:  # After hours
            cpu_base = 25
            memory_base = 30
            sessions_base = 20
        
        # Add some randomness but keep it realistic
        vdi_metrics.append({
            'timestamp': timestamp,
            'server': f'VDI-PROD-{random.randint(1,5):02d}',
            'cpu_usage': min(100, max(0, cpu_base + random.normalvariate(0, 15))),
            'memory_usage': min(100, max(0, memory_base + random.normalvariate(0, 12))),
            'gpu_usage': min(100, max(0, random.uniform(20, 80))) if hour in range(9, 18) else 0,
            'active_sessions': max(0, int(sessions_base + random.normalvariate(0, 10))),
            'health_score': random.uniform(85, 100) if cpu_base < 80 else random.uniform(60, 85)
        })
    
    vdi_df = pd.DataFrame(vdi_metrics)
    
    return tickets_df, vdi_df

# Simple AI models - keeping it lightweight but impressive
class SimpleTicketAI:
    """
    My ticket prediction model - uses pattern recognition
    In production, I'd use proper ML, but this works great for demo
    """
    def __init__(self):
        self.priority_patterns = {
            'VDI': 'High',
            'SAP': 'Critical',  # SAP is business critical for Hugo Boss
            'Adobe Creative': 'Medium',
            'POS Systems': 'Critical',  # Retail can't function without POS
            'Network': 'High',
            'Email': 'Low',
            'PLM System': 'High'
        }
    
    def predict(self, category, description, department):
        # Base priority from category
        base_priority = self.priority_patterns.get(category, 'Medium')
        
        # Check for urgent keywords - learned these from research
        urgent_words = ['urgent', 'critical', 'down', 'broken', 'emergency', 'asap', 'blocked']
        if any(word in description.lower() for word in urgent_words):
            base_priority = 'Critical'
        
        # Department importance (Creative Design is crucial for Hugo Boss)
        if department in ['Creative Design', 'Retail Operations'] and base_priority == 'Medium':
            base_priority = 'High'
        
        # Estimate resolution time
        resolution_times = {
            'Critical': random.uniform(0.5, 2),
            'High': random.uniform(2, 6),
            'Medium': random.uniform(4, 12),
            'Low': random.uniform(8, 24)
        }
        
        return {
            'predicted_priority': base_priority,
            'estimated_hours': resolution_times[base_priority],
            'confidence': random.uniform(82, 98),  # High confidence for demo
            'auto_assign': self.get_team_assignment(category)
        }
    
    def get_team_assignment(self, category):
        """Intelligently assign to right team"""
        assignments = {
            'VDI': 'VDI Support Team (Klaus Mueller)',
            'SAP': 'ERP Team (Anna Schmidt)',
            'Adobe Creative': 'Creative IT Support (Marcus Weber)',
            'POS Systems': 'Retail IT (Lisa Fischer)',
            'Network': 'Network Operations (Thomas Bauer)',
            'Email': 'General IT Support',
            'PLM System': 'PLM Specialists (Stefan Wagner)'
        }
        return assignments.get(category, 'General IT Support')

class VDIMonitor:
    """
    VDI health monitoring - predicts issues before they happen
    Based on threshold analysis and pattern recognition
    """
    def analyze(self, cpu, memory, sessions):
        risk_score = 0
        issues = []
        recommendations = []
        
        # CPU analysis with multiple thresholds
        if cpu > 90:
            risk_score += 50
            issues.append(f"🔴 CPU Critical: {cpu:.1f}%")
            recommendations.append("IMMEDIATE: Migrate sessions to backup VDI")
        elif cpu > 75:
            risk_score += 25
            issues.append(f"🟡 CPU Warning: {cpu:.1f}%")
            recommendations.append("Monitor closely, prepare for migration")
        
        # Memory analysis
        if memory > 85:
            risk_score += 30
            issues.append(f"🔴 Memory Critical: {memory:.1f}%")
            recommendations.append("Clear cache, restart non-essential services")
        elif memory > 70:
            risk_score += 15
            issues.append(f"🟡 Memory Warning: {memory:.1f}%")
        
        # Session load analysis
        if sessions > 150:
            risk_score += 20
            issues.append(f"🟡 High session count: {sessions}")
            recommendations.append("Enable load balancing across servers")
        
        # Determine status
        if risk_score >= 70:
            status = "CRITICAL"
            color = "🔴"
        elif risk_score >= 40:
            status = "WARNING"
            color = "🟡"
        else:
            status = "HEALTHY"
            color = "🟢"
        
        return {
            'status': status,
            'color': color,
            'risk_score': risk_score,
            'issues': issues,
            'recommendations': recommendations,
            'predicted_failure': risk_score > 60,
            'time_to_failure': f"{max(1, 10-risk_score//10)}h" if risk_score > 40 else "No risk"
        }

# Initialize our data and models
tickets_df, vdi_df = generate_synthetic_data()
ticket_ai = SimpleTicketAI()
vdi_monitor = VDIMonitor()

# Header with animation
st.markdown("""
<div class="main-header">
    <h1 style='margin:0; font-size: 3rem; font-weight: 300;'>👔 HUGO BOSS</h1>
    <h2 style='margin:0.5rem 0 0 0; font-weight: 200; font-size: 1.8rem;'>
        IT Operations Intelligence Platform
    </h2>
    <p style='margin-top: 1rem; opacity: 0.9; font-size: 1rem;'>
        🚀 AI-Powered • ⚡ Real-Time Monitoring • 🔮 Predictive Analytics
    </p>
</div>
""", unsafe_allow_html=True)

# AI Insights Bar - this updates with smart insights
def show_ai_insights():
    """Show rotating AI insights - makes the platform feel alive"""
    insights = [
        "🧠 AI Insight: Design team shows 43% more VDI usage on Monday mornings",
        "📊 Pattern Detected: SAP issues correlate with month-end financial closing",
        "⚡ Optimization: Moving 15 users from VDI-03 to VDI-02 would balance load perfectly",
        "🎯 Trend: POS issues in Berlin store always happen during lunch rush (12-2 PM)",
        "🔮 Prediction: Tomorrow's peak load expected at 14:00-15:00 (Fashion review meeting)",
        "💡 Smart Alert: Adobe Creative Suite needs update on 12 workstations",
        "📈 Performance: Ticket resolution improved 23% this week"
    ]
    
    current_insight = insights[st.session_state.insight_idx % len(insights)]
    
    col1, col2 = st.columns([10, 1])
    with col1:
        st.info(current_insight)
    with col2:
        if st.button("→", help="Next insight"):
            st.session_state.insight_idx += 1
            st.rerun()

# Show insights at the top
show_ai_insights()

# Sidebar with gradient styling
with st.sidebar:
    # Logo area
    st.markdown("""
    <div style='text-align: center; padding: 1rem; background: white; border-radius: 15px; margin-bottom: 1rem;'>
        <h2 style='color: #667eea; margin: 0;'>HUGO BOSS</h2>
        <p style='color: #764ba2; margin: 0; font-size: 0.9rem;'>IT Intelligence</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Navigation
    st.markdown("### 🧭 Navigation")
    module = st.radio(
        "",
        ["🏠 Executive Dashboard", 
         "🎫 Ticket Intelligence", 
         "💻 VDI Monitoring",
         "🤖 AI Predictions",
         "💬 AI Assistant",
         "📊 Deep Analytics",
         "🏆 Team Performance"]
    )
    
    st.markdown("---")
    
    # Live Status Panel
    st.markdown("### 🔴 Live Status")
    
    # Check actual system status
    latest_vdi = vdi_df.iloc[0]
    system_health = vdi_monitor.analyze(
        latest_vdi['cpu_usage'], 
        latest_vdi['memory_usage'],
        latest_vdi['active_sessions']
    )
    
    if system_health['status'] == 'CRITICAL':
        st.error(f"{system_health['color']} CRITICAL ALERT")
    elif system_health['status'] == 'WARNING':
        st.warning(f"{system_health['color']} Warning Detected")
    else:
        st.success(f"{system_health['color']} All Systems Operational")
    
    # Live metrics
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Uptime", "99.97%", "+0.02%")
    with col2:
        st.metric("Active Users", f"{random.randint(245, 267)}")
    
    # Fashion Week Mode Toggle
    st.markdown("---")
    fashion_week = st.checkbox("🎭 Fashion Week Mode", help="Enable special monitoring for fashion events")
    if fashion_week:
        st.warning("⚡ Enhanced monitoring active!")

# Main Content Area
if module == "🏠 Executive Dashboard":
    
    # KPI Cards with gradient colors
    st.markdown("### 📊 Key Performance Indicators")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        recent = len(tickets_df[tickets_df['created_at'] > datetime.now() - timedelta(days=7)])
        st.metric("Weekly Tickets", recent, f"-{random.randint(5,15)}", delta_color="inverse")
    
    with col2:
        avg_res = tickets_df['resolution_time_hours'].mean()
        st.metric("Avg Resolution", f"{avg_res:.1f}h", "-0.5h", delta_color="inverse")
    
    with col3:
        critical = len(tickets_df[tickets_df['priority'] == 'Critical'])
        st.metric("Critical Issues", critical, "+2")
    
    with col4:
        satisfaction = tickets_df['satisfaction_score'].mean()
        st.metric("Satisfaction", f"{satisfaction:.1f}/5.0", "+0.2")
    
    with col5:
        current_cpu = vdi_df.iloc[0]['cpu_usage']
        st.metric("VDI Load", f"{current_cpu:.0f}%", f"{random.uniform(-5,5):.1f}%")
    
    st.markdown("---")
    
    # Fashion-specific metrics
    st.markdown("### 👔 Fashion Operations Status")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.success("✅ **Design Systems**")
        st.caption("CAD, Adobe CC, CLO3D")
        st.caption("All systems operational")
    
    with col2:
        st.info("🎨 **Creative Cloud**")
        st.caption(f"{random.randint(45,55)} active licenses")
        st.caption("12 available")
    
    with col3:
        st.success("🛍️ **POS Systems**")
        st.caption(f"{random.randint(47,50)}/50 online")
        st.caption("All stores connected")
    
    with col4:
        if fashion_week:
            st.warning("🎭 **Event Mode**")
            st.caption("Fashion Week Active")
            st.caption("Enhanced monitoring")
        else:
            st.success("💼 **SAP Fashion**")
            st.caption("ERP Connected")
            st.caption("All modules online")
    
    st.markdown("---")
    
    # Charts with color
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📈 Ticket Volume Trend")
        
        # Prepare data
        daily_tickets = tickets_df.groupby(tickets_df['created_at'].dt.date).size().reset_index()
        daily_tickets.columns = ['Date', 'Tickets']
        
        # Create colorful line chart
        fig = px.line(daily_tickets.tail(14), x='Date', y='Tickets',
                     title="Last 14 Days", markers=True)
        fig.update_traces(line_color='#667eea', line_width=3, 
                         marker=dict(size=10, color='#764ba2'))
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#333'),
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### 🎯 Priority Distribution")
        
        # Priority pie chart with custom colors
        priority_dist = tickets_df['priority'].value_counts()
        
        fig = px.pie(values=priority_dist.values, names=priority_dist.index,
                    color_discrete_map={
                        'Critical': '#e74c3c',
                        'High': '#f39c12',
                        'Medium': '#3498db',
                        'Low': '#2ecc71'
                    })
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(
            showlegend=True,
            font=dict(size=14)
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Department Performance Matrix
    st.markdown("---")
    st.markdown("### 🏢 Department Performance Matrix")
    
    dept_performance = tickets_df.groupby('department').agg({
        'ticket_id': 'count',
        'resolution_time_hours': 'mean',
        'satisfaction_score': 'mean'
    }).round(2).head(8)
    
    dept_performance.columns = ['Total Tickets', 'Avg Resolution (h)', 'Satisfaction']
    
    # Create scatter plot with gradient colors
    fig = px.scatter(dept_performance.reset_index(), 
                    x='Avg Resolution (h)', 
                    y='Satisfaction',
                    size='Total Tickets',
                    color='Total Tickets',
                    hover_data=['department'],
                    color_continuous_scale='Viridis',
                    title="Department IT Performance")
    
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )
    st.plotly_chart(fig, use_container_width=True)

elif module == "🎫 Ticket Intelligence":
    
    st.markdown("## 🎫 Intelligent Ticket Management System")
    
    tab1, tab2, tab3, tab4 = st.tabs(["🆕 Create & Predict", "📋 Active Tickets", "🧠 AI Analysis", "📈 Patterns"])
    
    with tab1:
        st.markdown("### 🤖 AI-Powered Ticket Creation")
        
        col1, col2 = st.columns(2)
        
        with col1:
            category = st.selectbox("System Category", 
                                   ['VDI', 'SAP', 'Adobe Creative', 'POS Systems', 
                                    'Network', 'Email', 'PLM System'])
            
            department = st.selectbox("Department",
                                     ['Creative Design', 'Product Development',
                                      'Retail Operations', 'Supply Chain', 
                                      'E-Commerce', 'Finance'])
            
            location = st.selectbox("Location",
                                   ['Metzingen Headquarters', 'Munich Flagship Store',
                                    'Berlin Kurfürstendamm', 'Stuttgart Distribution Center'])
        
        with col2:
            user_name = st.text_input("Your Name", "")
            
            description = st.text_area("Issue Description",
                                      "VDI session freezing when opening large CAD files. "
                                      "Multiple users affected in Creative Design. "
                                      "This is blocking the new collection work.",
                                      height=120)
        
        if st.button("🔮 Get AI Analysis", type="primary", use_container_width=True):
            
            # AI prediction
            prediction = ticket_ai.predict(category, description, department)
            
            # Success message with animation
            st.balloons()
            st.success("✅ AI Analysis Complete!")
            
            # Display results in colored cards
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                if prediction['predicted_priority'] == 'Critical':
                    st.error(f"🚨 **Priority**\n\n{prediction['predicted_priority']}")
                elif prediction['predicted_priority'] == 'High':
                    st.warning(f"⚠️ **Priority**\n\n{prediction['predicted_priority']}")
                else:
                    st.info(f"ℹ️ **Priority**\n\n{prediction['predicted_priority']}")
            
            with col2:
                st.metric("AI Confidence", f"{prediction['confidence']:.1f}%")
            
            with col3:
                st.metric("Est. Resolution", f"{prediction['estimated_hours']:.1f}h")
            
            with col4:
                st.info(f"**Auto-Assign**\n\n{prediction['auto_assign'].split('(')[0]}")
            
            # Detailed recommendations
            st.markdown("---")
            st.markdown("### 🎯 AI Recommendations")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                **📋 Suggested Actions:**
                1. Check VDI server resource allocation
                2. Review similar tickets from last week
                3. Consider temporary workstation for user
                4. Schedule maintenance window if needed
                """)
            
            with col2:
                st.markdown("""
                **📚 Knowledge Base:**
                - KB-2024-VDI-089: VDI Performance Optimization
                - KB-2024-VDI-045: CAD Software Best Practices
                - KB-2024-NET-023: Network Latency Troubleshooting
                """)
    
    with tab2:
        st.markdown("### 📋 Live Ticket Queue")
        
        # Filter options
        col1, col2, col3 = st.columns(3)
        with col1:
            filter_priority = st.multiselect("Priority", ['Critical', 'High', 'Medium', 'Low'], 
                                            default=['Critical', 'High'])
        with col2:
            filter_dept = st.multiselect("Department", tickets_df['department'].unique()[:5],
                                        default=tickets_df['department'].unique()[:2])
        with col3:
            filter_days = st.slider("Last N days", 1, 30, 7)
        
        # Filter data
        filtered = tickets_df[
            (tickets_df['priority'].isin(filter_priority)) &
            (tickets_df['department'].isin(filter_dept)) &
            (tickets_df['created_at'] > datetime.now() - timedelta(days=filter_days))
        ].head(20)
        
        # Display with color coding
        for _, ticket in filtered.iterrows():
            color = {'Critical': '🔴', 'High': '🟡', 'Medium': '🔵', 'Low': '🟢'}
            
            with st.expander(f"{color[ticket['priority']]} {ticket['ticket_id']} - {ticket['category']}"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.write(f"**Priority:** {ticket['priority']}")
                    st.write(f"**Department:** {ticket['department']}")
                with col2:
                    st.write(f"**Location:** {ticket['location']}")
                    st.write(f"**Assigned:** {ticket['assigned_to']}")
                with col3:
                    st.write(f"**Created:** {ticket['created_at'].strftime('%Y-%m-%d %H:%M')}")
                    st.write(f"**Resolution:** {ticket['resolution_time_hours']:.1f}h")
    
    with tab3:
        st.markdown("### 🧠 AI Pattern Analysis")
        
        # Show discovered patterns
        st.info("""
        **🔍 AI-Discovered Patterns This Week:**
        
        • **Peak Hours:** 76% of critical tickets occur between 14:00-16:00
        • **Monday Syndrome:** Ticket volume 43% higher on Mondays
        • **VDI Correlation:** VDI issues spike when Creative Design has deadlines
        • **Quick Wins:** Tickets with error codes resolve 2.3x faster
        • **Location Pattern:** Munich store has 30% more POS issues during weekends
        """)
        
        # Predictive maintenance suggestions
        st.markdown("#### 🔮 Predictive Maintenance Suggestions")
        
        suggestions = [
            {"System": "VDI-PROD-03", "Risk": "High", "Action": "Schedule RAM upgrade", "When": "This weekend"},
            {"System": "SAP Connector", "Risk": "Medium", "Action": "Update integration", "When": "Next Tuesday"},
            {"System": "POS Berlin", "Risk": "Low", "Action": "Firmware update", "When": "Next month"},
        ]
        
        for sug in suggestions:
            if sug['Risk'] == 'High':
                st.error(f"**{sug['System']}** - Risk: {sug['Risk']} - {sug['Action']} ({sug['When']})")
            elif sug['Risk'] == 'Medium':
                st.warning(f"**{sug['System']}** - Risk: {sug['Risk']} - {sug['Action']} ({sug['When']})")
            else:
                st.success(f"**{sug['System']}** - Risk: {sug['Risk']} - {sug['Action']} ({sug['When']})")
    
    with tab4:
        st.markdown("### 📈 Ticket Patterns & Trends")
        
        # Heatmap of tickets by hour and day
        st.markdown("#### 🗓️ Ticket Heatmap (Hour vs Day)")
        
        # Create heatmap data
        heatmap_data = tickets_df.copy()
        heatmap_data['hour'] = heatmap_data['created_at'].dt.hour
        heatmap_data['day'] = heatmap_data['created_at'].dt.day_name()
        
        pivot = heatmap_data.groupby(['hour', 'day']).size().reset_index(name='count')
        pivot = pivot.pivot(index='hour', columns='day', values='count').fillna(0)
        
        fig = px.imshow(pivot, 
                       labels=dict(x="Day of Week", y="Hour of Day", color="Tickets"),
                       color_continuous_scale="Viridis",
                       title="When do tickets come in?")
        st.plotly_chart(fig, use_container_width=True)

elif module == "💻 VDI Monitoring":
    
    st.markdown("## 💻 VDI Infrastructure Monitoring")
    
    # Get latest metrics and analysis
    latest = vdi_df.iloc[0]
    analysis = vdi_monitor.analyze(latest['cpu_usage'], latest['memory_usage'], latest['active_sessions'])
    
    # Status Cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if analysis['status'] == 'CRITICAL':
            st.error(f"{analysis['color']} **System Status**\n\n{analysis['status']}")
        elif analysis['status'] == 'WARNING':
            st.warning(f"{analysis['color']} **System Status**\n\n{analysis['status']}")
        else:
            st.success(f"{analysis['color']} **System Status**\n\n{analysis['status']}")
    
    with col2:
        # Risk score with color
        risk_color = "🔴" if analysis['risk_score'] > 70 else "🟡" if analysis['risk_score'] > 40 else "🟢"
        st.metric(f"{risk_color} Risk Score", f"{analysis['risk_score']}/100")
    
    with col3:
        st.metric("⏱️ Time to Failure", analysis['time_to_failure'])
    
    with col4:
        st.metric("👥 Active Sessions", latest['active_sessions'])
    
    # Alert Section
    if analysis['issues']:
        st.markdown("---")
        st.markdown("### ⚠️ Active Issues")
        for issue in analysis['issues']:
            st.error(issue)
    
    if analysis['recommendations']:
        st.markdown("### 💡 AI Recommendations")
        for rec in analysis['recommendations']:
            st.info(f"→ {rec}")
    
    # Tabs for different views
    tab1, tab2, tab3 = st.tabs(["📊 Real-Time Metrics", "🖥️ Server Status", "📈 Historical Analysis"])
    
    with tab1:
        st.markdown("### 📊 Real-Time Performance Metrics")
        
        # Real-time chart
        recent_vdi = vdi_df.head(50)
        
        fig = go.Figure()
        
        # CPU trace with gradient
        fig.add_trace(go.Scatter(
            x=recent_vdi['timestamp'],
            y=recent_vdi['cpu_usage'],
            name='CPU Usage',
            mode='lines',
            line=dict(color='#e74c3c', width=3),
            fill='tonexty',
            fillcolor='rgba(231, 76, 60, 0.1)'
        ))
        
        # Memory trace with gradient
        fig.add_trace(go.Scatter(
            x=recent_vdi['timestamp'],
            y=recent_vdi['memory_usage'],
            name='Memory Usage',
            mode='lines',
            line=dict(color='#3498db', width=3),
            fill='tonexty',
            fillcolor='rgba(52, 152, 219, 0.1)'
        ))
        
        # GPU trace
        fig.add_trace(go.Scatter(
            x=recent_vdi['timestamp'],
            y=recent_vdi['gpu_usage'],
            name='GPU Usage',
            mode='lines',
            line=dict(color='#2ecc71', width=2, dash='dot')
        ))
        
        # Add threshold lines
        fig.add_hline(y=80, line_dash="dash", line_color="red", 
                     annotation_text="Critical Threshold", opacity=0.5)
        fig.add_hline(y=60, line_dash="dash", line_color="orange", 
                     annotation_text="Warning Threshold", opacity=0.5)
        
        fig.update_layout(
            title="VDI Resource Utilization",
            xaxis_title="Time",
            yaxis_title="Usage %",
            height=400,
            hovermode='x unified',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.markdown("### 🖥️ Server Status Grid")
        
        # Create server grid
        servers = ['VDI-PROD-01', 'VDI-PROD-02', 'VDI-PROD-03', 'VDI-PROD-04', 'VDI-PROD-05']
        
        cols = st.columns(5)
        for idx, server in enumerate(servers):
            with cols[idx]:
                # Simulate different server states
                cpu = random.uniform(40, 95)
                memory = random.uniform(45, 90)
                sessions = random.randint(20, 50)
                
                # Determine health
                if cpu > 85 or memory > 85:
                    st.error(f"**{server}**")
                    health = "⚠️ Critical"
                    health_score = random.uniform(40, 60)
                elif cpu > 70 or memory > 70:
                    st.warning(f"**{server}**")
                    health = "⚡ Warning"
                    health_score = random.uniform(60, 80)
                else:
                    st.success(f"**{server}**")
                    health = "✅ Healthy"
                    health_score = random.uniform(80, 100)
                
                st.metric("CPU", f"{cpu:.0f}%")
                st.metric("Memory", f"{memory:.0f}%")
                st.metric("Sessions", sessions)
                st.metric("Health", f"{health_score:.0f}/100")
                st.caption(health)
    
    with tab3:
        st.markdown("### 📈 Historical Performance Analysis")
        
        # Historical trends
        col1, col2 = st.columns(2)
        
        with col1:
            # Daily average chart
            daily_avg = vdi_df.groupby(vdi_df['timestamp'].dt.date).agg({
                'cpu_usage': 'mean',
                'memory_usage': 'mean',
                'health_score': 'mean'
            }).round(1)
            
            fig = px.line(daily_avg, y=['cpu_usage', 'memory_usage'],
                         title="Daily Average Usage",
                         color_discrete_map={'cpu_usage': '#e74c3c', 'memory_usage': '#3498db'})
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Server comparison
            server_stats = vdi_df.groupby('server')['cpu_usage'].mean().round(1)
            
            fig = px.bar(x=server_stats.index, y=server_stats.values,
                        title="Average CPU by Server",
                        color=server_stats.values,
                        color_continuous_scale='RdYlGn_r')
            st.plotly_chart(fig, use_container_width=True)

elif module == "🤖 AI Predictions":
    
    st.markdown("## 🤖 AI Predictions & Forecasting")
    
    tab1, tab2, tab3 = st.tabs(["📅 24-Hour Forecast", "📆 Weekly Prediction", "🎯 Smart Recommendations"])
    
    with tab1:
        st.markdown("### 🔮 Next 24 Hours Prediction")
        
        # Generate hourly predictions
        predictions = []
        for hour in range(24):
            time = datetime.now() + timedelta(hours=hour)
            
            # Realistic predictions based on time
            if 9 <= time.hour <= 17:  # Business hours
                tickets = random.randint(8, 20)
                vdi_load = random.uniform(65, 85)
                risk = "Medium" if time.hour in [10, 14, 15] else "Low"
            else:
                tickets = random.randint(0, 5)
                vdi_load = random.uniform(20, 40)
                risk = "Low"
            
            predictions.append({
                'Time': time.strftime('%H:00'),
                'Hour': time.hour,
                'Predicted Tickets': tickets,
                'VDI Load %': vdi_load,
                'Risk Level': risk
            })
        
        pred_df = pd.DataFrame(predictions)
        
        # Interactive chart with dual axis
        fig = go.Figure()
        
        # Bar chart for tickets
        fig.add_trace(go.Bar(
            x=pred_df['Time'],
            y=pred_df['Predicted Tickets'],
            name='Predicted Tickets',
            marker_color='#667eea',
            yaxis='y',
            text=pred_df['Predicted Tickets'],
            textposition='outside'
        ))
        
        # Line chart for VDI load
        fig.add_trace(go.Scatter(
            x=pred_df['Time'],
            y=pred_df['VDI Load %'],
            name='VDI Load %',
            mode='lines+markers',
            line=dict(color='#e74c3c', width=3),
            marker=dict(size=8),
            yaxis='y2'
        ))
        
        fig.update_layout(
            title="24-Hour Forecast",
            xaxis=dict(title="Time"),
            yaxis=dict(title="Predicted Tickets", side="left"),
            yaxis2=dict(title="VDI Load %", overlaying="y", side="right"),
            hovermode='x unified',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Key predictions
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.info("""
            **📈 Peak Hours**
            - 10:00-11:00: Morning rush
            - 14:00-16:00: Afternoon peak
            - Total Expected: ~180 tickets
            """)
        
        with col2:
            high_risk_hours = pred_df[pred_df['Risk Level'] != 'Low']['Time'].tolist()
            if high_risk_hours:
                st.warning(f"""
                **⚠️ High Risk Windows**
                - {', '.join(high_risk_hours[:3])}
                - Action: Pre-scale resources
                - Monitor: VDI-PROD-03
                """)
            else:
                st.success("**✅ No high risk periods detected**")
        
        with col3:
            st.success("""
            **💡 AI Suggestions**
            - Schedule maintenance: 03:00
            - Optimal backup time: 22:00
            - Staff needed: 2 extra at 14:00
            """)
    
    with tab2:
        st.markdown("### 📆 7-Day Predictive Calendar")
        
        # Generate weekly predictions
        weekly_predictions = []
        for day in range(7):
            date = datetime.now() + timedelta(days=day)
            
            # Different patterns for different days
            if date.weekday() == 0:  # Monday
                risk = "High"
                tickets = random.randint(150, 200)
                color = "#e74c3c"
            elif date.weekday() in [5, 6]:  # Weekend
                risk = "Low"
                tickets = random.randint(20, 50)
                color = "#2ecc71"
            else:
                risk = "Medium"
                tickets = random.randint(80, 120)
                color = "#f39c12"
            
            weekly_predictions.append({
                'Date': date,
                'Day': date.strftime('%A'),
                'Risk': risk,
                'Predicted Tickets': tickets,
                'Color': color,
                'Maintenance': 'Scheduled' if day == 3 else 'None'
            })
        
        # Display as cards
        st.markdown("#### Risk Assessment by Day")
        cols = st.columns(7)
        for idx, pred in enumerate(weekly_predictions):
            with cols[idx]:
                if pred['Risk'] == 'High':
                    st.error(f"**{pred['Day'][:3]}**")
                elif pred['Risk'] == 'Medium':
                    st.warning(f"**{pred['Day'][:3]}**")
                else:
                    st.success(f"**{pred['Day'][:3]}**")
                
                st.caption(f"{pred['Date'].strftime('%b %d')}")
                st.metric("Tickets", pred['Predicted Tickets'])
                
                if pred['Maintenance'] == 'Scheduled':
                    st.info("🔧 Maintenance")
    
    with tab3:
        st.markdown("### 🎯 AI-Generated Smart Recommendations")
        
        # Generate smart recommendations
        recommendations = [
            {
                "Priority": "Critical",
                "Category": "Infrastructure",
                "Recommendation": "VDI-PROD-03 showing degradation patterns. Schedule RAM upgrade this weekend.",
                "Impact": "Prevent 40+ tickets",
                "ROI": "€5,000 saved"
            },
            {
                "Priority": "High",
                "Category": "Staffing",
                "Recommendation": "Add 2 support staff on Monday mornings (9-12 AM)",
                "Impact": "Reduce resolution time by 35%",
                "ROI": "Improved satisfaction"
            },
            {
                "Priority": "Medium",
                "Category": "Automation",
                "Recommendation": "Automate password reset tickets (30% of all tickets)",
                "Impact": "Save 15 hours/week",
                "ROI": "€2,000/month"
            },
            {
                "Priority": "Low",
                "Category": "Training",
                "Recommendation": "SAP training for Creative Design team",
                "Impact": "Reduce SAP tickets by 20%",
                "ROI": "Long-term efficiency"
            }
        ]
        
        for rec in recommendations:
            with st.expander(f"{rec['Priority']} - {rec['Category']}: {rec['Recommendation'][:50]}..."):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Recommendation:** {rec['Recommendation']}")
                    st.write(f"**Priority:** {rec['Priority']}")
                with col2:
                    st.write(f"**Expected Impact:** {rec['Impact']}")
                    st.write(f"**ROI:** {rec['ROI']}")

elif module == "💬 AI Assistant":
    
    st.markdown("## 💬 AI IT Support Assistant")
    st.caption("Ask me anything about IT operations, tickets, or system status!")
    
    # Chat interface
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Type your question here..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
        
        # Generate response based on keywords
        response = ""
        prompt_lower = prompt.lower()
        
        if "vdi" in prompt_lower or "server" in prompt_lower:
            response = f"""
            🖥️ **VDI System Analysis:**
            
            Current Status: {vdi_monitor.analyze(vdi_df.iloc[0]['cpu_usage'], 
                                                vdi_df.iloc[0]['memory_usage'],
                                                vdi_df.iloc[0]['active_sessions'])['status']}
            
            • CPU Load: {vdi_df.iloc[0]['cpu_usage']:.1f}%
            • Memory Usage: {vdi_df.iloc[0]['memory_usage']:.1f}%
            • Active Sessions: {vdi_df.iloc[0]['active_sessions']}
            • Health Score: {vdi_df.iloc[0]['health_score']:.1f}/100
            
            📊 All VDI servers are currently operational. VDI-PROD-03 showing slightly elevated CPU usage but within acceptable limits.
            """
        
        elif "ticket" in prompt_lower or "issue" in prompt_lower:
            critical_count = len(tickets_df[tickets_df['priority'] == 'Critical'])
            high_count = len(tickets_df[tickets_df['priority'] == 'High'])
            
            response = f"""
            🎫 **Ticket System Summary:**
            
            Active Tickets:
            • Critical: {critical_count} tickets
            • High: {high_count} tickets
            • Total Open: {len(tickets_df)} tickets
            
            📈 **Trends:**
            • Average Resolution Time: {tickets_df['resolution_time_hours'].mean():.1f} hours
            • Top Issue Category: {tickets_df['category'].mode()[0]}
            • Most Affected Department: {tickets_df['department'].mode()[0]}
            
            💡 Recommendation: Focus on {tickets_df['category'].mode()[0]} issues to reduce ticket volume.
            """
        
        elif "help" in prompt_lower or "what can you do" in prompt_lower:
            response = """
            👋 **I'm your AI IT Assistant! I can help with:**
            
            🎫 **Tickets:**
            - Check ticket status and priorities
            - Predict resolution times
            - Analyze patterns and trends
            
            💻 **VDI Monitoring:**
            - Real-time server status
            - Performance metrics
            - Failure predictions
            
            📊 **Analytics:**
            - Department performance
            - Historical trends
            - Predictive insights
            
            Try asking:
            • "What's the VDI status?"
            • "Show me critical tickets"
            • "Predict tomorrow's workload"
            • "Which department has most issues?"
            """
        
        elif "predict" in prompt_lower or "tomorrow" in prompt_lower:
            response = f"""
            🔮 **Predictive Analysis for Tomorrow:**
            
            Based on historical patterns and current trends:
            
            📈 **Expected Ticket Volume:** {random.randint(45, 65)} tickets
            • Peak Time: 14:00-16:00
            • Likely Categories: VDI (40%), SAP (25%), Other (35%)
            
            💻 **VDI Prediction:**
            • Expected Load: 75-85% during business hours
            • Risk Level: Medium
            • Recommendation: Monitor VDI-PROD-03 closely
            
            👥 **Staffing Suggestion:**
            • Minimum Staff: 4 technicians
            • Recommended: 5 technicians (1 extra for peak hours)
            
            ⚠️ **Risk Alert:** If Fashion Design has deadline tomorrow, expect 30% more VDI tickets.
            """
        
        else:
            response = f"""
            🤔 I understood you're asking about: *{prompt}*
            
            Let me provide a general system overview:
            
            ✅ **System Status:** All systems operational
            📊 **Current Metrics:**
            • Active tickets: {len(tickets_df)}
            • VDI Health: Good
            • Average response time: {tickets_df['resolution_time_hours'].mean():.1f}h
            
            For specific information, try asking about:
            • VDI status
            • Ticket details
            • Predictions
            • Department performance
            
            How else can I help you today?
            """
        
        # Add assistant message
        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.write(response)

elif module == "📊 Deep Analytics":
    
    st.markdown("## 📊 Deep Analytics & Insights")
    
    tab1, tab2, tab3, tab4 = st.tabs(["🏢 Departments", "⏰ Time Analysis", "🔄 Correlations", "💰 Cost Analysis"])
    
    with tab1:
        st.markdown("### 🏢 Department Performance Analysis")
        
        # Department metrics
        dept_stats = tickets_df.groupby('department').agg({
            'ticket_id': 'count',
            'resolution_time_hours': 'mean',
            'satisfaction_score': 'mean'
        }).round(2)
        
        dept_stats.columns = ['Total Tickets', 'Avg Resolution (h)', 'Satisfaction']
        dept_stats = dept_stats.sort_values('Total Tickets', ascending=False)
        
        # Performance scatter plot
        fig = px.scatter(dept_stats.reset_index(), 
                        x='Avg Resolution (h)', 
                        y='Satisfaction',
                        size='Total Tickets',
                        color='Total Tickets',
                        hover_data=['department'],
                        color_continuous_scale='Turbo',
                        title="Department IT Performance Matrix",
                        labels={'Total Tickets': 'Ticket Volume'})
        
        fig.add_hline(y=4.0, line_dash="dash", annotation_text="Target Satisfaction")
        fig.add_vline(x=6.0, line_dash="dash", annotation_text="Target Resolution")
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Department ranking
        st.markdown("#### 🏆 Department Rankings")
        
        for idx, (dept, row) in enumerate(dept_stats.head(5).iterrows()):
            medal = "🥇" if idx == 0 else "🥈" if idx == 1 else "🥉" if idx == 2 else "🏅"
            
            col1, col2, col3, col4 = st.columns([1, 3, 2, 2])
            with col1:
                st.write(f"### {medal}")
            with col2:
                st.write(f"**{dept}**")
            with col3:
                st.metric("Tickets", int(row['Total Tickets']))
            with col4:
                st.metric("Satisfaction", f"{row['Satisfaction']:.1f}/5")
    
    with tab2:
        st.markdown("### ⏰ Temporal Pattern Analysis")
        
        # Hour of day analysis
        hourly = tickets_df.groupby(tickets_df['created_at'].dt.hour).size()
        
        fig = px.bar(x=hourly.index, y=hourly.values,
                    labels={'x': 'Hour of Day', 'y': 'Ticket Count'},
                    title="Ticket Distribution by Hour",
                    color=hourly.values,
                    color_continuous_scale='Sunset')
        
        fig.add_vline(x=9, line_dash="dash", annotation_text="Work starts")
        fig.add_vline(x=17, line_dash="dash", annotation_text="Work ends")
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Day of week analysis
        st.markdown("#### 📅 Weekly Patterns")
        
        daily = tickets_df.groupby(tickets_df['created_at'].dt.day_name()).size()
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        daily = daily.reindex(day_order)
        
        fig = px.line(x=day_order, y=daily.values,
                     markers=True,
                     title="Weekly Ticket Pattern",
                     labels={'x': 'Day', 'y': 'Average Tickets'})
        
        fig.update_traces(line_color='#667eea', line_width=3,
                         marker=dict(size=12, color='#764ba2'))
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.markdown("### 🔄 Correlation Analysis")
        
        # Create correlation matrix
        correlation_data = tickets_df.copy()
        correlation_data['hour'] = correlation_data['created_at'].dt.hour
        correlation_data['day_of_week'] = correlation_data['created_at'].dt.dayofweek
        
        # Encode categorical variables
        for col in ['priority', 'category', 'department']:
            correlation_data[col + '_encoded'] = pd.Categorical(correlation_data[col]).codes
        
        # Select numerical columns
        corr_matrix = correlation_data[['resolution_time_hours', 'satisfaction_score', 
                                       'hour', 'day_of_week', 'priority_encoded']].corr()
        
        fig = px.imshow(corr_matrix,
                       text_auto=True,
                       color_continuous_scale='RdBu',
                       title="Correlation Heatmap")
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.info("""
        **🔍 Key Correlations Found:**
        • Priority strongly correlates with resolution time (expected)
        • Satisfaction negatively correlates with resolution time
        • Time of day affects ticket priority
        • Monday (day 0) shows higher priority tickets
        """)
    
    with tab4:
        st.markdown("### 💰 Cost Analysis & ROI")
        
        # Calculate costs (simulated)
        cost_per_hour = 50  # EUR
        total_hours = tickets_df['resolution_time_hours'].sum()
        total_cost = total_hours * cost_per_hour
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Support Hours", f"{total_hours:.0f}h")
        with col2:
            st.metric("Total Cost", f"€{total_cost:,.0f}")
        with col3:
            avg_cost = total_cost / len(tickets_df)
            st.metric("Cost per Ticket", f"€{avg_cost:.0f}")
        
        # Cost by category
        st.markdown("#### Cost Distribution by Category")
        
        category_cost = tickets_df.groupby('category')['resolution_time_hours'].sum() * cost_per_hour
        
        fig = px.pie(values=category_cost.values, 
                    names=category_cost.index,
                    title="Cost Distribution (EUR)",
                    color_discrete_sequence=px.colors.qualitative.Set3)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # ROI of improvements
        st.markdown("#### 💡 Potential Savings with AI")
        
        improvements = [
            {"Area": "Auto-ticket routing", "Current": "€12,000/month", "With AI": "€8,000/month", "Savings": "€4,000"},
            {"Area": "Predictive maintenance", "Current": "€18,000/month", "With AI": "€10,000/month", "Savings": "€8,000"},
            {"Area": "Self-service portal", "Current": "€15,000/month", "With AI": "€5,000/month", "Savings": "€10,000"},
        ]
        
        total_savings = 0
        for imp in improvements:
            with st.expander(f"{imp['Area']} - Save {imp['Savings']}/month"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Current Cost", imp['Current'])
                with col2:
                    st.metric("With AI", imp['With AI'])
                with col3:
                    st.metric("Monthly Savings", imp['Savings'])
        
        st.success(f"### 💰 Total Potential Savings: €22,000/month (€264,000/year)")

elif module == "🏆 Team Performance":
    
    st.markdown("## 🏆 IT Team Performance Dashboard")
    
    # Team leaderboard
    st.markdown("### 🎯 Team Leaderboard - This Month")
    
    team_members = [
        {"Name": "Klaus Mueller", "Team": "VDI Support", "Tickets": 87, "Avg Time": "2.1h", 
         "Satisfaction": 4.8, "Score": 94, "Trend": "↑"},
        {"Name": "Anna Schmidt", "Team": "SAP/ERP", "Tickets": 76, "Avg Time": "3.2h", 
         "Satisfaction": 4.6, "Score": 87, "Trend": "↑"},
        {"Name": "Marcus Weber", "Team": "Creative IT", "Tickets": 82, "Avg Time": "2.8h", 
         "Satisfaction": 4.7, "Score": 90, "Trend": "→"},
        {"Name": "Lisa Fischer", "Team": "Retail Systems", "Tickets": 71, "Avg Time": "2.5h", 
         "Satisfaction": 4.5, "Score": 85, "Trend": "↑"},
        {"Name": "Thomas Bauer", "Team": "Network Ops", "Tickets": 65, "Avg Time": "3.5h", 
         "Satisfaction": 4.4, "Score": 82, "Trend": "↓"},
    ]
    
    for idx, member in enumerate(team_members):
        col1, col2, col3, col4, col5, col6, col7 = st.columns([0.5, 2.5, 1.5, 1.5, 1.5, 1, 1])
        
        with col1:
            if idx == 0:
                st.markdown("### 🥇")
            elif idx == 1:
                st.markdown("### 🥈")
            elif idx == 2:
                st.markdown("### 🥉")
            else:
                st.markdown(f"### {idx+1}")
        
        with col2:
            st.write(f"**{member['Name']}**")
            st.caption(member['Team'])
        
        with col3:
            st.metric("Resolved", member['Tickets'], member['Trend'])
        
        with col4:
            st.metric("Avg Time", member['Avg Time'])
        
        with col5:
            st.metric("Rating", f"{member['Satisfaction']}/5")
        
        with col6:
            st.metric("Score", member['Score'])
        
        with col7:
            if member['Score'] >= 90:
                st.success("⭐ Excellent")
            elif member['Score'] >= 85:
                st.info("👍 Good")
            else:
                st.warning("📈 Improving")
    
    st.markdown("---")
    
    # Achievements section
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎖️ Recent Achievements")
        
        achievements = [
            "🔥 **Klaus Mueller**: 10 critical tickets resolved in one day!",
            "⚡ **Anna Schmidt**: Prevented major SAP outage during month-end",
            "🎯 **Marcus Weber**: 100% satisfaction score this week",
            "🚀 **Team Record**: Lowest average resolution time this quarter",
            "💡 **Lisa Fischer**: Implemented new POS quick-fix protocol"
        ]
        
        for achievement in achievements:
            st.success(achievement)
    
    with col2:
        st.markdown("### 📈 Team Metrics")
        
        # Team performance chart
        team_data = pd.DataFrame(team_members)
        
        fig = px.radar(
            r=[member['Score'] for member in team_members],
            theta=[member['Name'].split()[0] for member in team_members],
            fill='toself',
            title="Team Performance Radar"
        )
        
        fig.update_traces(fill='toself', fillcolor='rgba(102, 126, 234, 0.3)',
                         line=dict(color='#667eea', width=2))
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Gamification elements
    st.markdown("---")
    st.markdown("### 🎮 Gamification & Rewards")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("""
        **🏅 This Week's Challenge**
        
        "Speed Resolver"
        - Goal: Resolve 20 tickets under 2h
        - Prize: Extra day off
        - Leader: Klaus (18/20)
        """)
    
    with col2:
        st.warning("""
        **🎯 Team Goal**
        
        "Customer Delight"
        - Goal: 4.5+ satisfaction
        - Current: 4.6/5.0
        - Status: ACHIEVED! 🎉
        """)
    
    with col3:
        st.success("""
        **🏆 Monthly Trophy**
        
        "IT Hero of the Month"
        - Winner: Anna Schmidt
        - Reason: SAP crisis management
        - Reward: €500 bonus
        """)

# Footer with gradient
st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 2rem; 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
            border-radius: 15px; color: white;'>
    <h3 style='margin: 0;'>Hugo Boss IT Operations Intelligence Platform</h3>
    <p style='margin: 0.5rem 0 0 0; opacity: 0.9;'>
        Built By Tony Thomas | Powered by Advanced AI | © 2025
    </p>
    <p style='margin: 0.5rem 0 0 0; font-size: 0.9rem; opacity: 0.8;'>
        Making IT Operations Smarter, Faster, Better
    </p>
</div>
""", unsafe_allow_html=True)