"""
THE AI BRAIN
This is where we teach machines to think like IT experts
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, mean_absolute_error
import pickle
import warnings
warnings.filterwarnings('ignore')

class TicketBrain:
    """
    This AI learns from historical tickets to predict:
    1. What priority a new ticket should have
    2. How long it will take to resolve
    3. Which team should handle it
    """
    
    def __init__(self):
        print("🧠 Initializing Ticket Intelligence Brain...")
        self.priority_model = None
        self.resolution_model = None
        self.encoders = {}
        
    def prepare_features(self, df):
        """
        Convert raw ticket data into numbers the AI can understand
        This is called 'Feature Engineering' - a critical skill
        """
        print("🔧 Engineering features from raw data...")
        
        # Time-based features (AI learns patterns)
        df['hour'] = pd.to_datetime(df['created_at']).dt.hour
        df['day_of_week'] = pd.to_datetime(df['created_at']).dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['is_business_hours'] = df['hour'].between(8, 18).astype(int)
        
        # Text features (simplified - in production, use NLP)
        df['description_length'] = df['description'].str.len()
        df['has_error_code'] = df['description'].str.contains('Error|ERROR|error').astype(int)
        df['is_urgent_language'] = df['description'].str.contains('urgent|ASAP|immediately|blocked').astype(int)
        
        # Encode categorical variables
        for col in ['category', 'department', 'location']:
            if col not in self.encoders:
                self.encoders[col] = LabelEncoder()
                df[f'{col}_encoded'] = self.encoders[col].fit_transform(df[col])
            else:
                df[f'{col}_encoded'] = self.encoders[col].transform(df[col])
        
        # Encode priority for training
        if 'priority' in df.columns:
            if 'priority' not in self.encoders:
                self.encoders['priority'] = LabelEncoder()
                # Custom order for priority
                priority_order = ['Low', 'Medium', 'High', 'Critical']
                self.encoders['priority'].fit(priority_order)
            df['priority_encoded'] = self.encoders['priority'].transform(df['priority'])
        
        return df
    
    def train(self, tickets_df):
        """
        The Training Process - This is where the magic happens
        """
        print("🎓 Starting AI Training Process...")
        print("-" * 40)
        
        # Prepare the data
        df = self.prepare_features(tickets_df.copy())
        
        # Select features for training
        feature_columns = [
            'hour', 'day_of_week', 'is_weekend', 'is_business_hours',
            'description_length', 'has_error_code', 'is_urgent_language',
            'category_encoded', 'department_encoded', 'location_encoded'
        ]
        
        X = df[feature_columns]
        
        # Train Priority Predictor
        print("📊 Training Priority Prediction Model...")
        y_priority = df['priority_encoded']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_priority, test_size=0.2, random_state=42
        )
        
        # We use Random Forest - it's like asking 100 experts and taking a vote
        self.priority_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        
        self.priority_model.fit(X_train, y_train)
        
        # Evaluate
        predictions = self.priority_model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        print(f"✅ Priority Model Accuracy: {accuracy:.1%}")
        
        # Train Resolution Time Predictor
        print("⏱️ Training Resolution Time Model...")
        y_resolution = df['resolution_time_hours']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_resolution, test_size=0.2, random_state=42
        )
        
        # Gradient Boosting - starts with a weak model and keeps improving
        self.resolution_model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        
        self.resolution_model.fit(X_train, y_train)
        
        # Evaluate
        predictions = self.resolution_model.predict(X_test)
        mae = mean_absolute_error(y_test, predictions)
        print(f"✅ Resolution Time MAE: {mae:.1f} hours")
        
        print("-" * 40)
        print("🎉 AI Training Complete!")
        
        # Feature importance (what the AI learned matters most)
        importance = pd.DataFrame({
            'feature': feature_columns,
            'importance': self.priority_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\n🔍 Top 3 Most Important Features:")
        for idx, row in importance.head(3).iterrows():
            print(f"  - {row['feature']}: {row['importance']:.3f}")
        
        return self
    
    def predict(self, ticket_data):
        """
        Make predictions on new tickets
        """
        # Prepare features
        df = pd.DataFrame([ticket_data])
        df = self.prepare_features(df)
        
        feature_columns = [
            'hour', 'day_of_week', 'is_weekend', 'is_business_hours',
            'description_length', 'has_error_code', 'is_urgent_language',
            'category_encoded', 'department_encoded', 'location_encoded'
        ]
        
        X = df[feature_columns]
        
        # Predict
        priority_encoded = self.priority_model.predict(X)[0]
        priority = self.encoders['priority'].inverse_transform([priority_encoded])[0]
        
        resolution_time = self.resolution_model.predict(X)[0]
        
        # Get confidence (probability of the predicted class)
        priority_proba = self.priority_model.predict_proba(X)[0]
        confidence = max(priority_proba) * 100
        
        return {
            'predicted_priority': priority,
            'confidence': confidence,
            'estimated_resolution_hours': resolution_time,
            'recommended_sla': self._get_sla(priority)
        }
    
    def _get_sla(self, priority):
        """SLA based on priority"""
        slas = {
            'Critical': '1 hour',
            'High': '4 hours',
            'Medium': '8 hours',
            'Low': '24 hours'
        }
        return slas.get(priority, '8 hours')
    
    def save(self, filename='ticket_brain.pkl'):
        """Save the trained brain"""
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
        print(f"💾 Brain saved to {filename}")

class VDIBrain:
    """
    This AI monitors VDI health and predicts failures
    """
    
    def __init__(self):
        print("🖥️ Initializing VDI Intelligence Brain...")
        self.anomaly_threshold = None
        self.pattern_model = None
        
    def train(self, vdi_df):
        """
        Learn normal patterns to detect anomalies
        """
        print("📊 Learning VDI behavior patterns...")
        
        # Calculate normal operating ranges
        self.cpu_normal = (vdi_df['cpu_usage'].quantile(0.05), 
                          vdi_df['cpu_usage'].quantile(0.95))
        self.memory_normal = (vdi_df['memory_usage'].quantile(0.05),
                             vdi_df['memory_usage'].quantile(0.95))
        
        # Calculate risk thresholds
        self.cpu_warning = vdi_df['cpu_usage'].quantile(0.90)
        self.cpu_critical = vdi_df['cpu_usage'].quantile(0.95)
        self.memory_warning = vdi_df['memory_usage'].quantile(0.90)
        self.memory_critical = vdi_df['memory_usage'].quantile(0.95)
        
        print(f"✅ Learned Normal CPU Range: {self.cpu_normal[0]:.1f}% - {self.cpu_normal[1]:.1f}%")
        print(f"✅ Learned Normal Memory Range: {self.memory_normal[0]:.1f}% - {self.memory_normal[1]:.1f}%")
        
        return self
    
    def predict_health(self, cpu, memory, active_sessions):
        """
        Predict VDI health and failure risk
        """
        risk_score = 0
        issues = []
        recommendations = []
        
        # CPU Analysis
        if cpu > self.cpu_critical:
            risk_score += 40
            issues.append(f"CPU critically high: {cpu:.1f}%")
            recommendations.append("Immediate: Migrate active sessions to backup VDI")
        elif cpu > self.cpu_warning:
            risk_score += 20
            issues.append(f"CPU warning: {cpu:.1f}%")
            recommendations.append("Monitor closely, prepare backup VDI")
        
        # Memory Analysis
        if memory > self.memory_critical:
            risk_score += 40
            issues.append(f"Memory critically high: {memory:.1f}%")
            recommendations.append("Clear cache, restart non-critical services")
        elif memory > self.memory_warning:
            risk_score += 20
            issues.append(f"Memory warning: {memory:.1f}%")
            
        # Session load analysis
        if active_sessions > 150:
            risk_score += 20
            issues.append(f"High session count: {active_sessions}")
            recommendations.append("Consider load balancing")
        
        # Determine overall status
        if risk_score >= 70:
            status = "CRITICAL"
            action = "IMMEDIATE ACTION REQUIRED"
        elif risk_score >= 40:
            status = "WARNING"
            action = "Preventive action recommended"
        else:
            status = "HEALTHY"
            action = "No action needed"
        
        return {
            'status': status,
            'risk_score': risk_score,
            'action': action,
            'issues': issues,
            'recommendations': recommendations,
            'predicted_failure_probability': min(risk_score, 100)
        }

# Train the brains
if __name__ == "__main__":
    print("=" * 50)
    print("🧠 HUGO BOSS AI BRAIN TRAINING CENTER")
    print("=" * 50)
    
    # Load data
    tickets_df = pd.read_csv('hugo_boss_tickets.csv')
    vdi_df = pd.read_csv('hugo_boss_vdi_metrics.csv')
    
    # Train Ticket Brain
    ticket_brain = TicketBrain()
    ticket_brain.train(tickets_df)
    ticket_brain.save()
    
    print()
    
    # Train VDI Brain
    vdi_brain = VDIBrain()
    vdi_brain.train(vdi_df)
    
    # Save VDI brain
    with open('vdi_brain.pkl', 'wb') as f:
        pickle.dump(vdi_brain, f)
    print("💾 VDI Brain saved")
    
    print("\n" + "=" * 50)
    print("✅ ALL AI BRAINS TRAINED AND READY!")
    print("=" * 50)