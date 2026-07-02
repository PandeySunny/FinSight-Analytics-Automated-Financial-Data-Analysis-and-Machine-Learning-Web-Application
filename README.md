## 🎯 Project Overview

**Website Link : https://finance-data-analysis.onrender.com/**

**FinSight Analytics** is a comprehensive financial ML platform that automates advanced data analysis, machine learning model training, and intelligent visualization for financial datasets. The platform performs sophisticated tasks including fraud detection, customer segmentation, risk scoring, market analysis, and predictive modeling.

## 🔬 Core ML Capabilities

### 1. **Fraud & Anomaly Detection**
- **Algorithm**: Isolation Forest with decision scores
- **Features**: 
  - Detects statistically abnormal transactions
  - Flags suspicious patterns with anomaly score ranking
  - Identifies top 20 most anomalous records for review
  - Automatically adjustable contamination threshold (2% default)
- **Output**: Anomaly flags, fraud rate %, suspicious transaction indices

### 2. **Customer Segmentation & Behavioral Analysis**
- **Algorithm**: K-Means Clustering with DBSCAN comparison
- **Features**:
  - Adaptive cluster count based on dataset size
  - Segment profiling (mean values per feature)
  - Anomaly rate per segment identification
  - Transaction pattern grouping
- **Output**: Segment profiles, cluster inertia, behavioral insights

### 3. **Risk Scoring & Tier Classification**
- **Algorithm**: Quantile-based risk tier stratification
- **Features**:
  - 4-tier risk classification (Low, Medium, High, Critical)
  - Quantile distribution analysis
  - Risk tier breakdowns with counts
- **Output**: Risk tier assignments, tier-wise record distribution

### 4. **Supervised Fraud Classification**
- **Algorithm**: Gradient Boosting Classifier with ROC-AUC scoring
- **Features**:
  - Trains on labeled fraud/non-fraud data
  - Feature importance ranking
  - Classification report generation
  - ROC-AUC performance metrics
- **Output**: Model predictions, feature importances, ROC-AUC score

### 5. **Dimensionality Reduction & Visualization**
- **Algorithm**: Principal Component Analysis (PCA)
- **Features**:
  - 2D/3D component projection
  - Explained variance tracking
  - Anomaly visualization in PCA space
  - Segment cluster visualization
- **Output**: PCA coordinates, explained variance ratios

### 6. **Market Data Analysis**
- **Metrics Computed**:
  - Return calculations (daily/periodic)
  - Annualized volatility (rolling 20-day)
  - Moving averages (20-day & 50-day)
  - Relative Strength Index (RSI-14)
  - Sharpe Ratio (risk-adjusted returns)
  - Maximum Drawdown analysis
- **Use Cases**: Stock prices, crypto, commodity data, financial indices

### 7. **Time-Series Analysis**
- **Features**:
  - Automatic resampling (daily/weekly/monthly)
  - Transaction volume trends
  - Value aggregation over time
  - Rolling averages overlays
- **Output**: Temporal trend visualizations, pattern detection

### 8. **Feature Importance & Model Interpretability**
- **Methods**:
  - Tree-based feature importance from Gradient Boosting
  - Correlation matrix analysis
  - Feature relationship mapping
- **Output**: Ranked feature importance, correlation heatmaps

## Graphical Representation

                  User
                    │
                    ▼
         Upload CSV / Excel File
                    │
                    ▼
            Flask Backend (app.py)
                    │
      ┌─────────────┴──────────────┐
      │                            │
Data Preprocessing           Dataset Detection
      │                            │
      └─────────────┬──────────────┘
                    ▼
          Machine Learning Pipeline
                    │
 ┌─────────┬──────────┬─────────┬─────────┐
 │         │          │         │         │
 ▼         ▼          ▼         ▼         ▼
Isolation KMeans    PCA     Gradient   Market
Forest              Analysis Boosting  Metrics
                    │
                    ▼
         Graphs + KPIs + Insights
                    │
                    ▼
            Results Dashboard

## ✨ Key Features

### Automated Analysis
- **Statistical Summary**: Comprehensive dataset overview with data types, null counts, unique values, mean, and standard deviation
- **Column Classification**: Intelligent financial domain mapping (transactions, market data, banking, credit risk)
- **Data Cleaning**: Automatic handling of null values, duplicates, and datetime conversion
- **Smart Sampling**: Intelligent memory optimization for large datasets (up to 500MB)
- **Financial KPIs**: Transaction volume, fraud rate, risk distribution, model performance

### Advanced Visualizations
- **Anomaly Detection Chart**: PCA-projected fraud vs normal transactions
- **Customer Segmentation Chart**: Cluster visualization with segment sizes
- **Risk Tier Distribution**: Bar + pie charts showing risk tier breakdown
- **Feature Importance**: Horizontal bar chart from trained fraud model
- **Correlation Heatmap**: Feature relationship matrix with annotations
- **Distribution Plots**: Histograms with KDE for numerical features
- **Boxplots**: Outlier detection for each numerical column
- **Market Analysis Charts**: Price trends, volatility, RSI indicators
- **Time-Series Charts**: Transaction volume and count over time
- **Categorical Distribution**: Top categories breakdown

### Professional Interface
- **Modern Landing Page**: Corporate-grade hero section, features showcase, technology stack
- **Clean Dashboard**: Professional results presentation with stats cards and responsive grids
- **Responsive Design**: Works seamlessly on desktop, tablet, and mobile devices
- **Intuitive Navigation**: Professional navbar with smooth scrolling and clear CTAs
- **Financial Domain UI**: Sector-specific color scheme and terminology

## 🛠️ Technology Stack

### Backend
- **Python 3.x** - Core programming language
- **Flask** - Web framework for routing and templating
- **Scikit-Learn** - ML algorithms (KMeans, IsolationForest, GradientBoosting, PCA)
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing
- **Matplotlib** - Static visualizations
- **Seaborn** - Statistical data visualization

### ML Algorithms
- **Unsupervised**: K-Means, DBSCAN, Isolation Forest, PCA
- **Supervised**: Gradient Boosting Classifier, Logistic Regression
- **Data Processing**: StandardScaler, LabelEncoder, SimpleImputer, train_test_split

### Frontend  
- **HTML5** - Semantic markup
- **CSS3** - Professional styling with custom design system
- **Font Awesome** - Professional icon library
- **Responsive Design** - Mobile-first approach

### Data Processing
- **CSV/XLSX Support** - Multiple file formats
- **Large File Handling** - Up to 500MB with chunking
- **Memory Optimization** - Intelligent sampling for performance
- **Datetime Handling** - Automatic conversion to year/month/day features
- **Financial Column Detection** - Domain-specific feature recognition

## 📊 Features Showcase

### ML Analysis Capabilities
✅ Fraud detection with anomaly scoring  
✅ Customer segmentation into behavioral groups  
✅ Risk tier classification (Low/Medium/High/Critical)  
✅ Supervised fraud model training with feature importance  
✅ PCA dimensionality reduction and visualization  
✅ Market data technical analysis (RSI, Sharpe, volatility)  
✅ Time-series trend analysis  
✅ Dataset type auto-detection (transactions/market/banking/credit_risk)  
✅ Financial KPI computation  

### Supported Dataset Types
✅ **Transactions**: Amount, merchant, category, timestamp  
✅ **Market Data**: Price, volume, high/low, RSI  
✅ **Banking**: Balance, credit, debit, loans  
✅ **Credit Risk**: Fraud/default labels, credit scores  
✅ **General**: Any numerical/categorical data  

### Visualization Types
✅ 8+ distribution charts (histograms + boxplots)  
✅ Fraud detection PCA scatter plot  
✅ Customer segmentation cluster visualization  
✅ Risk tier bar & pie charts  
✅ Feature importance horizontal bar chart  
✅ Correlation heatmap with values  
✅ Market analysis (price, volatility, RSI)  
✅ Time-series volume & count trends  
✅ Categorical distribution charts  

## 🚀 How to Run Locally

### Prerequisites
```bash
Python 3.8+
pip (Python package manager)
```

### Installation Steps

1. **Clone/Download the project**
```bash
cd "FinSight-Analytics"
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the application**
```bash
python app.py
```

4. **Access the application**
Open your browser and navigate to: `http://127.0.0.1:5000`

## 📁 Project Structure

```
FinSight-Analytics/
├── app.py                 # Main Flask app with ML pipeline & routes
├── requirements.txt       # Python dependencies
├── templates/
│   ├── index.html        # Professional landing page
│   ├── upload.html       # File upload interface
│   └── results.html      # ML analysis results dashboard
├── static/
│   ├── style.css         # Professional CSS design system
│   └── plots/            # Generated visualization PNG files
├── uploads/              # Uploaded CSV/XLSX files storage
└── README.md             # Project documentation
```

## 💼 ML Pipeline Architecture

### Stage 1: Data Ingestion & Preparation
- File upload with format validation
- Automatic file format detection (CSV/XLSX)
- Data type inference and column classification
- Missing value imputation (median strategy)
- Duplicate detection and removal

### Stage 2: Financial Domain Intelligence
- Column mapping to financial concepts
- Automatic dataset type detection
- Datetime feature extraction (year/month/day)
- Feature scaling and normalization

### Stage 3: ML Model Execution
1. **Anomaly Detection**: Isolation Forest on scaled features
2. **Segmentation**: K-Means clustering with profile analysis
3. **Risk Scoring**: Quantile-based tier assignment
4. **Dimensionality Reduction**: PCA for visualization
5. **Supervised Learning**: Gradient Boosting on labeled data
6. **Market Analysis**: Financial metrics computation (RSI, Sharpe)
7. **Time-Series**: Temporal pattern extraction

### Stage 4: Visualization Generation
- Automatic chart type selection per data type
- PCA scatter plots for clusters and anomalies
- Distribution and outlier detection charts
- Feature importance visualization
- Risk and segmentation heatmaps

### Stage 5: Insights & KPI Generation
- Fraud rate & anomaly analysis
- Segment profiling and comparison
- Risk tier distribution
- Model performance metrics
- Financial KPIs (volume, mean, p95)
- Actionable recommendations

## 🎓 Skills Demonstrated

### Machine Learning
- Anomaly detection & fraud prevention
- Customer segmentation & clustering
- Supervised classification modeling
- Dimensionality reduction
- Feature engineering & importance
- Risk scoring frameworks
- Time-series analysis
- Model evaluation metrics

### Data Science
- Exploratory Data Analysis (EDA)
- Statistical analysis & profiling
- Data visualization & storytelling
- Financial domain knowledge
- Technical indicator computation
- Domain-specific feature extraction

### Software Development
- Python programming (OOP, functional)
- Web development (Flask, HTML, CSS)
- File handling and I/O
- Session management
- Error handling and logging
- Security best practices
- Code organization & modularity

### Problem Solving
- Memory optimization for large files
- Automatic algorithm selection
- Efficient data processing pipelines
- Production-ready error handling
- User experience optimization

## 📈 Real-World Use Cases

- **Fraud Detection**: Flag suspicious transactions automatically
- **Customer Analytics**: Segment users by behavior for targeted campaigns
- **Risk Assessment**: Classify transactions by risk tier for compliance
- **Market Analysis**: Technical analysis for trading decisions
- **Credit Risk**: Predict default likelihood with ML models
- **Data Quality**: Identify anomalies and data issues
- **Report Generation**: Automated statistical summaries and visualizations
- **Educational Tool**: Learn ML concepts interactively

## 🌟 Future Enhancements

Potential improvements to showcase continuous learning:
- [ ] Deep learning models (LSTM, VAE for anomaly detection)
- [ ] Interactive Plotly visualizations with Dash
- [ ] Export results to PDF/Excel with formatted reports
- [ ] SHAP model interpretability
- [ ] Database storage for analysis history
- [ ] User authentication system
- [ ] Real-time data streaming support
- [ ] API endpoints for programmatic access
- [ ] Ensemble model approaches
- [ ] Collaborative features and sharing
- [ ] Advanced time-series forecasting
- [ ] Hyperparameter tuning automation

---

**Built with** ❤️ **using Python, Flask, Scikit-Learn, Pandas, and modern web technologies**

**Key Algorithms**: Isolation Forest, K-Means, Gradient Boosting, PCA, StandardScaler, LabelEncoder
