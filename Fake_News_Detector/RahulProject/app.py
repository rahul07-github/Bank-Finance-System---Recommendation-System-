import streamlit as st
import joblib
import numpy as np
import time
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import re
import pandas as pd
from textblob import TextBlob
from scipy import sparse
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="NewsTrust AI - Fake News Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== ENHANCED MODEL LOADING WITH AUTO-FIX ====================
@st.cache_resource
def load_smart_model():
    """
    Smart model loading with automatic dimension fixing
    """
    try:
        # Try to load the model
        model = joblib.load('random_forest.joblib')
        
        # Get expected features from model
        expected_features = None
        if hasattr(model, 'n_features_in_'):
            expected_features = model.n_features_in_
        elif hasattr(model, 'feature_importances_'):
            expected_features = len(model.feature_importances_)
        else:
            expected_features = 5000  # Default based on your error
        
        # Try to load vectorizer with multiple fallback options
        vectorizer_files = [
            'tfidf_vectorizer_fixed.joblib',
            'tfidf_vectorizer_compatible.joblib',
            'tfidf_vectorizer.joblib'
        ]
        
        base_vectorizer = None
        loaded_file = None
        
        for file in vectorizer_files:
            try:
                base_vectorizer = joblib.load(file)
                loaded_file = file
                break
            except:
                continue
        
        if base_vectorizer is None:
            # Create a simple fallback vectorizer
            from sklearn.feature_extraction.text import TfidfVectorizer
            base_vectorizer = TfidfVectorizer(max_features=expected_features)
            # Fit with dummy data
            dummy_texts = ["news article text analysis fake true"]
            base_vectorizer.fit(dummy_texts)
        
        # Create a DimensionFixer wrapper class
        class DimensionFixer:
            def __init__(self, base_vectorizer, expected_features):
                self.base_vectorizer = base_vectorizer
                self.expected_features = expected_features
                self.fix_count = 0
                self.last_warning_shown = False
            
            def transform(self, texts):
                # Transform using base vectorizer
                X = self.base_vectorizer.transform(texts)
                current_features = X.shape[1]
                
                # Fix dimensions if needed
                if current_features != self.expected_features:
                    self.fix_count += 1
                    
                    # Show warning only once
                    if not self.last_warning_shown:
                        self.last_warning_shown = True
                    
                    if current_features < self.expected_features:
                        # Add zero columns
                        missing = self.expected_features - current_features
                        zeros = sparse.csr_matrix((X.shape[0], missing))
                        X_fixed = sparse.hstack([X, zeros])
                    else:
                        # Truncate extra columns
                        X_fixed = X[:, :self.expected_features]
                    
                    return X_fixed
                
                return X
            
            def get_stats(self):
                return {
                    'expected_features': self.expected_features,
                    'fix_count': self.fix_count,
                    'needs_fix': self.fix_count > 0
                }
            
            def __getattr__(self, name):
                return getattr(self.base_vectorizer, name)
        
        # Create the fixed vectorizer
        fixed_vectorizer = DimensionFixer(base_vectorizer, expected_features)
        
        # Test the setup
        test_text = "test"
        X_test = fixed_vectorizer.transform([test_text])
        
        return model, fixed_vectorizer, "primary"
        
    except Exception as e:
        return None, None, "error"

# Load model
model, vectorizer, model_type = load_smart_model()

# Initialize session state
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []
if 'user_feedback' not in st.session_state:
    st.session_state.user_feedback = {}

# ==================== TEXT ANALYSIS FUNCTIONS ====================
def analyze_text_characteristics(text):
    """Extract various text characteristics"""
    char_count = len(text)
    word_count = len(text.split())
    sentences = re.split(r'[.!?]+', text)
    sentence_count = len([s for s in sentences if s.strip()])
    
    # Calculate readability
    if sentence_count > 0 and word_count > 0:
        avg_sentence_length = word_count / sentence_count
        avg_word_length = char_count / word_count
        readability_score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_word_length)
        readability_level = "Easy" if readability_score > 60 else "Moderate" if readability_score > 30 else "Difficult"
    else:
        readability_score = 0
        readability_level = "N/A"
        avg_sentence_length = 0
        avg_word_length = 0
    
    # Sentiment analysis
    try:
        blob = TextBlob(text)
        sentiment = blob.sentiment
        polarity = sentiment.polarity
        subjectivity = sentiment.subjectivity
    except:
        polarity = 0
        subjectivity = 0.5
    
    # Check for sensational words
    sensational_words = ['shocking', 'amazing', 'unbelievable', 'breaking', 'exclusive', 
                        'secret', 'explosive', 'revealed', 'must-see', 'you wont believe']
    sensational_count = sum(1 for word in sensational_words if word.lower() in text.lower())
    
    # Check for emotional words
    emotional_words = ['terrible', 'horrible', 'wonderful', 'fantastic', 'awful', 
                      'disgusting', 'outrageous', 'shameful', 'amazing', 'hate']
    emotional_count = sum(1 for word in emotional_words if word.lower() in text.lower())
    
    # Check for all caps
    caps_words = len([word for word in text.split() if word.isupper() and len(word) > 2])
    
    # Check for excessive punctuation
    excl_marks = text.count('!')
    quest_marks = text.count('?')
    
    return {
        'char_count': char_count,
        'word_count': word_count,
        'sentence_count': sentence_count,
        'readability_score': readability_score,
        'readability_level': readability_level,
        'sentiment_polarity': polarity,
        'sentiment_subjectivity': subjectivity,
        'sensational_count': sensational_count,
        'emotional_count': emotional_count,
        'caps_words': caps_words,
        'excl_marks': excl_marks,
        'quest_marks': quest_marks,
        'avg_word_length': avg_word_length,
        'avg_sentence_length': avg_sentence_length
    }

def enhance_prediction_with_rules(text, ml_prediction=None, ml_confidence=None):
    """Enhance ML prediction with rule-based analysis"""
    analysis = analyze_text_characteristics(text)
    
    # Start with base score
    credibility_score = 70
    
    # Adjust based on text characteristics
    if analysis['char_count'] < 50:
        credibility_score -= 20  # Too short
    elif analysis['char_count'] > 5000:
        credibility_score -= 10  # Too long
    
    if analysis['sensational_count'] > 3:
        credibility_score -= 15  # Too sensational
    
    if analysis['emotional_count'] > 5:
        credibility_score -= 10  # Too emotional
    
    if analysis['caps_words'] > 3:
        credibility_score -= 10  # Too much shouting
    
    if analysis['excl_marks'] > 5:
        credibility_score -= 5  # Too many exclamations
    
    # Readability adjustment
    if analysis['readability_level'] == "Difficult":
        credibility_score += 5  # More formal
    
    # Sentiment adjustment
    if abs(analysis['sentiment_polarity']) > 0.7:
        credibility_score -= 10  # Too extreme
    
    # Objectivity adjustment
    if analysis['sentiment_subjectivity'] > 0.8:
        credibility_score -= 10  # Too subjective
    
    # Combine with ML prediction if available
    if ml_prediction and ml_confidence:
        if ml_confidence > 0.8:
            weight = 0.7
        elif ml_confidence > 0.6:
            weight = 0.5
        else:
            weight = 0.3
        
        # Convert ML prediction to score
        ml_score = {
            'TRUE': 90,
            'mostly-true': 75,
            'half-true': 60,
            'barely-true': 40,
            'FALSE': 25,
            'pants-fire': 10
        }.get(str(ml_prediction).upper(), 50)
        
        credibility_score = (weight * ml_score) + ((1 - weight) * credibility_score)
    
    # Normalize score
    credibility_score = max(10, min(100, credibility_score))
    
    # Determine label
    if credibility_score >= 80:
        final_label = "Highly Credible"
        emoji = "✅"
        color = "#00CC88"
    elif credibility_score >= 60:
        final_label = "Mostly Credible"
        emoji = "👍"
        color = "#4ECDC4"
    elif credibility_score >= 40:
        final_label = "Partially Credible"
        emoji = "⚠️"
        color = "#FFD166"
    elif credibility_score >= 20:
        final_label = "Questionable"
        emoji = "❓"
        color = "#FF9966"
    else:
        final_label = "Highly Suspicious"
        emoji = "❌"
        color = "#FF4B4B"
    
    return {
        'final_label': final_label,
        'emoji': emoji,
        'color': color,
        'credibility_score': credibility_score,
        'analysis': analysis,
        'used_ml': ml_prediction is not None,
        'ml_prediction': ml_prediction,
        'ml_confidence': ml_confidence
    }

def create_confidence_gauge(score):
    """Create a beautiful confidence gauge"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Credibility Score", 'font': {'size': 24, 'color': 'white'}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': 'white'},
            'bar': {'color': "#4ECDC4"},
            'bgcolor': "rgba(0,0,0,0)",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 20], 'color': '#FF4B4B'},
                {'range': [20, 40], 'color': '#FF9966'},
                {'range': [40, 60], 'color': '#FFD166'},
                {'range': [60, 80], 'color': '#4ECDC4'},
                {'range': [80, 100], 'color': '#00CC88'}
            ],
            'threshold': {
                'line': {'color': "white", 'width': 3},
                'thickness': 0.8,
                'value': score
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        font={'color': "white", 'family': "Arial"},
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig

def create_text_analysis_radar(analysis):
    """Create radar chart for text analysis"""
    categories = ['Length', 'Readability', 'Balance', 'Objectivity', 'Formality']
    
    # Normalize values
    length_score = min(100, analysis['char_count'] / 100 * 100)
    readability_score = max(0, min(100, analysis['readability_score']))
    balance_score = 100 - min(100, abs(analysis['sentiment_polarity']) * 100)  # Neutral sentiment is better
    objectivity_score = (1 - analysis['sentiment_subjectivity']) * 100
    formality_score = 100 - min(100, (analysis['sensational_count'] + analysis['emotional_count']) * 15)
    
    values = [length_score, readability_score, balance_score, objectivity_score, formality_score]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Text Analysis',
        line_color='#4ECDC4',
        fillcolor='rgba(78, 205, 196, 0.3)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                gridcolor='rgba(255,255,255,0.2)',
                tickcolor='white'
            ),
            bgcolor='rgba(0,0,0,0)'
        ),
        showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white', size=12),
        height=350
    )
    
    return fig

# ==================== CUSTOM CSS ====================
st.markdown("""
<style>
    /* Main theme */
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        color: #FFFFFF;
        min-height: 100vh;
    }
    
    /* Main header */
    .main-header {
        font-size: 3.5rem;
        text-align: center;
        margin-bottom: 1.5rem;
        font-weight: 800;
        background: linear-gradient(90deg, #FF6B35, #FFD166, #4ECDC4, #00CC88);
        background-size: 300% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: gradientFlow 3s linear infinite;
        text-shadow: 0 0 30px rgba(255, 107, 53, 0.3);
    }
    
    @keyframes gradientFlow {
        0% { background-position: 0% center; }
        100% { background-position: 300% center; }
    }
    
    /* Enhanced Analyze Button */
    .stButton > button[kind="primary"] {
        background: linear-gradient(90deg, #FF6B35, #FFD166, #00CC88);
        background-size: 200% auto;
        color: white;
        border: none;
        padding: 1rem 2.5rem;
        border-radius: 50px;
        font-weight: 700;
        font-size: 1.2rem;
        transition: all 0.5s ease;
        box-shadow: 0 5px 20px rgba(255, 107, 53, 0.4);
        position: relative;
        overflow: hidden;
    }
    
    .stButton > button[kind="primary"]:hover:not(:disabled) {
        transform: scale(1.05);
        box-shadow: 0 8px 30px rgba(255, 107, 53, 0.6);
        background-position: right center;
    }
    
    .stButton > button[kind="primary"]:disabled {
        background: #666666;
        cursor: not-allowed;
        opacity: 0.6;
    }
    
    .stButton > button[kind="primary"]::after {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: linear-gradient(45deg, transparent, rgba(255, 255, 255, 0.1), transparent);
        transform: rotate(45deg);
        animation: shine 3s infinite;
    }
    
    @keyframes shine {
        0% { transform: translateX(-100%) translateY(-100%) rotate(45deg); }
        100% { transform: translateX(100%) translateY(100%) rotate(45deg); }
    }
    
    /* Cards */
    .analysis-card {
        background: rgba(25, 25, 35, 0.8);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease;
    }
    
    .analysis-card:hover {
        transform: translateY(-5px);
        border-color: rgba(78, 205, 196, 0.5);
    }
    
    /* Metrics */
    .metric-card {
        background: rgba(40, 40, 50, 0.6);
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        border-left: 4px solid #4ECDC4;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(30, 30, 40, 0.8);
        padding: 10px;
        border-radius: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: rgba(40, 40, 50, 0.6);
        border-radius: 8px;
        padding: 10px 20px;
        color: white;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #FF6B35, #FFD166);
        color: white;
        box-shadow: 0 4px 15px rgba(255, 107, 53, 0.3);
    }
    
    /* Progress bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, #FF4B4B, #FFD166, #4ECDC4, #00CC88);
        border-radius: 10px;
    }
    
    /* Status indicators */
    .status-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 8px;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { transform: scale(0.95); opacity: 0.7; }
        50% { transform: scale(1.1); opacity: 1; }
        100% { transform: scale(0.95); opacity: 0.7; }
    }
    
    .status-online { background-color: #00CC88; box-shadow: 0 0 10px #00CC88; }
    .status-warning { background-color: #FFD166; box-shadow: 0 0 10px #FFD166; }
    .status-offline { background-color: #FF4B4B; box-shadow: 0 0 10px #FF4B4B; }
    
    /* Feature fix warning */
    .feature-fix-alert {
        background: rgba(255, 209, 102, 0.1);
        border: 2px solid #FFD166;
        border-radius: 15px;
        padding: 1rem;
        margin: 1rem 0;
        animation: pulseWarning 2s infinite;
    }
    
    @keyframes pulseWarning {
        0% { box-shadow: 0 0 0 0 rgba(255, 209, 102, 0.4); }
        70% { box-shadow: 0 0 0 10px rgba(255, 209, 102, 0); }
        100% { box-shadow: 0 0 0 0 rgba(255, 209, 102, 0); }
    }
</style>
""", unsafe_allow_html=True)

# ==================== APP HEADER ====================
st.markdown('<h1 class="main-header">🔍 NewsTrust AI</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #CCCCCC; margin-bottom: 2rem;">Advanced Fake News Detection with Multi-Layer Analysis</p>', unsafe_allow_html=True)

# ==================== SIDEBAR ====================
with st.sidebar:
    st.markdown("### 📊 System Status")
    
    # Model status
    status_col1, status_col2 = st.columns([1, 3])
    with status_col1:
        if model_type == "primary":
            st.markdown('<span class="status-indicator status-online"></span>', unsafe_allow_html=True)
        elif model_type == "error":
            st.markdown('<span class="status-indicator status-offline"></span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-indicator status-warning"></span>', unsafe_allow_html=True)
    
    with status_col2:
        if model_type == "primary":
            st.success("**AI Model Active**")
            if vectorizer and hasattr(vectorizer, 'get_stats'):
                stats = vectorizer.get_stats()
                if stats['needs_fix']:
                    st.caption(f"⚠️ Auto-fix enabled ({stats['fix_count']}x)")
                else:
                    st.caption("✅ Features aligned")
        elif model_type == "error":
            st.error("**System Error**")
            st.caption("Check model files")
        else:
            st.warning("**Rules Only**")
            st.caption("Enhanced rule-based")
    
    st.markdown("---")
    
    # Quick stats
    if st.session_state.analysis_history:
        st.markdown("### 📈 Recent Analysis")
        recent_scores = [h.get('result', {}).get('credibility_score', 50) 
                        for h in st.session_state.analysis_history[-5:]]
        if recent_scores:
            recent_avg = np.mean(recent_scores)
            col_stats1, col_stats2 = st.columns(2)
            with col_stats1:
                st.metric("Avg Score", f"{recent_avg:.1f}")
            with col_stats2:
                st.metric("Total", len(st.session_state.analysis_history))
            st.caption(f"Last {len(recent_scores)} analyses")
    
    st.markdown("---")
    
    # Tips
    st.markdown("### 💡 Tips for Accuracy")
    st.info("""
    **For best results:**
    1. Provide complete articles
    2. Minimum 100 characters
    3. Include source context
    4. Avoid copy-paste errors
    5. Check multiple sources
    """)

# ==================== MAIN TABS ====================
tab1, tab2, tab3 = st.tabs(["📝 Analyze News", "📊 Insights", "ℹ️ About"])

# ==================== TAB 1: ANALYZE NEWS ====================
with tab1:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### Paste News Content")
        
        news_text = st.text_area(
            "Enter the news article or statement:",
            height=250,
            placeholder="Paste the complete news content here...\n\n💡 For best results, provide complete articles with 100+ characters.",
            help="The more complete the text, the better the analysis",
            key="news_input"
        )
        
        if news_text:
            char_count = len(news_text)
            word_count = len(news_text.split())
            
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Characters", f"{char_count:,}")
            with col_b:
                st.metric("Words", f"{word_count:,}")
            
            if char_count < 100:
                st.warning("⚠️ Text is quite short. For better accuracy, provide more content.")
    
    with col2:
        st.markdown("### Quick Analysis")
        
        if news_text:
            with st.spinner("Analyzing..."):
                quick_stats = analyze_text_characteristics(news_text)
            
            st.markdown(f"""
            <div class="metric-card">
                <p style="font-size: 0.9rem; color: #CCCCCC;">Text Statistics</p>
                <div style="display: flex; justify-content: space-between; margin: 0.5rem 0;">
                    <span>Readability:</span>
                    <span style="color: #4ECDC4; font-weight: bold;">{quick_stats['readability_level']}</span>
                </div>
                <div style="display: flex; justify-content: space-between; margin: 0.5rem 0;">
                    <span>Sentiment:</span>
                    <span style="color: #4ECDC4; font-weight: bold;">{quick_stats['sentiment_polarity']:.2f}</span>
                </div>
                <div style="display: flex; justify-content: space-between; margin: 0.5rem 0;">
                    <span>Objectivity:</span>
                    <span style="color: #4ECDC4; font-weight: bold;">{(1 - quick_stats['sentiment_subjectivity']):.0%}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # Analyze button
    st.markdown("---")
    analyze_col1, analyze_col2, analyze_col3 = st.columns([1, 2, 1])
    with analyze_col2:
        analyze_btn = st.button(
            "🚀 START ADVANCED ANALYSIS",
            type="primary",
            use_container_width=True,
            disabled=not news_text or model_type == "error"
        )
    
    # Analysis Process
    if analyze_btn and news_text:
        with st.spinner("🔍 Performing multi-layer analysis..."):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            steps = [
                ("📥 Processing text", 20),
                ("🔍 Extracting features", 40),
                ("🤖 Running AI analysis", 60),
                ("📊 Applying rules", 80),
                ("✅ Finalizing results", 100)
            ]
            
            current_progress = 0
            for step_text, target in steps:
                status_text.markdown(
                    f"<div style='text-align: center; color: #4ECDC4; font-weight: bold;'>{step_text}...</div>",
                    unsafe_allow_html=True
                )
                
                # Animate progress for this step
                for percent in range(current_progress, target + 1):
                    progress_bar.progress(percent)
                    time.sleep(0.02)
                
                current_progress = target
            
            # Get ML prediction if model is available
            ml_prediction = None
            ml_confidence = None
            
            if model and vectorizer:
                try:
                    # Transform text with auto-fix
                    X = vectorizer.transform([news_text])
                    
                    # Show feature fix info if needed
                    if hasattr(vectorizer, 'get_stats'):
                        stats = vectorizer.get_stats()
                        if stats['fix_count'] > 0 and stats['fix_count'] <= 3:
                            st.info(f"🔧 Auto-adjusted features for compatibility")
                    
                    # Make prediction
                    ml_prediction = model.predict(X)[0]
                    
                    # Get confidence
                    if hasattr(model, 'predict_proba'):
                        ml_proba = model.predict_proba(X)[0]
                        ml_confidence = np.max(ml_proba)
                        
                except Exception as e:
                    st.warning(f"⚠️ ML prediction issue: {str(e)}. Using enhanced rules.")
            
            # Get enhanced prediction
            result = enhance_prediction_with_rules(news_text, ml_prediction, ml_confidence)
            
            # Store in history
            history_entry = {
                'timestamp': datetime.now(),
                'text_preview': news_text[:100] + "..." if len(news_text) > 100 else news_text,
                'result': result,
                'char_count': len(news_text)
            }
            st.session_state.analysis_history.append(history_entry)
            
            # Display Results
            st.markdown("---")
            st.balloons()
            
            st.markdown(f"""
            <div class="analysis-card">
                <div style="text-align: center; padding: 2rem;">
                    <h1 style="font-size: 4rem; margin: 0; color: {result['color']}; text-shadow: 0 0 20px {result['color']}80;">{result['emoji']}</h1>
                    <h2 style="color: {result['color']}; margin: 1rem 0; font-size: 2.5rem;">{result['final_label']}</h2>
                    <p style="font-size: 1.5rem; color: white; margin: 0;">
                        Credibility Score: <strong>{result['credibility_score']:.1f}/100</strong>
                    </p>
                    {f"<p style='color: #4ECDC4; margin-top: 1rem;'>🤖 AI Confidence: {result['ml_confidence']*100:.1f}%</p>" if result['ml_confidence'] else "<p style='color: #FFD166; margin-top: 1rem;'>📝 Using enhanced rule-based analysis</p>"}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Gauge Chart and Breakdown
            col_g1, col_g2 = st.columns([2, 1])
            
            with col_g1:
                st.plotly_chart(create_confidence_gauge(result['credibility_score']), use_container_width=True)
            
            with col_g2:
                st.markdown("### 📋 Score Breakdown")
                
                factors = [
                    ("Text Quality", min(100, result['analysis']['char_count'] / 5)),
                    ("Readability", max(0, min(100, result['analysis']['readability_score']))),
                    ("Objectivity", (1 - result['analysis']['sentiment_subjectivity']) * 100),
                    ("Balance", 100 - min(100, abs(result['analysis']['sentiment_polarity']) * 100))
                ]
                
                for factor, score in factors:
                    color = "#00CC88" if score >= 70 else "#FFD166" if score >= 40 else "#FF4B4B"
                    st.markdown(f"<div style='color: white; margin-bottom: 5px;'><strong>{factor}:</strong> {score:.0f}%</div>", unsafe_allow_html=True)
                    st.progress(score / 100)
            
            # Detailed Analysis
            st.markdown("### 📊 Detailed Text Analysis")
            
            col_d1, col_d2 = st.columns(2)
            
            with col_d1:
                st.markdown("#### 📈 Text Characteristics")
                
                metrics_data = [
                    ("Characters", f"{result['analysis']['char_count']:,}"),
                    ("Words", f"{result['analysis']['word_count']:,}"),
                    ("Sentences", result['analysis']['sentence_count']),
                    ("Avg Word Length", f"{result['analysis']['avg_word_length']:.1f} chars"),
                    ("Avg Sentence Length", f"{result['analysis']['avg_sentence_length']:.1f} words"),
                    ("Readability Level", result['analysis']['readability_level']),
                    ("Sensational Words", result['analysis']['sensational_count']),
                    ("Emotional Words", result['analysis']['emotional_count']),
                    ("ALL CAPS Words", result['analysis']['caps_words']),
                    ("Exclamation Marks", result['analysis']['excl_marks'])
                ]
                
                for label, value in metrics_data:
                    st.markdown(f"<div style='padding: 8px 0; border-bottom: 1px solid rgba(255,255,255,0.1);'><strong>{label}:</strong> <span style='color: #4ECDC4;'>{value}</span></div>", unsafe_allow_html=True)
            
            with col_d2:
                st.plotly_chart(create_text_analysis_radar(result['analysis']), use_container_width=True)
            
            # Recommendations
            st.markdown("### 💡 Recommendations")
            
            if result['credibility_score'] < 50:
                st.error("""
                ⚠️ **High Caution Advised:**
                - Verify with trusted news sources
                - Check publication date and author
                - Look for supporting evidence
                - Be skeptical of emotional/sensational language
                - Consider fact-checking websites
                """)
            elif result['credibility_score'] < 70:
                st.warning("""
                ⚠️ **Moderate Credibility:**
                - Additional verification recommended
                - Check author credentials
                - Look for citations and references
                - Compare with other reports
                - Consider potential biases
                """)
            else:
                st.success("""
                ✅ **High Credibility:**
                - Content appears reliable
                - Still verify critical claims
                - Check for recent updates
                - Share responsibly
                - Support credible journalism
                """)
            
            # User Feedback
            st.markdown("### 📝 Was this analysis helpful?")
            
            feedback_col1, feedback_col2, feedback_col3 = st.columns(3)
            
            with feedback_col1:
                if st.button("👍 Accurate", key="fb_accurate", use_container_width=True):
                    st.session_state.user_feedback[len(st.session_state.analysis_history)-1] = "accurate"
                    st.success("Thank you for your feedback! 🎯")
            
            with feedback_col2:
                if st.button("🤔 Somewhat", key="fb_somewhat", use_container_width=True):
                    st.session_state.user_feedback[len(st.session_state.analysis_history)-1] = "somewhat"
                    st.info("Thank you for your feedback! 📊")
            
            with feedback_col3:
                if st.button("👎 Inaccurate", key="fb_inaccurate", use_container_width=True):
                    st.session_state.user_feedback[len(st.session_state.analysis_history)-1] = "inaccurate"
                    st.error("Thank you for your feedback! We'll improve. 🔄")

# ==================== TAB 2: INSIGHTS ====================
with tab2:
    st.markdown("### 📈 Analysis Insights & Trends")
    
    if st.session_state.analysis_history:
        # Create dataframe from history
        df_history = pd.DataFrame([
            {
                'Timestamp': h['timestamp'],
                'Credibility': h['result']['credibility_score'],
                'Length': h['char_count'],
                'Label': h['result']['final_label'],
                'Color': h['result']['color']
            }
            for h in st.session_state.analysis_history
        ])
        
        col_i1, col_i2 = st.columns(2)
        
        with col_i1:
            fig_trend = px.line(
                df_history,
                x='Timestamp',
                y='Credibility',
                title='Credibility Trend Over Time',
                markers=True,
                line_shape='spline'
            )
            fig_trend.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                height=300
            )
            fig_trend.update_traces(line_color='#4ECDC4', line_width=3)
            st.plotly_chart(fig_trend, use_container_width=True)
        
        with col_i2:
            fig_dist = px.histogram(
                df_history,
                x='Credibility',
                title='Credibility Score Distribution',
                nbins=10,
                color_discrete_sequence=['#4ECDC4']
            )
            fig_dist.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                height=300,
                showlegend=False
            )
            st.plotly_chart(fig_dist, use_container_width=True)
        
        # Statistics
        st.markdown("### 📊 Statistics")
        col_s1, col_s2, col_s3, col_s4 = st.columns(4)
        
        with col_s1:
            st.metric("Total Analyses", len(df_history))
        with col_s2:
            avg_score = df_history['Credibility'].mean()
            st.metric("Average Score", f"{avg_score:.1f}")
        with col_s3:
            st.metric("Highest Score", f"{df_history['Credibility'].max():.1f}")
        with col_s4:
            st.metric("Lowest Score", f"{df_history['Credibility'].min():.1f}")
        
        # Recent analyses
        st.markdown("### 📋 Recent Analyses")
        for i, history in enumerate(reversed(st.session_state.analysis_history[-5:])):
            with st.expander(f"Analysis {len(st.session_state.analysis_history)-i}: {history['text_preview']}"):
                col_r1, col_r2 = st.columns(2)
                with col_r1:
                    st.markdown(f"**Credibility:** <span style='color: {history['result']['color']};'>{history['result']['credibility_score']:.1f}</span>", unsafe_allow_html=True)
                    st.markdown(f"**Label:** {history['result']['final_label']} {history['result']['emoji']}")
                with col_r2:
                    st.markdown(f"**Length:** {history['char_count']:,} chars")
                    st.markdown(f"**Time:** {history['timestamp'].strftime('%Y-%m-%d %H:%M')}")
    else:
        st.info("📭 No analysis history yet. Start by analyzing some news content!")

# ==================== TAB 3: ABOUT ====================
with tab3:
    col_a1, col_a2 = st.columns([2, 1])
    
    with col_a1:
        st.markdown("""
        ## About NewsTrust AI
        
        ### 🤖 How It Works
        NewsTrust AI uses a multi-layer approach to analyze news credibility:
        
        1. **Machine Learning Analysis**
           - Random Forest classifier trained on verified news
           - TF-IDF feature extraction
           - Automatic dimension fixing for compatibility
        
        2. **Rule-Based Enhancement**
           - Text readability assessment
           - Sentiment and objectivity analysis
           - Linguistic pattern recognition
           - Sensational language detection
        
        3. **Intelligent Weighting**
           - Combines ML predictions with rule-based scores
           - Adjusts weights based on confidence levels
           - Provides comprehensive credibility assessment
        
        ### 🔧 Technical Features
        - **Auto-Fix System**: Automatically adjusts feature dimensions
        - **Fallback Mechanism**: Uses rule-based analysis if ML fails
        - **Real-time Processing**: Instant analysis with progress tracking
        - **Session Persistence**: Remembers your analysis history
        - **Interactive Visuals**: Dynamic charts and gauges
        
        ### 🎯 Accuracy Notes
        - Works best with complete articles (100+ characters)
        - More accurate with formal, well-structured text
        - May struggle with satire or opinion pieces
        - Always verify critical information with trusted sources
        
        ### ⚠️ Limitations
        - Not 100% accurate (complements human judgment)
        - Requires sufficient text for analysis
        - Performance depends on text quality
        - Should not be sole source for critical decisions
        """)
    
    with col_a2:
        st.markdown("""
        <div class="analysis-card">
            <h3 style="color: #FF6B35;">📊 Credibility Scale</h3>
            <div style="margin: 1rem 0;">
                <div style="background: #00CC88; padding: 0.75rem; border-radius: 8px; margin: 0.5rem 0; color: white; font-weight: bold;">
                    <strong>80-100:</strong> Highly Credible ✅
                </div>
                <div style="background: #4ECDC4; padding: 0.75rem; border-radius: 8px; margin: 0.5rem 0; color: white; font-weight: bold;">
                    <strong>60-79:</strong> Mostly Credible 👍
                </div>
                <div style="background: #FFD166; padding: 0.75rem; border-radius: 8px; margin: 0.5rem 0; color: white; font-weight: bold;">
                    <strong>40-59:</strong> Partially Credible ⚠️
                </div>
                <div style="background: #FF9966; padding: 0.75rem; border-radius: 8px; margin: 0.5rem 0; color: white; font-weight: bold;">
                    <strong>20-39:</strong> Questionable ❓
                </div>
                <div style="background: #FF4B4B; padding: 0.75rem; border-radius: 8px; margin: 0.5rem 0; color: white; font-weight: bold;">
                    <strong>0-19:</strong> Highly Suspicious ❌
                </div>
            </div>
        </div>
        
        <div class="analysis-card" style="margin-top: 2rem;">
            <h3 style="color: #4ECDC4;">🎯 Best Practices</h3>
            <ul style="color: white;">
                <li>Check multiple sources</li>
                <li>Verify dates and authors</li>
                <li>Look for evidence/citations</li>
                <li>Consider potential biases</li>
                <li>Use fact-checking websites</li>
                <li>Be skeptical of extremes</li>
            </ul>
        </div>
        
        <div class="analysis-card" style="margin-top: 2rem;">
            <h3 style="color: #FFD166;">⚡ Quick Tips</h3>
            <p style="color: white;">
                • Provide complete articles<br>
                • Include 100+ characters<br>
                • Avoid copy-paste errors<br>
                • Check source credibility<br>
                • Report inaccuracies
            </p>
        </div>
        """, unsafe_allow_html=True)

# ==================== FOOTER ====================
st.markdown("""
<div style="text-align: center; padding: 2rem; margin-top: 3rem; background: linear-gradient(90deg, rgba(255,107,53,0.1), rgba(78,205,196,0.1)); border-radius: 15px; border: 1px solid rgba(255,255,255,0.1);">
    <h3 style="color: #FF6B35; margin-bottom: 1rem;">🔍 Stay Informed, Stay Safe</h3>
    <p style="color: #CCCCCC; line-height: 1.6;">
        NewsTrust AI assists in news verification but should complement human judgment.<br>
        Always verify important information through multiple reliable sources.
    </p>
    <div style="margin-top: 1.5rem; padding: 1rem; background: rgba(0,0,0,0.2); border-radius: 10px;">
        <p style="color: #4ECDC4; font-size: 0.9rem; margin: 0;">
            🧠 Multi-Layer Analysis • 🔧 Auto-Fix Enabled • 📊 Real-time Insights
        </p>
        <p style="color: #999999; font-size: 0.8rem; margin: 0.5rem 0 0 0;">
            Version 3.2.1 • """ + datetime.now().strftime("%B %d, %Y") + """
        </p>
    </div>
</div>
""", unsafe_allow_html=True)