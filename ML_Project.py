import streamlit as st
import pandas as pd
import joblib

# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="CardioCheck | Heart Risk Prediction",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------
# Load model files
# ---------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("KNN_heart.pkl")
    scaler = joblib.load("scalar.pkl")
    expected_columns = joblib.load("columns.pkl")
    return model, scaler, expected_columns


try:
    model, scaler, expected_columns = load_artifacts()
except Exception as e:
    st.error("Unable to load the model files.")
    st.code(str(e))
    st.stop()


# ---------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    * {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background:
            radial-gradient(circle at 10% 10%, rgba(239, 68, 68, 0.08), transparent 28%),
            radial-gradient(circle at 90% 15%, rgba(59, 130, 246, 0.08), transparent 25%),
            #f7f9fc;
    }

    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    .block-container {
        max-width: 1250px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #111827 0%, #1f2937 100%);
    }

    section[data-testid="stSidebar"] * {
        color: #f9fafb !important;
    }

    .brand {
        font-size: 25px;
        font-weight: 800;
        letter-spacing: -0.8px;
        margin-bottom: 4px;
    }

    .brand span {
        color: #ef4444;
    }

    .sidebar-text {
        color: #d1d5db !important;
        font-size: 14px;
        line-height: 1.7;
    }

    .side-card {
        margin-top: 28px;
        padding: 18px;
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 16px;
        background: rgba(255,255,255,0.06);
    }

    .side-card-title {
        font-weight: 700;
        margin-bottom: 8px;
    }

    /* Hero */
    .hero {
        padding: 35px 38px;
        border-radius: 25px;
        background: linear-gradient(135deg, #111827 0%, #243244 100%);
        color: white;
        box-shadow: 0 18px 45px rgba(15, 23, 42, 0.12);
        margin-bottom: 25px;
    }

    .hero-badge {
        display: inline-block;
        padding: 7px 12px;
        border-radius: 999px;
        background: rgba(239, 68, 68, 0.16);
        color: #fecaca;
        font-size: 12px;
        font-weight: 700;
        margin-bottom: 15px;
    }

    .hero h1 {
        margin: 0;
        font-size: clamp(30px, 4vw, 50px);
        line-height: 1.08;
        letter-spacing: -1.8px;
    }

    .hero h1 span {
        color: #f87171;
    }

    .hero p {
        margin: 15px 0 0;
        color: #d1d5db;
        font-size: 16px;
        max-width: 760px;
        line-height: 1.7;
    }

    /* Section headings */
    .section-title {
        font-size: 22px;
        font-weight: 800;
        color: #111827;
        margin: 25px 0 5px;
    }

    .section-subtitle {
        color: #6b7280;
        margin-bottom: 18px;
    }

    /* Input cards */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 18px !important;
        border: 1px solid #e5e7eb !important;
        background: rgba(255,255,255,0.92);
        box-shadow: 0 8px 25px rgba(15, 23, 42, 0.05);
    }

    .input-card-title {
        font-size: 16px;
        font-weight: 800;
        color: #111827;
        margin-bottom: 10px;
    }

    /* Make all form labels clearly visible on the light page */
    .stApp [data-testid="stWidgetLabel"] p,
    .stApp [data-testid="stWidgetLabel"] label,
    .stApp [data-testid="stWidgetLabel"] div,
    .stApp label,
    .stApp .stMarkdown p {
        color: #111827 !important;
    }

    .stApp [data-testid="stWidgetLabel"] p {
        font-size: 14px !important;
        font-weight: 700 !important;
    }

    .stApp [data-testid="stWidgetLabel"] small,
    .stApp [data-testid="stWidgetLabel"] span {
        color: #6b7280 !important;
    }

    /* Keep sidebar text white */
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
        color: #f9fafb !important;
    }

    /* Input text */
    .stApp input,
    .stApp textarea,
    .stApp [data-baseweb="select"] *,
    .stApp [data-baseweb="input"] * {
        color: #111827 !important;
    }

    .stApp [data-baseweb="select"] {
        background-color: #ffffff !important;
    }

    /* Predict button */
    div.stButton > button {
        width: 100%;
        border: none;
        border-radius: 14px;
        min-height: 52px;
        font-size: 16px;
        font-weight: 800;
        background: linear-gradient(135deg, #ef4444, #dc2626);
        color: white;
        box-shadow: 0 10px 25px rgba(220, 38, 38, 0.22);
        transition: all 0.2s ease;
    }

    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 14px 30px rgba(220, 38, 38, 0.30);
    }

    /* Result */
    .result-card {
        padding: 25px;
        border-radius: 20px;
        margin-top: 25px;
        text-align: center;
        border: 1px solid #e5e7eb;
        background: white;
        box-shadow: 0 12px 35px rgba(15, 23, 42, 0.08);
    }

    .result-icon {
        font-size: 42px;
        margin-bottom: 5px;
    }

    .result-title {
        font-size: 24px;
        font-weight: 800;
        color: #111827;
    }

    .result-text {
        color: #6b7280;
        line-height: 1.6;
        margin-top: 8px;
    }

    .metric-box {
        padding: 15px;
        border-radius: 14px;
        background: #f9fafb;
        text-align: center;
        border: 1px solid #e5e7eb;
    }

    .metric-number {
        font-size: 22px;
        font-weight: 800;
        color: #111827;
    }

    .metric-label {
        font-size: 12px;
        color: #6b7280;
        margin-top: 3px;
    }

    .disclaimer {
        margin-top: 25px;
        padding: 15px 18px;
        border-radius: 14px;
        background: #fff7ed;
        border: 1px solid #fed7aa;
        color: #7c2d12;
        font-size: 13px;
        line-height: 1.6;
    }

    .footer {
        text-align: center;
        color: #9ca3af;
        font-size: 12px;
        margin-top: 35px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
with st.sidebar:
    st.markdown(
        '<div class="brand">Cardio<span>Check</span> ❤️</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="sidebar-text">
        Machine-learning based heart disease risk assessment.
        Enter the patient's information and click <b>Predict Risk</b>.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="side-card">
            <div class="side-card-title">How it works</div>
            <div class="sidebar-text">
                1. Enter patient details<br>
                2. Data is prepared and scaled<br>
                3. KNN model evaluates the input<br>
                4. Risk prediction is displayed
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="side-card">
            <div class="side-card-title">Important</div>
            <div class="sidebar-text">
                This application is an educational machine-learning
                project and should not be used as a medical diagnosis.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------
# Hero
# ---------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <div class="hero-badge">AI-POWERED HEALTH SCREENING</div>
        <h1>Understand Your <span>Heart Risk</span></h1>
        <p>
            Enter the patient's clinical information below to get a
            machine-learning based heart disease risk prediction.
            The result is intended for educational screening only.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# Patient information
# ---------------------------------------------------------
st.markdown(
    '<div class="section-title">Patient Information</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="section-subtitle">Basic demographic and clinical details</div>',
    unsafe_allow_html=True,
)

with st.container(border=True):
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.slider("Age", 18, 100, 40)

    with col2:
        sex = st.selectbox("Sex", ["M", "F"])

    with col3:
        resting_bp = st.number_input(
            "Resting Blood Pressure (mm Hg)",
            min_value=80,
            max_value=200,
            value=120,
            step=1,
        )


# ---------------------------------------------------------
# Clinical information
# ---------------------------------------------------------
st.markdown(
    '<div class="section-title">Clinical Information</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="section-subtitle">Select or enter the relevant health measurements</div>',
    unsafe_allow_html=True,
)

with st.container(border=True):
    col1, col2, col3 = st.columns(3)

    with col1:
        chest_pain = st.selectbox(
            "Chest Pain Type",
            ["ATA", "NAP", "TA", "ASY"],
            help="ATA = Atypical Angina, NAP = Non-Anginal Pain, TA = Typical Angina, ASY = Asymptomatic",
        )

        cholesterol = st.number_input(
            "Cholesterol (mg/dL)",
            min_value=100,
            max_value=600,
            value=200,
            step=1,
        )

    with col2:
        fasting_bs = st.selectbox(
            "Fasting Blood Sugar > 120 mg/dL",
            [0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No",
        )

        resting_ecg = st.selectbox(
            "Resting ECG",
            ["Normal", "ST", "LVH"],
        )

    with col3:
        max_hr = st.slider(
            "Maximum Heart Rate",
            60,
            220,
            150,
        )

        exercise_angina = st.selectbox(
            "Exercise-Induced Angina",
            ["Y", "N"],
            format_func=lambda x: "Yes" if x == "Y" else "No",
        )


# ---------------------------------------------------------
# ECG / exercise details
# ---------------------------------------------------------
st.markdown(
    '<div class="section-title">Exercise & ECG Details</div>',
    unsafe_allow_html=True,
)

with st.container(border=True):
    col1, col2 = st.columns(2)

    with col1:
        oldpeak = st.slider(
            "Oldpeak (ST Depression)",
            0.0,
            6.0,
            1.0,
            step=0.1,
        )

    with col2:
        st_slope = st.selectbox(
            "ST Slope",
            ["Up", "Flat", "Down"],
        )


# ---------------------------------------------------------
# Prediction
# ---------------------------------------------------------
st.markdown("<br>", unsafe_allow_html=True)

if st.button("❤️  Predict Heart Risk", type="primary"):

    # Raw input
    raw_input = {
        "Age": age,
        "RestingBP": resting_bp,
        "Cholesterol": cholesterol,
        "FastingBS": fasting_bs,
        "MaxHR": max_hr,
        "Oldpeak": oldpeak,
        "Sex_" + sex: 1,
        "ChestPainType_" + chest_pain: 1,
        "RestingECG_" + resting_ecg: 1,
        "ExerciseAngina_" + exercise_angina: 1,
        "ST_Slope_" + st_slope: 1,
    }

    input_df = pd.DataFrame([raw_input])

    # Add missing one-hot encoded columns
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    # Match training column order
    input_df = input_df[expected_columns]

    try:
        # Scale
        scaled_input = scaler.transform(input_df)

        # Prediction
        prediction = model.predict(scaled_input)[0]

        # Probability, if supported by the model
        probability = None
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(scaled_input)[0]
            if len(probabilities) > 1:
                probability = float(probabilities[1] * 100)

        # Result
        if prediction == 1:
            st.markdown(
                """
                <div class="result-card">
                    <div class="result-icon">⚠️</div>
                    <div class="result-title">Higher Risk Detected</div>
                    <div class="result-text">
                        The model predicts a higher likelihood of heart disease
                        based on the information provided.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div class="result-card">
                    <div class="result-icon">✅</div>
                    <div class="result-title">Lower Risk Detected</div>
                    <div class="result-text">
                        The model predicts a lower likelihood of heart disease
                        based on the information provided.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Metrics
        metric1, metric2, metric3 = st.columns(3)

        with metric1:
            st.markdown(
                f"""
                <div class="metric-box">
                    <div class="metric-number">{age}</div>
                    <div class="metric-label">Age</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with metric2:
            st.markdown(
                f"""
                <div class="metric-box">
                    <div class="metric-number">{max_hr}</div>
                    <div class="metric-label">Max Heart Rate</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with metric3:
            if probability is not None:
                metric_value = f"{probability:.1f}%"
                metric_label = "Model Probability"
            else:
                metric_value = "N/A"
                metric_label = "Probability Not Available"

            st.markdown(
                f"""
                <div class="metric-box">
                    <div class="metric-number">{metric_value}</div>
                    <div class="metric-label">{metric_label}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown(
            """
            <div class="disclaimer">
                <b>Medical disclaimer:</b> This prediction is generated by a
                machine-learning model and is for educational/informational
                purposes only. It is not a medical diagnosis. If you have
                symptoms or health concerns, consult a qualified healthcare
                professional.
            </div>
            """,
            unsafe_allow_html=True,
        )

    except Exception as e:
        st.error("Prediction failed. Please check that your model, scaler, and columns files match the training data.")
        st.code(str(e))


# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.markdown(
    """
    <div class="footer">
        CardioCheck • Machine Learning Project • Built with Streamlit ❤️
    </div>
    """,
    unsafe_allow_html=True,
)
