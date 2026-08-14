import streamlit as st
import joblib
import pandas as pd
import numpy as np

pipeline = joblib.load("pipeline_autism_xgb.pkl")

st.set_page_config(page_title="Q-chat-10 ASD Predictor", layout="centered")
st.title(" Q-chat-10 ASD Predictor")
st.markdown("Answer the questions below to estimate the risk of Autism Spectrum Disorder (ASD) in toddlers.")

A1  = st.selectbox("A1 - Do you make eye contact when someone calls your name?", ["No", "Yes"])
A2  = st.selectbox("A2 - Is it easy for others to get eye contact with you?", ["No", "Yes"])
A3  = st.selectbox("A3 - Do you point to request something you want?", ["No", "Yes"])
A4  = st.selectbox("A4 - Do you point to share something interesting with others?", ["No", "Yes"])
A5  = st.selectbox("A5 - Do you engage in imaginative activities or role-play?", ["No", "Yes"])
A6  = st.selectbox("A6 - Do you follow others' gaze to see what they're looking at?", ["No", "Yes"])
A7  = st.selectbox("A7 - Do you try to comfort someone who appears upset?", ["No", "Yes"])
A8  = st.selectbox("A8 - Would you describe your communication style as typical?", ["No", "Yes"])
A9  = st.selectbox("A9 - Do you use common gestures (like waving goodbye)?", ["No", "Yes"])
A10 = st.selectbox("A10 - Do you sometimes stare into space without focus?", ["No", "Yes"])

sex        = st.selectbox("Your gender", ["Male", "Female"])
jaundice   = st.selectbox("Were you born with jaundice?", ["No", "Yes"])
family_asd = st.selectbox("Is there a family history of autism?", ["No", "Yes"])
age        = st.number_input("Your age (in years)", 1, 100, 25, step=1)


# Function to convert Yes/No to 0/1 for A1-A10 questions
def convert_response(response):
    return 0 if response == "Yes" else 1

# Create display dictionary with original responses
display_dict = {
    "A1": A1,
    "A2": A2,
    "A3": A3,
    "A4": A4,
    "A5": A5,
    "A6": A6,
    "A7": A7,
    "A8": A8,
    "A9": A9,
    "A10": A10,
    "Sex": sex,
    "Jaundice": jaundice,
    "Family_ASD": family_asd,
    "Age": age
}

# Create input dictionary with converted values for model prediction
model_input_dict = {
    "A1": convert_response(A1),
    "A2": convert_response(A2),
    "A3": convert_response(A3),
    "A4": convert_response(A4),
    "A5": convert_response(A5),
    "A6": convert_response(A6),
    "A7": convert_response(A7),
    "A8": convert_response(A8),
    "A9": convert_response(A9),
    "A10": convert_response(A10),
    "Sex": sex,
    "Jaundice": jaundice,
    "Family_ASD": family_asd,
    "Age": age
}

st.markdown("---")
st.subheader("Prediction")

if st.button("Predict ASD Risk", type="primary"):
    try:
        # Use the converted values for model prediction
        input_df = pd.DataFrame([model_input_dict])
        
        # Show the original responses (not converted)
        display_df = pd.DataFrame([display_dict])
        st.write("Input data:", display_df)
        
        prediction = pipeline.predict(input_df)[0]
        prediction_proba = pipeline.predict_proba(input_df)[0]
        
        if prediction == 1:
            st.error(" **High Risk of ASD**")
            st.markdown(f"**Probability of ASD:** {prediction_proba[1]:.2%}")
            st.markdown("**Recommendation:** Consider consulting with a pediatric specialist for further evaluation.")
        else:
            st.success(" **Low Risk of ASD**")
            st.markdown(f"**Probability of ASD:** {prediction_proba[1]:.2%}")
            st.markdown("**Note:** This is a screening tool and not a diagnostic instrument.")
        
        confidence = max(prediction_proba)
        st.info(f"**Model Confidence:** {confidence:.2%}")
        
        with st.expander("View Input Summary"):
            # Show only the original responses
            st.json(display_dict)
            
    except Exception as e:
        st.error(f"An error occurred during prediction: {str(e)}")
        
        # Debug information
        st.write("Debug Info:")
        st.write("Input DataFrame shape:", input_df.shape if 'input_df' in locals() else "Not created")
        st.write("Input DataFrame columns:", input_df.columns.tolist() if 'input_df' in locals() else "Not created")
        st.write("Input DataFrame dtypes:", input_df.dtypes if 'input_df' in locals() else "Not created")

# Add disclaimer
st.markdown("---")
st.markdown("""
**Disclaimer:** This tool is for screening purposes only and should not replace professional medical advice. 
The Q-CHAT-10 is a validated screening tool, but a definitive diagnosis requires comprehensive evaluation 
by qualified Healthcare professionals.
""")