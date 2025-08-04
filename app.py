import streamlit as st
import pandas as pd
import joblib
import numpy as np
import sklearn

# --- Page Configuration ---
st.set_page_config(
    page_title="Restaurant Food Waste Cost Predictor",
    page_icon="💰",
    layout="centered",
    initial_sidebar_state="auto"
)

# --- Data and Model Loading ---

@st.cache_resource
def load_model(model_path):
    """Loads the trained machine learning model with robust error handling."""
    try:
        model = joblib.load(model_path)
        return model
    except FileNotFoundError:
        st.error(f"🚨 **Model file not found!** Please make sure `{model_path}` is in the same directory.")
        return None
    except AttributeError:
        # This is the specific error for scikit-learn version mismatch
        st.error(
            """
            **AttributeError: Model Version Mismatch**

            This error means the `scikit-learn` version used to save the model is different from the one running this app.

            **To fix this:**
            1. Stop this app (press `Ctrl+C` in your terminal).
            2. Re-run the training script: `python3 food_waste_prediction_model.py`
            3. Relaunch this app: `streamlit run app.py`
            """
        )
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred while loading the model: {e}")
        return None


@st.cache_data
def load_unique_values_from_csv(data_path):
    """Loads unique values for dropdowns from the original dataset."""
    try:
        df = pd.read_csv(data_path)
        ingredients = sorted(df['Ingredient_Name'].unique())
        days = sorted(df['Day_of_Week'].unique())
        weather_conditions = sorted(df['Weather'].unique())
        return ingredients, days, weather_conditions
    except FileNotFoundError:
        st.error(f"🚨 **Dataset not found!** Please make sure `{data_path}` is in the same directory.")
        return None, None, None

# Load the model and unique values for dropdowns
model = load_model('food_waste_predictor.joblib')
ingredients, days, weather_conditions = load_unique_values_from_csv('restaurant_food_waste_dataset.csv')

# --- UI and Application Logic ---
st.title("💰 Restaurant Food Waste Cost Predictor")
st.markdown("Forecast the **financial cost** and **quantity** of ingredient wastage to make better purchasing decisions.")

if model and ingredients:
    # --- User Input Form ---
    with st.form("prediction_form"):
        st.header("Enter Prediction Details")
        
        col1, col2 = st.columns(2)

        with col1:
            day_of_week = st.selectbox("📅 Day of the Week", options=days)
            weather = st.selectbox("🌦️ Weather", options=weather_conditions)
            ingredient_name = st.selectbox("🥕 Ingredient Name", options=ingredients)
            # User input for cost in Rupees
            cost_per_kg = st.number_input("💰 Cost per kg (₹)", min_value=0.0, value=400.00, step=1.00, format="%.2f")
        
        with col2:
            special_event = st.radio("🎉 Special Event?", ("Yes", "No"), index=1, help="Was there a special event like a holiday or promotion?")
            ingredient_ordered = st.number_input("📦 Ingredient Ordered (kg)", min_value=0.1, step=0.1, format="%.2f")
            dish_sales_count = st.number_input("📈 Dish Sales Count", min_value=0, step=1)

        submit_button = st.form_submit_button(label="🔮 Predict Wastage")

    # --- Prediction Logic ---
    if submit_button:
        # Calculate the sales per kg feature
        sales_per_kg_ordered = dish_sales_count / ingredient_ordered if ingredient_ordered > 0 else 0
        special_event_numeric = 1 if special_event == "Yes" else 0

        # Create a DataFrame for the model using user-provided cost.
        # The column names must exactly match those used in the training script.
        input_data = pd.DataFrame({
            'Special_Event': [special_event_numeric],
            'Ingredient_Ordered': [ingredient_ordered],
            'Dish_Sales_Count': [dish_sales_count],
            'Sales_per_kg_Ordered': [sales_per_kg_ordered],
            'Cost_per_kg': [cost_per_kg],
            'Day_of_Week': [day_of_week],
            'Weather': [weather],
            'Ingredient_Name': [ingredient_name]
        })

        st.markdown("---")
        st.subheader("📊 Input for Prediction")
        st.dataframe(input_data)

        # Make the prediction
        try:
            prediction = model.predict(input_data)
            predicted_cost = prediction[0]
            
            # --- MODIFIED SECTION START ---
            # Calculate predicted wastage in kg
            predicted_wastage_kg = 0
            if cost_per_kg > 0:
                predicted_wastage_kg = predicted_cost / cost_per_kg

            st.subheader("✅ Prediction Results")
            
            res_col1, res_col2 = st.columns(2)
            with res_col1:
                st.metric(label="Predicted Cost of Wastage", value=f"₹{predicted_cost:.2f}")

            with res_col2:
                st.metric(label=f"Predicted Wastage ({ingredient_name})", value=f"{predicted_wastage_kg:.2f} kg")
            
            st.info("This is the estimated financial loss and quantity of waste for this ingredient under the given conditions.")
            # --- MODIFIED SECTION END ---

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")

# --- Instructions and Info Section ---
st.sidebar.header("Instructions")
st.sidebar.info(
    "1. **Train Model**: Run the `food_waste_prediction_model.py` script to generate the `.joblib` file.\n\n"
    "2. **Relaunch App**: If you retrain the model or encounter errors, restart this app by running `streamlit run app.py` in your terminal."
)
st.sidebar.header("Version Info")
st.sidebar.write(f"Scikit-learn version in this app: `{sklearn.__version__}`")
