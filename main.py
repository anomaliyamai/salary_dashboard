import relations
from time import sleep
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from streamlit_option_menu import option_menu


def run():
    st.set_page_config(
        page_title="Dashboard",
        page_icon="📊",
        layout="wide"
    )

    @st.cache_data
    def load_data(the_file_path):
        df = pd.read_csv(the_file_path)
        return df

    @st.cache_data
    def load_linear_regression_model(model_path):
        return pd.read_pickle(model_path)

    df = load_data("Salary Data.csv")
    model = load_linear_regression_model(
        "random_forest_regressor_salary_predictor_v1.pkl")
    header = st.container()
    content = st.container()

    with st.sidebar:
        st.title("Salary Stats")
        page = option_menu(
            menu_title=None,
            options=['Home', 'Relations & Correlations', 'Prediction'],

            menu_icon="cast",
            default_index=0
        )

        if page == "Home":

            with header:
                st.header('Employee Salary')

            with content:
                st.dataframe(df.sample(frac=0.3, random_state=35).reset_index(drop=True),
                             use_container_width=True)

                st.write("***")

                st.subheader("Data Summary Overview")

                len_numerical_data = df.select_dtypes(
                    include="number").shape[1]
                len_string_data = df.select_dtypes(include="object").shape[1]

                if len_numerical_data > 0:
                    st.subheader("Numerical Data")

                    data_stats = df.describe().T
                    st.table(data_stats)

                if len_string_data > 0:
                    st.subheader("String Data")

                    data_stats = df.select_dtypes(
                        include="object").describe().T
                    st.table(data_stats)

        if page == "Relations & Correlations":
            with header:
                st.header("Correlations Between Data")

            with content:
                st.plotly_chart(relations.get_avg_salary_by_gender(df),
                                use_container_width=True)

                st.header("Mean salary by job title")
                titles = df['Job Title'].unique()
                job_type = st.selectbox("Choose job title: ",
                                        titles.tolist())
                st.write(df[df['Job Title'] == job_type]['Salary'].mean())

                st.plotly_chart(relations.create_heat_map(df),
                                use_container_width=True)

                st.plotly_chart(relations.create_scatter_matrix(
                    df), use_container_width=True)

                st.write("***")
                col1, col2 = st.columns(2)
                with col1:
                    first_feature = st.selectbox(
                        "First Feature", options=(df.select_dtypes(
                            include="number").columns.tolist()), index=0).strip()

                temp_columns = df.select_dtypes(
                    include="number").columns.to_list().copy()

                temp_columns.remove(first_feature)

                with col2:
                    second_feature = st.selectbox(
                        "Second Feature", options=(temp_columns), index=0).strip()

                st.plotly_chart(relations.create_relation_scatter(
                    df, first_feature, second_feature), use_container_width=True)

        if page == "Prediction":
            with header:
                st.header("Prediction Model")

            with content:
                with st.form("Predict_value"):

                    c1, c2 = st.columns(2)
                    with c1:
                        age = st.number_input(
                            'Employee Age', min_value=20, max_value=60, value=24)

                    with c2:
                        exp_year = st.number_input(

                            'Experience Years', min_value=0, max_value=30, value=2)

                    education_level = st.selectbox(
                        "Education Level", options=["Bachelor's", "Master's", "PhD"])

                    st.write("")

                    predict_button = st.form_submit_button(
                        label='Predict', use_container_width=True)

                    st.write("***")

                    if predict_button:
                        education = [0, 0]  # Bachelor's

                        if education_level == "Master's":
                            education = [1, 0]

                        elif education_level == "PhD":
                            education = [0, 1]

                        with st.spinner(text='Predict The Value..'):
                            new_data = [age, exp_year]
                            new_data.extend(education)

                            predicted_value = f"{model.predict([new_data])[0]:,.0f}"
                            sleep(1.2)

                            predicted_col, score_col = st.columns(2)

                            with predicted_col:
                                st.subheader("Expected Salary")
                                st.subheader(f"${predicted_value}")

                            with score_col:
                                st.subheader("Model Accuracy")
                                st.subheader(f"{np.round(91.85, 2)}%")


run()
