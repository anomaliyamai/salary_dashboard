import pandas as pd
import numpy as np
import plotly.express as px


import streamlit as st
import warnings

pd.set_option('future.no_silent_downcasting', True)
pd.options.mode.copy_on_write = "warn"


def create_heat_map(the_df):
    correlation = the_df.corr(numeric_only=True)

    fig = px.imshow(
        correlation,
        template="plotly_dark",
        text_auto="0.2f",
        aspect=1,
        color_continuous_scale="purpor",
        title="Correlation Heatmap of Data",
        height=650,
    )
    fig.update_traces(
        textfont={
            "size": 16,
            "family": "consolas"
        },

    )
    fig.update_layout(
        title={
            "font": {
                "size": 30,
                "family": "tahoma"
            }
        },
        hoverlabel={
            "bgcolor": "#111",
            "font_size": 15,
            "font_family": "consolas"
        }
    )
    return fig


def create_scatter_matrix(the_df):

    fig = px.scatter_matrix(
        the_df,
        dimensions=the_df.select_dtypes(include="number").columns,
        height=800,
        color=the_df.iloc[:, -1],
        opacity=0.65,
        title="Relationships Between Numerical Data",
        template="plotly_dark"

    )

    fig.update_layout(
        title={
            "font": {
                "size": 30,
                "family": "tahoma"
            }
        },
        hoverlabel={
            "bgcolor": "#111",
            "font_size": 14,
            "font_family": "consolas"
        }
    )
    return fig


def create_relation_scatter(the_df, x, y):

    fig = px.scatter(
        data_frame=the_df,
        x=x,
        y=y,
        color=y,
        opacity=0.78,
        title="Predicted Vs. Actual",
        template="plotly_dark",
        trendline="ols",
        height=650

    )

    fig.update_layout(
        title={
            "font": {
                "size": 28,
                "family": "tahoma"
            }
        }
    )
    return fig


def get_avg_salary_by_gender(the_df):
    salary_by_gender = the_df.groupby("Gender")["Salary"].mean().sort_values(ascending=False)
    salary_by_gender.apply(lambda x: f"${x:,.2f}")
    fig = px.bar(data_frame=salary_by_gender,
                 x=salary_by_gender.index,
                 y=salary_by_gender,
                 color=salary_by_gender.index,
                 title="Average salary by gender",
                 labels={"index": "Education", "y": "Frequency in %"},
                 template="plotly_dark",
                 text_auto="0.4s"
                 )
    return fig
