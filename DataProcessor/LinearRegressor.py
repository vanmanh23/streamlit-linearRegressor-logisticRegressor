import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression


class LinearRegressor:

    def __init__(_self, df):
        _self.df = df.select_dtypes(include=['number']).dropna()
        _self.columns = _self.df.columns.values

    def linear(_self):
        X = st.multiselect('X', _self.columns, default=None, key='x_linear_reg')
        y = st.selectbox('y', _self.columns, key='y_linear_reg')
        if st.button('Calculate linear regression'):
            if X:
                if y:
                    # Training the model
                    reg = LinearRegression()
                    reg.fit(_self.df[X], _self.df[[y]])
                    st.write('Training score: ' + str(round(reg.score(_self.df[X], _self.df[[y]]), 3)))

                    # Formatting coefficients for printing
                    coefficients_values = reg.coef_[0].tolist() + [reg.intercept_[0]]
                    coefficients_names = X.copy() + ['Constant']
                    coefficients_df = pd.DataFrame([coefficients_values], columns=coefficients_names,
                                                   index=['Coefficients'])

                    st.table(coefficients_df)