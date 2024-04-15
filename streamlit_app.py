import streamlit as st
from DataProcessor.DataLoader import DataLoader
from DataProcessor.DataEvaluator import DataEvaluator
from DataProcessor.GraphicGenerator import GraphicGenerator
from DataProcessor.LinearRegressor import LinearRegressor
from DataProcessor.LogisticRegressor import LogisticRegressor

# To run the APP: streamlit run streamlit_app.py
if __name__ == '__main__':
    st.header('Analisis data')

    # Data Loader
    st.header('Data loader')
    dataLoader = DataLoader()
    dataLoader.check_labels()
    dataLoader.check_separator()
    file = dataLoader.load_file()

    if file is not None:
        df = dataLoader.load_data(file)

    # Model Selection Dropdown
        st.header('Model Selection')
        selected_model = st.selectbox('Select Model', ['Linear Regression', 'Logistic Regression'])

  # Linear Regression
        if selected_model == 'Linear Regression':
            st.header('Linear Regression')
            regressor = LinearRegressor(df)
            regressor.linear()
            st.markdown('<hr/>', unsafe_allow_html=True)

        # Logistic Regression
        elif selected_model == 'Logistic Regression':
            st.header('Logistic Regression')
            regressor = LogisticRegressor(df)
            regressor.logistic()
            st.markdown('<hr/>', unsafe_allow_html=True)


        # Data evaluation
        st.header('Data evaluation')
        st.write('Non-numeric columns and rows with missing values have been dropped.')
        dataEvaluator = DataEvaluator(df)
        dataEvaluator.show_head()
        dataEvaluator.show_dimensions()
        dataEvaluator.show_columns()

        # Graphic Plots
        st.header('Graphic Plots')
        plotGenerator = GraphicGenerator(df)

        checked_pairplot = st.checkbox('PairPlot')
        checked_scatterPlot = st.checkbox('ScatterPlot')
        checked_correlationPlot = st.checkbox('Correlation')
        checked_linearRegPlot = st.checkbox('LinearRegPlot')
        checked_logisticRegPlot = st.checkbox('LogisticRegPlot')

        if checked_pairplot:
            plotGenerator.pairplot()
            st.markdown('<hr/>', unsafe_allow_html=True)

        if checked_scatterPlot:
            plotGenerator.scatterplot()
            st.markdown('<hr/>', unsafe_allow_html=True)

        if checked_correlationPlot:
            plotGenerator.correlationPlot()
            st.markdown('<hr/>', unsafe_allow_html=True)

        if checked_linearRegPlot:
            plotGenerator.linearRegressionPlot()
            st.markdown('<hr/>', unsafe_allow_html=True)

        if checked_logisticRegPlot:
            plotGenerator.logisticRegressionPlot()
            st.markdown('<hr/>', unsafe_allow_html=True)

        # # Linear Regression
        # st.header('Linear Regression')
        # regressor = LinearRegressor(df)
        # regressor.linear()
        # st.markdown('<hr/>', unsafe_allow_html=True)

        #  # Logistic Regression
        # st.header('Logistic Regression')
        # regressor = LogisticRegressor(df)
        # regressor.logistic()
        # st.markdown('<hr/>', unsafe_allow_html=True)
        
        # Linear Regression
      