import io
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Title of the app
st.title('Exploratory Data Analysis with Streamlit')

# File uploader
uploaded_file = st.file_uploader("Upload CSV file here", type="csv")

if uploaded_file is not None:
    # Load the data
    df = pd.read_csv(uploaded_file, delimiter=";")

    # Display the first few rows of the dataframe
    st.subheader('Data Preview')
    st.write(df.head())

    # Display summary statistics
    st.subheader('Summary Statistics')
    st.write(df.describe())

    # Display dataset information
    st.subheader('Data Info')
    buffer = io.StringIO()
    df.info(buf=buffer)
    s = buffer.getvalue()

    filtered_info = "\n".join(s.split('\n')[1:])
    st.text(filtered_info)

    # Display missing values
    st.subheader('Missing Values')
    st.write(df.isnull().sum())
    df = df.fillna(df.select_dtypes(include=[float, int]).mean())

    # Pie Chart
    st.subheader('Pie Chart')
    categorical_cols = df.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        selected_col = st.selectbox('Select a category you want to see on the piechart:', categorical_cols)

        if selected_col:
            category_counts = df[selected_col].value_counts()
            fig, ax = plt.subplots(figsize=(6, 6))
            wedges, _, autotexts = ax.pie(category_counts, autopct='%1.1f%%', startangle=140, textprops={'fontsize': 8}, radius=1)
            ax.legend(wedges, category_counts.index, title=selected_col, loc="center left", bbox_to_anchor=(1, 0.5))
            ax.set_title(f'Pie Chart of {selected_col}')
            st.pyplot(fig)
    else:
        st.write("No categorical columns found in the dataset.")

    # Plot histograms for each numerical feature in two columns
    st.subheader('Histograms')
    num_cols = df.select_dtypes(include=[np.number]).columns
    col1, col2 = st.columns(2)
    with col1:
        for col in num_cols[:len(num_cols)//2]:
            fig, ax = plt.subplots()
            df[col].hist(ax=ax, bins=20)
            ax.set_title(f'Histogram of {col}')
            st.pyplot(fig)
    with col2:
        for col in num_cols[len(num_cols)//2:]:
            fig, ax = plt.subplots()
            df[col].hist(ax=ax, bins=20)
            ax.set_title(f'Histogram of {col}')
            st.pyplot(fig)

    # Plot density plots in two columns
    st.subheader('Density Plots')
    col1, col2 = st.columns(2)
    with col1:
        for col in num_cols[:len(num_cols)//2]:
            fig, ax = plt.subplots()
            sns.kdeplot(df[col], ax=ax, fill=True)
            ax.set_title(f'Density Plot of {col}')
            st.pyplot(fig)
    with col2:
        for col in num_cols[len(num_cols)//2:]:
            fig, ax = plt.subplots()
            sns.kdeplot(df[col], ax=ax, fill=True)
            ax.set_title(f'Density Plot of {col}')
            st.pyplot(fig)

    # Plot box and whisker plots in two columns
    st.subheader('Box and Whisker Plots')
    col1, col2 = st.columns(2)
    with col1:
        for col in num_cols[:len(num_cols)//2]:
            fig, ax = plt.subplots()
            sns.boxplot(x=df[col], ax=ax)
            ax.set_title(f'Box and Whisker Plot of {col}')
            st.pyplot(fig)
    with col2:
        for col in num_cols[len(num_cols)//2:]:
            fig, ax = plt.subplots()
            sns.boxplot(x=df[col], ax=ax)
            ax.set_title(f'Box and Whisker Plot of {col}')
            st.pyplot(fig)
    
    # Scatter Plot
    st.subheader('Scatter Plot')
    # Select two numerical columns for scatter plot
    x_col = st.selectbox('Select X-axis column', num_cols)
    y_col = st.selectbox('Select Y-axis column', num_cols)

    # Plot the scatter plot
    fig, ax = plt.subplots()
    sns.scatterplot(x=df[x_col], y=df[y_col], ax=ax)
    ax.set_title(f'Scatter Plot of {x_col} vs {y_col}')
    st.pyplot(fig)

    # Plot correlations heatmap
    st.subheader('Correlation Heatmap')
    corr = df.select_dtypes(include=[float, int]).corr()
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', ax=ax,
                annot_kws={"size": 10}, cbar_kws={'shrink': .8}, 
                linewidths=0.5, linecolor='gray')
    
    # Rotate labels
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=10)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=10)
    
    st.pyplot(fig)
