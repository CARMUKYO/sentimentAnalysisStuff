import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import datetime
from wordcloud import WordCloud
from pandas.api.types import is_numeric_dtype
from analysisModule import analyzeText, processCsv, calculateMetrics, runScraper
import plotly.express as px
from collections import Counter
import re
import nltk
from nltk.corpus import stopwords


# # **References**
# https://github.com/dataprofessor/dashboard-kit/        
        
# Page Title
st.set_page_config(page_title="Sentiment Analysis Dashboard", layout="wide")

# Sidebar
menuOptions = ["Upload/Scrape CSV & Analyze", "Analyze Text", "Visualizations"]
with st.sidebar:
    st.title("Sentiment Analysis Dashboard")
    selectedOption = st.sidebar.selectbox("⚙️ Select Option", menuOptions)

# Initialize Session State for Uploaded Data
if "uploadedDataFrames" not in st.session_state:
    st.session_state.uploadedDataFrames = {}
if "uploadedFileNames" not in st.session_state:
    st.session_state.uploadedFileNames = []

# Upload CSV & Scrape Data Section
if selectedOption == "Upload/Scrape CSV & Analyze":
    st.subheader("📂 Upload CSV Files or Scrape Data for Sentiment Analysis")

    # Upload CSV Files
    uploadedFiles = st.file_uploader("Choose CSV files", type=["csv"], accept_multiple_files=True)

    if uploadedFiles:
        for uploadedFile in uploadedFiles:
            if uploadedFile.name not in st.session_state.uploadedFileNames:
                st.session_state.uploadedFileNames.append(uploadedFile.name)
                filePath = os.path.join("uploads", uploadedFile.name)
                with open(filePath, "wb") as f:
                    f.write(uploadedFile.getbuffer())

                processedFileName = "analyzed_" + uploadedFile.name
                processCsv(filePath, processedFileName)  # Run sentiment analysis on uploaded CSV
                st.success(f"File {uploadedFile.name} processed successfully!")

                processedData = pd.read_csv(processedFileName)
                st.session_state.uploadedDataFrames[uploadedFile.name] = processedData  # Store uploaded data in session

    # Scrape Data
    st.subheader("🔍 Scrape Twitter Data")
    query = st.text_input("Enter keywords for  search (For date use: until:2023-02-09 since:2021-12-22 -filter:links)", "")
    limit = st.slider("Select number of data to be scrape", min_value=10, max_value=110, value=20)

    if st.button("Scrape Tweets"):
        if query:
            with st.spinner("Fetching tweets..."):
                    runScraper(query, limit) 
                    st.success(f"Scraping complete! {limit}")

                    # Process and display the scraped data
                    processedFile = "analyzed_ScrapeData.csv"
                    processCsv("ScrapeData.csv", processedFile)

                    # Load and display the scraped data
                    df = pd.read_csv(processedFile)
                    st.session_state.uploadedDataFrames["Scraped Twitter Data"] = df
                    st.session_state.uploadedFileNames.append("Scraped Twitter Data")
                    st.write("### Scraped Twitter Data")
        else:
            st.warning("Please enter a search query before scraping.")

    # Show previously uploaded files and scraped data
    if st.session_state.uploadedDataFrames:
        st.write("### Analyzed Files:")
        for fileName, dataFrame in st.session_state.uploadedDataFrames.items():
            st.write(f"#### {fileName}")
            st.dataframe(dataFrame)

# Analyze Text Section
elif selectedOption == "Analyze Text":
    st.subheader("Enter Text for Sentiment Analysis")
    
    userInputText = st.text_area("Enter text here...", "")

    if st.button("Analyze Sentiment"):
        if userInputText:
            analysisResults = analyzeText(userInputText)
            st.success("Analysis Complete!")
            st.write(f"**VADER Sentiment:** {analysisResults['vaderSentiment']}")
            st.write(f"**Naive Bayes Sentiment:** {analysisResults['naiveBayesSentiment']}")
            st.write(f"**SVM Sentiment:** {analysisResults['svmSentiment']}")
            st.write(f"**Logistic Regression Sentiment:** {analysisResults['logisticRegressionSentiment']}")

            # Store single text input results
            singleTextResult = {
                "text": userInputText,
                "VADER": analysisResults['vaderSentiment'],
                "Naive Bayes": analysisResults['naiveBayesSentiment'],
                "SVM": analysisResults['svmSentiment'],
                "Logistic Regression": analysisResults['logisticRegressionSentiment']
            }
            
            st.session_state.uploadedDataFrames["single_text"] = pd.DataFrame([singleTextResult])

        else:
            st.warning("Please enter text before analyzing.")

# Visualizations Section
elif selectedOption == "Visualizations":
    st.subheader("Visualizations")

    if st.session_state.uploadedDataFrames:
        # Select which dataset to visualize
        selectedFile = st.selectbox("Select Dataset", st.session_state.uploadedFileNames)
        dataFrame = st.session_state.uploadedDataFrames[selectedFile].copy()

        # Submenu for different visualizations
        visualizationOptions = ["Model Performance", "Sentiment Trends", "Correlation Heatmap", "Word Cloud"]
        selectedVisualization = st.selectbox("Select Visualization", visualizationOptions)

        if selectedVisualization == "Model Performance":
            st.subheader("Model Performance Comparison")

            # Define Models for Comparison
            modelPredictions = {
                "Naive Bayes": "naiveBayesSentiment",
                "SVM": "svmSentiment",
                "Logistic Regression": "logisticRegressionSentiment",
                "VADER": "vaderSentiment"
            }

            # Check if trueSentiment is available
            if "trueSentiment" in dataFrame.columns and dataFrame["trueSentiment"].notna().any():
                trueSentiment = dataFrame["trueSentiment"].tolist()
                st.info("Using manually annotated `trueSentiment` for evaluation.")
            else:
                trueSentiment = None
                st.warning("No `trueSentiment` found. Evaluating model agreement instead.")

            # Calculate Metrics for Each Model
            modelResults = {}
            if trueSentiment is not None:
                # Evaluate against true sentiment
                for modelName, column in modelPredictions.items():
                    if column in dataFrame.columns and dataFrame[column].notna().any():
                        predictedSentiment = dataFrame[column].tolist()
                        modelResults[modelName] = calculateMetrics(trueSentiment, predictedSentiment)
            else:
                # Evaluate model agreement
                consensusPredictions = []
                for i in range(len(dataFrame)):
                    predictions = [dataFrame[column].iloc[i] for column in modelPredictions.values() if column in dataFrame.columns]
                    if predictions:
                        consensusPredictions.append(max(set(predictions), key=predictions.count))

                # Compare each model's predictions to the consensus
                for modelName, column in modelPredictions.items():
                    if column in dataFrame.columns and dataFrame[column].notna().any():
                        predictedSentiment = dataFrame[column].tolist()
                        modelResults[modelName] = calculateMetrics(consensusPredictions, predictedSentiment)

            if not modelResults:
                st.warning("Not enough valid data for model comparison.")
            else:
                # Extract Metrics for Visualization
                def extractMetrics(report):
                    classes = ['pos', 'neg', 'neut']
                    metricsTypes = ['precision', 'recall', 'f1-score']
                    metricsList = [report.get('accuracy', 0)]  # Start with accuracy
                    for cls in classes:
                        for metric in metricsTypes:
                            metricsList.append(report.get(cls, {}).get(metric, 0))  # Ensure defaults
                    return metricsList

                # Prepare Data for Visualization
                models = list(modelResults.keys())
                metricsLabels = ['Accuracy'] + [f"{cls}_{metric}" for cls in ['pos', 'neg', 'neut'] for metric in ['precision', 'recall', 'f1-score']]
                modelMetrics = np.array([extractMetrics(modelResults[model]) for model in models])

                # Matplotlib Bar Chart
                fig, ax = plt.subplots(figsize=(15, 7))
                x = np.arange(len(metricsLabels))  # Set X-axis positions
                width = 0.15  # Width of each bar

                for i, model in enumerate(models):
                    ax.bar(x + (i - 1.5) * width, modelMetrics[i], width, label=model)

                ax.set_xlabel("Metrics")
                ax.set_ylabel("Scores")
                ax.set_title("Model Performance Comparison")
                ax.set_xticks(x)
                ax.set_xticklabels(metricsLabels, rotation=45, ha='right')
                ax.legend()

                ax.grid(True, axis='y', linestyle='--', alpha=0.7)

                plt.tight_layout()
                st.pyplot(fig)

        elif selectedVisualization == "Sentiment Trends":
            st.subheader("Sentiment Trends Over Time")
            if 'timestamp' not in dataFrame.columns:
                st.warning("The dataset does not contain a 'timestamp' column. Cannot plot sentiment trends over time.")
            else:
                try:
                    dataFrame['timestamp'] = pd.to_datetime(dataFrame['timestamp'], errors='coerce')
                except Exception as e:
                    st.error(f"Error converting 'timestamp' column to datetime: {e}")
                    st.stop()

                dataFrame = dataFrame.dropna(subset=['timestamp'])

                if dataFrame.empty:
                    st.warning("No valid timestamps found in the dataset.")
                else:
                    sentimentMapping = {"neg": -1, "neut": 0, "pos": 1}
                    sentimentColumns = ['vaderSentiment', 'naiveBayesSentiment', 'svmSentiment', 'logisticRegressionSentiment']

                    for col in sentimentColumns:
                        if col in dataFrame.columns:
                            dataFrame[col] = dataFrame[col].map(sentimentMapping)

                    if "likes" in dataFrame.columns and is_numeric_dtype(dataFrame["likes"]):
                        dataFrame['sentimentNumeric'] = dataFrame[sentimentColumns].mean(axis=1) * dataFrame["likes"].astype(float)
                    else:
                        dataFrame['sentimentNumeric'] = dataFrame[sentimentColumns].mean(axis=1)

                    dataFrame = dataFrame.dropna(subset=['sentimentNumeric'])

                    minDate = dataFrame['timestamp'].min()
                    maxDate = dataFrame['timestamp'].max()

                    minDateDt = datetime.datetime(minDate.year, minDate.month, minDate.day)
                    maxDateDt = datetime.datetime(maxDate.year, maxDate.month, maxDate.day)

                    preSelectedDates = (minDateDt, maxDateDt)

                    selectedRange = st.slider(
                        "Select Date Range",
                        min_value=minDateDt,
                        max_value=maxDateDt,
                        value=preSelectedDates
                    )

                    filteredData = dataFrame[(dataFrame['timestamp'] >= selectedRange[0]) & (dataFrame['timestamp'] <= selectedRange[1])]

                    filteredData['weekStart'] = filteredData['timestamp'].dt.to_period('W').dt.start_time
                    
                    weeklySentiment = filteredData.groupby('weekStart')['sentimentNumeric'].mean().reset_index()

                    if weeklySentiment.empty:
                        st.warning("Not enough valid data to plot sentiment trends.")
                    else:
                        fig = px.line(
                            weeklySentiment,
                            x='weekStart',
                            y='sentimentNumeric',
                            title="Sentiment Trends Over Time",
                            labels={"sentimentNumeric": "Average Sentiment", "weekStart": "Week"}
                        )
                        fig.update_layout(
                            xaxis_title="Week",
                            yaxis_title="Average Sentiment",
                            hovermode="x unified",
                            xaxis=dict(
                                tickformat="%b %d, %Y",  # Month day, Year format
                                tickangle=-45  # Angled labels for better readability
                            )
                        )
                        st.plotly_chart(fig)

        elif selectedVisualization == "Correlation Heatmap":
            st.subheader("Correlation Between Sentiment & Election Results")


        elif selectedVisualization == "Word Cloud":
            st.subheader("Keyword Association Cloud")
            
            searchKeyword = st.text_input("Enter a keyword to find associated words:", "")
            
            if 'timestamp' in dataFrame.columns:
                try:
                    dataFrame['timestamp'] = pd.to_datetime(dataFrame['timestamp'], errors='coerce')
                    minDate = dataFrame['timestamp'].min()
                    maxDate = dataFrame['timestamp'].max()
                    
                    minDateDt = datetime.datetime(minDate.year, minDate.month, minDate.day)
                    maxDateDt = datetime.datetime(maxDate.year, maxDate.month, maxDate.day)
                    
                    preSelectedDates = (minDateDt, maxDateDt)
                    
                    selectedRange = st.slider(
                        "Select Date Range for Word Cloud",
                        min_value=minDateDt,
                        max_value=maxDateDt,
                        value=preSelectedDates
                    )
                    filteredData = dataFrame[(dataFrame['timestamp'] >= selectedRange[0]) & (dataFrame['timestamp'] <= selectedRange[1])]
                except Exception as e:
                    st.error(f"Error processing 'timestamp' column: {e}")
                    filteredData = dataFrame
            else:
                filteredData = dataFrame
            
            if not filteredData.empty:
                try:
                    nltk.data.find('corpora/stopwords')
                except LookupError:
                    nltk.download('stopwords', quiet=True)
                
                stopWords = set(stopwords.words('english'))
                
                customStopwords = {"na", "ng", "https", "www", "lang", "mga", "ka", "ang", "may", "yung", "yan", "nga", "com", "kung", "pag"}
                stopWords.update(customStopwords)
                
                if searchKeyword.strip():
                    searchKeyword = searchKeyword.lower().strip()
                    mask = filteredData['text'].astype(str).str.lower().str.contains(searchKeyword)
                    keywordData = filteredData[mask]
                    stopWords.add(searchKeyword)
                    titleText = f"Words Associated with '{searchKeyword}'"
                else:
                    keywordData = filteredData
                    titleText = "Most Common Words"
                
                if not keywordData.empty:
                    allWords = []
                    
                    for text in keywordData['text'].astype(str):
                        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
                        filteredWords = [word for word in words if word not in stopWords]
                        allWords.extend(filteredWords)
                    
                    wordCounts = Counter(allWords)
                    
                    if wordCounts:
                        wordCloud = WordCloud(
                            width=800, 
                            height=400, 
                            background_color="white",
                            colormap="viridis"
                        ).generate_from_frequencies(wordCounts)
                        
                        fig, ax = plt.subplots(figsize=(10, 5))
                        ax.imshow(wordCloud, interpolation="bilinear")
                        ax.axis("off")
                        ax.set_title(titleText)
                        st.pyplot(fig)
                        
                        st.subheader("Top Associated Words")
                        topWords = sorted(wordCounts.items(), key=lambda x: x[1], reverse=True)[:25]
                        wordDf = pd.DataFrame(topWords, columns=["Word", "Frequency"])
                        st.dataframe(wordDf)
                        
                    else:
                        st.warning("No meaningful words found in the filtered data.")
                else:
                    st.warning(f"No posts containing the keyword '{searchKeyword}' found in the selected date range.")
            else:
                st.warning("No data available for the selected filters.")
        else:
            st.warning("No uploaded data. Please upload a file first.")