import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import datetime
from wordcloud import WordCloud
from pandas.api.types import is_numeric_dtype
from analysisModule import analyzeText, processCsv, calculateMetrics, runScraper, loadModel, create_word_network
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import re
import nltk
from nltk.corpus import stopwords
import networkx as nx
import matplotlib.patches as patches

# # **References**
# https://github.com/dataprofessor/dashboard-kit/        

# Load our pre-trained ML models
nbModel = loadModel("naiveBayesModel.pkl")
svmModel = loadModel("svmModel.pkl")
lrModel = loadModel("logisticRegressionModel.pkl")

# Set up the page layout
st.set_page_config(page_title="Sentiment Analysis Dashboard", layout="wide")

# Main navigation options
menuOptions = ["Upload/Scrape CSV & Analyze", "Analyze Text", "Visualizations"]
with st.sidebar:
    st.title("Sentiment Analysis Dashboard")
    selectedOption = st.selectbox("⚙️ Select Option", menuOptions)

# Initialize session state variables to persist data between reruns
if "uploadedDataFrames" not in st.session_state:
    st.session_state.uploadedDataFrames = {}
if "uploadedFileNames" not in st.session_state:
    st.session_state.uploadedFileNames = []

# FILE UPLOAD & DATA SCRAPING SECTION
if selectedOption == "Upload/Scrape CSV & Analyze":
    st.subheader("📂 Upload CSV Files or Scrape Data for Sentiment Analysis")

    # Handle file uploads - multiple files supported
    uploadedFiles = st.file_uploader("Choose CSV files", type=["csv"], accept_multiple_files=True)
    if uploadedFiles:
        for uploadedFile in uploadedFiles:
            # Only process new files to avoid redundant work
            if uploadedFile.name not in st.session_state.uploadedFileNames:
                st.session_state.uploadedFileNames.append(uploadedFile.name)
                filePath = os.path.join("uploads", uploadedFile.name)
                with open(filePath, "wb") as f:
                    f.write(uploadedFile.getbuffer())

                # Process the CSV and add sentiment analysis
                processedFileName = "analyzed_" + uploadedFile.name
                processCsv(filePath, processedFileName)
                st.success(f"File {uploadedFile.name} processed successfully!")
                processedData = pd.read_csv(processedFileName)
                st.session_state.uploadedDataFrames[uploadedFile.name] = processedData

    # Twitter data scraping section
    st.subheader("🔍 Scrape Twitter Data")
    query = st.text_input("Enter keywords for search (For date use: until:2023-02-09 since:2021-12-22 -filter:links)", "")
    limit = st.slider("Select number of data to be scrape", min_value=10, max_value=110, value=20)

    if st.button("Scrape Tweets"):
        if query:
            with st.spinner("Fetching tweets..."):
                # Run the scraper and process the data
                runScraper(query, limit) 
                st.success(f"Scraping complete! {limit}")
                processedFile = "analyzed_ScrapeData.csv"
                processCsv("ScrapeData.csv", processedFile)
                df = pd.read_csv(processedFile)
                st.session_state.uploadedDataFrames["Scraped Twitter Data"] = df
                st.session_state.uploadedFileNames.append("Scraped Twitter Data")
                st.write("### Scraped Twitter Data")
        else:
            st.warning("Please enter a search query before scraping.")

    # Display all uploaded/processed dataframes
    if st.session_state.uploadedDataFrames:
        st.write("### Analyzed Files:")
        for fileName, dataFrame in st.session_state.uploadedDataFrames.items():
            st.write(f"#### {fileName}")
            st.dataframe(dataFrame)

# TEXT ANALYSIS SECTION
elif selectedOption == "Analyze Text":
    st.subheader("Enter Text for Sentiment Analysis")
    userInputText = st.text_area("Enter text here...", "")

    if st.button("Analyze Sentiment"):
        if userInputText:
            # Run sentiment analysis on user input using multiple models
            analysisResults = analyzeText(userInputText)
            st.success("Analysis Complete!")
            st.write(f"**VADER Sentiment:** {analysisResults['vaderSentiment']}")
            st.write(f"**Naive Bayes Sentiment:** {analysisResults['naiveBayesSentiment']}")
            st.write(f"**SVM Sentiment:** {analysisResults['svmSentiment']}")
            st.write(f"**Logistic Regression Sentiment:** {analysisResults['logisticRegressionSentiment']}")

            # Store the single text analysis results for visualization
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

# VISUALIZATIONS SECTION 
elif selectedOption == "Visualizations":
    st.subheader("Visualizations")

    if st.session_state.uploadedDataFrames:
        # Selection
        selectedFile = st.selectbox("Select Dataset", st.session_state.uploadedFileNames)
        dataFrame = st.session_state.uploadedDataFrames[selectedFile].copy()
        visualizationOptions = ["Model Performance", "Sentiment Trends", "Word Cloud", "Word Network"]
        selectedVisualization = st.selectbox("Select Visualization", visualizationOptions)

        # MODEL PERFORMANCE VISUALIZATION
        if selectedVisualization == "Model Performance":
            st.subheader("Model Performance Comparison")
            modelPredictions = {
                "Naive Bayes": "naiveBayesSentiment",
                "SVM": "svmSentiment",
                "Logistic Regression": "logisticRegressionSentiment",
                "VADER": "vaderSentiment"
            }

            if "trueSentiment" in dataFrame.columns and dataFrame["trueSentiment"].notna().any():
                trueSentiment = dataFrame["trueSentiment"].tolist()
                st.info("Using manually annotated `trueSentiment` for evaluation.")
            else:
                trueSentiment = None
                st.warning("No `trueSentiment` found. Evaluating model agreement instead.")

            modelResults = {}
            if trueSentiment is not None:
                for modelName, column in modelPredictions.items():
                    if column in dataFrame.columns and dataFrame[column].notna().any():
                        predictedSentiment = dataFrame[column].tolist()
                        modelResults[modelName] = calculateMetrics(trueSentiment, predictedSentiment)
            else:
                consensusPredictions = []
                for i in range(len(dataFrame)):
                    predictions = [dataFrame[column].iloc[i] for column in modelPredictions.values() if column in dataFrame.columns]
                    if predictions:
                        consensusPredictions.append(max(set(predictions), key=predictions.count))
                for modelName, column in modelPredictions.items():
                    if column in dataFrame.columns and dataFrame[column].notna().any():
                        predictedSentiment = dataFrame[column].tolist()
                        modelResults[modelName] = calculateMetrics(consensusPredictions, predictedSentiment)

            if not modelResults:
                st.warning("Not enough valid data for model comparison.")
            else:
                def extractMetrics(report):
                    classes = ['positive', 'negative', 'neutral']
                    metricsTypes = ['precision', 'recall', 'f1-score']
                    metricsList = [report.get('accuracy', 0)]
                    for cls in classes:
                        for metric in metricsTypes:
                            metricsList.append(report.get(cls, {}).get(metric, 0))
                    return metricsList

                models = list(modelResults.keys())
                metricsLabels = ['Accuracy'] + [f"{cls}_{metric}" for cls in ['positive', 'negative', 'neutral'] for metric in ['precision', 'recall', 'f1-score']]
                modelMetrics = np.array([extractMetrics(modelResults[model]) for model in models])

                best_indices = np.argmax(modelMetrics, axis=0)

                fig, ax = plt.subplots(figsize=(15, 7))
                x = np.arange(len(metricsLabels))
                width = 0.15

                for i, model in enumerate(models):
                    bars = ax.bar(x + (i - 1.5) * width, modelMetrics[i], width, label=model)
                    for j, bar in enumerate(bars):
                        height = bar.get_height()
                        ax.annotate(f'{height:.2f}',
                                    xy=(bar.get_x() + bar.get_width() / 2, height),
                                    xytext=(0, 3),
                                    textcoords="offset points",
                                    ha='center', va='bottom',
                                    fontsize=8)

                        if best_indices[j] == i:
                            rect = patches.Rectangle(
                                (bar.get_x(), 0), 
                                bar.get_width(),
                                height,
                                linewidth=2,
                                edgecolor='black',
                                facecolor='none'
                            )
                            ax.add_patch(rect)

                ax.set_xlabel("Metrics")
                ax.set_ylabel("Scores")
                ax.set_title("Model Performance Comparison")
                ax.set_xticks(x)
                ax.set_xticklabels(metricsLabels, rotation=45, ha='right')
                ax.legend()
                ax.grid(True, axis='y', linestyle='--', alpha=0.7)
                plt.tight_layout()
                st.pyplot(fig)


        # SENTIMENT TRENDS VISUALIZATION
        elif selectedVisualization == "Sentiment Trends":
            st.subheader("Sentiment Trends Over Time")
            if 'timestamp' not in dataFrame.columns:
                st.warning("The dataset does not contain a 'timestamp' column.")
            else:
                try:
                    # Convert to datetime format
                    dataFrame['timestamp'] = pd.to_datetime(dataFrame['timestamp'], errors='coerce')
                    dataFrame = dataFrame.dropna(subset=['timestamp'])

                    if not dataFrame.empty:
                        # Map sentiment categories to numeric values for plotting
                        sentimentMapping = {"negative": -1, "neutral": 0, "positive": 1}
                        sentimentColumns = ['vaderSentiment', 'naiveBayesSentiment', 'svmSentiment', 'logisticRegressionSentiment']

                        for col in sentimentColumns:
                            if col in dataFrame.columns:
                                dataFrame[col] = dataFrame[col].map(sentimentMapping)

                        # Weight sentiment by likes
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

                        # Date range selector
                        selectedRange = st.slider(
                            "Select Date Range",
                            min_value=minDateDt,
                            max_value=maxDateDt,
                            value=preSelectedDates
                        )

                        # Filter data by selected date range and group by week
                        filteredData = dataFrame[(dataFrame['timestamp'] >= selectedRange[0]) & (dataFrame['timestamp'] <= selectedRange[1])]
                        filteredData['weekStart'] = filteredData['timestamp'].dt.to_period('W').dt.start_time
                        weeklySentiment = filteredData.groupby('weekStart')['sentimentNumeric'].mean().reset_index()

                        # Create line chart with Plotly
                        if not weeklySentiment.empty:
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
                                    tickformat="%b %d, %Y",
                                    tickangle=-45
                                )
                            )
                            st.plotly_chart(fig)
                        else:
                            st.warning("Not enough valid data to plot sentiment trends.")
                except Exception as e:
                    st.error(f"Error processing time data: {e}")

        # WORD CLOUD & TEXT ANALYSIS VISUALIZATION
        elif selectedVisualization == "Word Cloud":
            st.subheader("Interactive Keyword Analysis")
            col1, col2 = st.columns([2, 1])
            
            with col1:
                searchKeyword = st.text_input("Enter a keyword to analyze:", "")
                
                # Date range filter
                if 'timestamp' in dataFrame.columns:
                    try:
                        dataFrame['timestamp'] = pd.to_datetime(dataFrame['timestamp'], errors='coerce')
                        minDate = dataFrame['timestamp'].min()
                        maxDate = dataFrame['timestamp'].max()
                        selectedRange = st.slider(
                            "Select Date Range",
                            min_value=datetime.datetime(minDate.year, minDate.month, minDate.day),
                            max_value=datetime.datetime(maxDate.year, maxDate.month, maxDate.day),
                            value=(datetime.datetime(minDate.year, minDate.month, minDate.day), 
                                  datetime.datetime(maxDate.year, maxDate.month, maxDate.day))
                        )
                        filteredData = dataFrame[(dataFrame['timestamp'] >= selectedRange[0]) & 
                                              (dataFrame['timestamp'] <= selectedRange[1])]
                    except Exception as e:
                        st.error(f"Error processing time data: {e}")
                        filteredData = dataFrame
                else:
                    filteredData = dataFrame
                
                # Sentiment filter
                sentimentOptions = ['All', 'Positive', 'Negative', 'Neutral']
                selectedSentiment = st.selectbox("Filter by sentiment:", sentimentOptions)
                
                if selectedSentiment != 'All':
                    sentimentCol = st.selectbox("Select sentiment model:", 
                                             ['vaderSentiment', 'naiveBayesSentiment', 
                                              'svmSentiment', 'logisticRegressionSentiment'])
                    if sentimentCol in filteredData.columns:
                        filteredData = filteredData[filteredData[sentimentCol] == selectedSentiment.lower()]
            
            with col2:
                colormap = st.selectbox("Color theme:", 
                                      ['viridis', 'plasma', 'inferno', 'magma', 'cividis'])
                maxWords = st.slider("Maximum words:", 50, 300, 150)
            
            # Make sure we have stopwords
            if not filteredData.empty:
                try:
                    nltk.data.find('corpora/stopwords')
                except LookupError:
                    nltk.download('stopwords', quiet=True)
                
                stopWords = set(stopwords.words('english'))
                customStopwords = {"na", "ng", "https", "www", "lang", "mga", "ka", "ang", "may", 
                                "yung", "yan", "nga", "com", "kung", "pag"}
                stopWords.update(customStopwords)
                
                # Filter by keyword if provided
                if searchKeyword.strip():
                    searchKeyword = searchKeyword.lower().strip()
                    mask = filteredData['text'].astype(str).str.lower().str.contains(searchKeyword)
                    keywordData = filteredData[mask]
                    stopWords.add(searchKeyword)  # Not include search term in Cloud
                    titleText = f"Words Associated with '{searchKeyword}'"
                else:
                    keywordData = filteredData
                    titleText = "Most Common Words"
                
                if not keywordData.empty:
                    # Extract and clean words from texts
                    allWords = []
                    for text in keywordData['text'].astype(str):
                        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())  
                        filteredWords = [word for word in words if word not in stopWords]
                        allWords.extend(filteredWords)
                    
                    wordCounts = Counter(allWords)
                    
                     # Generate and display word cloud
                    if wordCounts:
                        wordCloud = WordCloud(
                            width=800, 
                            height=400, 
                            background_color="white",
                            colormap=colormap,
                            max_words=maxWords
                        ).generate_from_frequencies(wordCounts)
                        
                        fig, ax = plt.subplots(figsize=(10, 5))
                        ax.imshow(wordCloud, interpolation="bilinear")
                        ax.axis("off")
                        ax.set_title(titleText)
                        st.pyplot(fig)
                        
                        # Word selection for analysis
                        selectedWord = st.selectbox(
                            "Select a word to analyze:", 
                            [""] + [word for word, count in wordCounts.most_common(50)]
                        )
                        
                        if selectedWord:
                            st.subheader(f"Analysis of '{selectedWord}'")
                            tab1, tab2, tab3 = st.tabs(["Sentiment Analysis", "Common Co-occurrences", "Example Contexts"])
                            
                            with tab1:
                                # Show sentiment breakdown for texts containing the word
                                wordTexts = keywordData[keywordData['text'].str.contains(selectedWord, case=False)]
                                if not wordTexts.empty:
                                    sentimentCols = ['vaderSentiment', 'naiveBayesSentiment', 
                                                   'svmSentiment', 'logisticRegressionSentiment']
                                    availableCols = [col for col in sentimentCols if col in wordTexts.columns]
                                    
                                    if availableCols:
                                        # Add sentiment from all models
                                        sentimentTotals = {'positive': 0, 'negative': 0, 'neutral': 0}
                                        totalPredictions = 0
                                        
                                        for col in availableCols:
                                            sentimentCounts = wordTexts[col].value_counts().to_dict()
                                            for sentiment, count in sentimentCounts.items():
                                                if sentiment.lower() in sentimentTotals:
                                                    sentimentTotals[sentiment.lower()] += count
                                            totalPredictions += len(wordTexts[col].dropna())
                                        
                                        if totalPredictions > 0:
                                            # Calculate percentages for each sentiment
                                            overallSentiment = {
                                                sentiment: (count / totalPredictions) * 100 
                                                for sentiment, count in sentimentTotals.items()
                                            }
                                            
                                            st.write(f"**Overall sentiment (average of {len(availableCols)} models):**")
                                            sentimentDf = pd.DataFrame({
                                                'Sentiment': ['Positive', 'Negative', 'Neutral'],
                                                'Percentage': [
                                                    overallSentiment.get('positive', 0),
                                                    overallSentiment.get('negative', 0),
                                                    overallSentiment.get('neutral', 0)
                                                ]
                                            })
                                            
                                            # Create bar chart 
                                            fig = px.bar(
                                                sentimentDf,
                                                x='Sentiment',
                                                y='Percentage',
                                                color='Sentiment',
                                                color_discrete_map={
                                                    'Positive': 'green',
                                                    'Negative': 'red',
                                                    'Neutral': 'gray'
                                                },
                                                text='Percentage',
                                                title=f"Overall Sentiment for '{selectedWord}'"
                                            )
                                            fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                                            fig.update_layout(yaxis_range=[0, 100])
                                            st.plotly_chart(fig)
                                            
                                        else:
                                            st.warning("No sentiment predictions available.")
                                    else:
                                        st.warning("No sentiment data available.")
                                else:
                                    st.warning(f"No occurrences of '{selectedWord}' found.")
                            
                            with tab2:
                                # Find words that commonly appear with the selected word
                                cooccurringWords = []
                                for text in keywordData['text'].astype(str):
                                    if selectedWord.lower() in text.lower():
                                        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
                                        filteredWords = [word for word in words 
                                                        if word != selectedWord.lower() 
                                                        and word not in stopWords]
                                        cooccurringWords.extend(filteredWords)
                                
                                if cooccurringWords:
                                    cooccurringCounts = Counter(cooccurringWords)
                                    topCooccurring = cooccurringCounts.most_common(15)
                                    cooccurDf = pd.DataFrame(topCooccurring, columns=['Word', 'Count'])
                                    fig = px.bar(cooccurDf, x='Word', y='Count',
                                               title=f"Words Most Often Used With '{selectedWord}'")
                                    st.plotly_chart(fig)
                                    st.dataframe(cooccurDf)
                                else:
                                    st.warning(f"No significant co-occurring words found.")
                            
                            with tab3:
                                # Show examples of the word in context
                                exampleTexts = keywordData[keywordData['text'].str.contains(selectedWord, case=False)]['text'].head(5)
                                if not exampleTexts.empty:
                                    st.write("**Example sentences:**")
                                    for i, text in enumerate(exampleTexts, 1):
                                        highlightedText = re.sub(
                                            f"({selectedWord})", 
                                            r"**\1**", 
                                            text, 
                                            flags=re.IGNORECASE
                                        )
                                        st.write(f"{i}. {highlightedText}")
                                else:
                                    st.warning(f"No example texts found.")
                        
                        # Show frequency table of top words
                        st.subheader("Top Words")
                        topWords = sorted(wordCounts.items(), key=lambda x: x[1], reverse=True)[:25]
                        wordDf = pd.DataFrame(topWords, columns=["Word", "Frequency"])
                        st.dataframe(wordDf)
                    else:
                        st.warning("No meaningful words found.")
                else:
                    st.warning(f"No posts containing '{searchKeyword}' found.")
            else:
                st.warning("No data available for selected filters.")
        
        # WORD NETWORK VISUALIZATION
        elif selectedVisualization == "Word Network":
            st.subheader("Word Network Analysis")
            
            # Date range filter
            if 'timestamp' in dataFrame.columns:
                try:
                    dataFrame['timestamp'] = pd.to_datetime(dataFrame['timestamp'], errors='coerce')
                    minDate = dataFrame['timestamp'].min()
                    maxDate = dataFrame['timestamp'].max()
                    selectedRange = st.slider(
                        "Select Date Range",
                        min_value=datetime.datetime(minDate.year, minDate.month, minDate.day),
                        max_value=datetime.datetime(maxDate.year, maxDate.month, maxDate.day),
                        value=(datetime.datetime(minDate.year, minDate.month, minDate.day), 
                            datetime.datetime(maxDate.year, maxDate.month, maxDate.day))
                    )
                    filteredData = dataFrame[(dataFrame['timestamp'] >= selectedRange[0]) & 
                                        (dataFrame['timestamp'] <= selectedRange[1])]
                except Exception as e:
                    st.error(f"Error processing time data: {e}")
                    filteredData = dataFrame
            else:
                filteredData = dataFrame
            
            # Sentiment filter
            col1, col2 = st.columns(2)
            
            with col1:
                sentimentOptions = ['All', 'Positive', 'Negative', 'Neutral']
                selectedSentiment = st.selectbox(
                    "Filter by sentiment:", 
                    sentimentOptions,
                    key="network_sentiment"
                )
                
                if selectedSentiment != 'All':
                    sentimentCol = st.selectbox(
                        "Select sentiment model:", 
                        ['vaderSentiment', 'naiveBayesSentiment', 'svmSentiment', 'logisticRegressionSentiment'],
                        key="network_sentiment_model"
                    )
                    if sentimentCol in filteredData.columns:
                        filteredData = filteredData[filteredData[sentimentCol] == selectedSentiment.lower()]
            
            with col2:
                # Network diagram options
                numWords = st.slider("Number of top words to include:", 10, 100, 50)
                minEdgeWeight = st.slider("Minimum connection strength:", 1, 10, 2, 
                                    help="Higher values show only stronger connections")
                expansionLevel = st.slider("Network expansion level:", 1, 3, 1,
                                        help="Level 1: direct connections, Level 2: connections of connections, etc.")
            
            # Keyword filter option
            searchTerm = st.text_input("Focus on a specific term (optional):", "", key="network_search")
            
            if searchTerm.strip():
                searchTerm = searchTerm.lower().strip()
                filteredData = filteredData[filteredData['text'].astype(str).str.lower().str.contains(searchTerm)]
                focusMode = True
                st.success(f"Showing network focused on term: '{searchTerm}'")
            else:
                focusMode = False
            
            # If we have filtered data, create the network
            if not filteredData.empty and 'text' in filteredData.columns:
                # Create network graph
                network_graph, word_counts = create_word_network(
                    filteredData['text'].astype(str),
                    num_words=numWords,
                    min_edge_weight=minEdgeWeight
                )
                
                if len(network_graph.nodes) > 0:
                    # Apply multi-level expansion if requested
                    if expansionLevel > 1 and searchTerm:
                        try:
                            central_node = searchTerm.lower()
                            if central_node in network_graph.nodes:
                                # Get nodes within N steps of the central node
                                ego_nodes = nx.ego_graph(network_graph, central_node, radius=expansionLevel).nodes()
                                # Filter the graph to only include these nodes and their connections
                                network_graph = network_graph.subgraph(ego_nodes)
                        except:
                            st.warning("Could not expand network. Showing basic network instead.")
                    
                    # Apply layout algorithm
                    pos = nx.spring_layout(network_graph, k=0.4, iterations=50, seed=42)
                    
                    # Prepare network data for Plotly
                    edge_traces = []
                    
                    # Create edges with thinner lines
                    for edge in network_graph.edges(data=True):
                        x0, y0 = pos[edge[0]]
                        x1, y1 = pos[edge[1]]
                        weight = edge[2]['weight']
                        
                        # Reduced line width from original (0.5 + weight/2) to (0.3 + weight/4)
                        edge_trace = go.Scatter(
                            x=[x0, x1], y=[y0, y1],
                            line=dict(width=0.3 + weight/4, color='rgba(150,150,150,0.5)'),  # Thinner and more transparent
                            hoverinfo='text',
                            text=f"{edge[0]} — {edge[1]}<br>Weight: {weight}",
                            mode='lines'
                        )
                        edge_traces.append(edge_trace)
                    
                    # Create node trace with sentiment coloring
                    node_x = []
                    node_y = []
                    node_colors = []
                    node_text = []
                    node_sizes = []
                    
                    # Get sentiment scores if available
                    sentiment_scores = {}
                    if 'vaderSentiment' in filteredData.columns:
                        for _, row in filteredData.iterrows():
                            words = str(row['text']).lower().split()
                            for word in words:
                                if word in network_graph.nodes:
                                    if word not in sentiment_scores:
                                        sentiment_scores[word] = []
                                    sentiment_scores[word].append(row['vaderSentiment'])
                    
                    for node in network_graph.nodes():
                        x, y = pos[node]
                        node_x.append(x)
                        node_y.append(y)
                        
                        # Calculate average sentiment if available
                        if node in sentiment_scores:
                            avg_sentiment = np.mean([1 if s == 'positive' else (-1 if s == 'negative' else 0) 
                                                for s in sentiment_scores[node]])
                            # Color mapping: red (negative) to green (positive)
                            if avg_sentiment > 0:
                                color = f'rgba(50, 200, 50, 0.8)'  # Green for positive
                            elif avg_sentiment < 0:
                                color = f'rgba(200, 50, 50, 0.8)'  # Red for negative
                            else:
                                color = f'rgba(150, 150, 150, 0.8)'  # Gray for neutral
                        else:
                            color = 'rgba(0, 149, 255, 0.8)'  # Default blue
                        
                        node_colors.append(color)
                        
                        # Scale node sizes by word frequency
                        size = word_counts.get(node, 1)
                        node_sizes.append(5 + (np.log1p(size) * 5))
                        
                        # Create hover text with sentiment info if available
                        connections = list(network_graph.neighbors(node))
                        hover_text = f"<b>{node}</b><br>Frequency: {size}<br>Connected to: {', '.join(connections[:10])}{' ...' if len(connections) > 5 else ''}"
                        
                        if node in sentiment_scores:
                            pos_count = sum(1 for s in sentiment_scores[node] if s == 'positive')
                            neg_count = sum(1 for s in sentiment_scores[node] if s == 'negative')
                            hover_text += f"<br>Sentiment: +{pos_count}/-{neg_count}"
                        
                        node_text.append(hover_text)
                    
                    node_trace = go.Scatter(
                        x=node_x, y=node_y,
                        mode='markers',
                        hoverinfo='text',
                        text=node_text,
                        marker=dict(
                            color=node_colors,
                            size=node_sizes,
                            line=dict(width=0.5, color='rgba(0, 0, 0, 0.5)') 
                        )
                    )
                    
                    # Create layout
                    annotations = []
                    for node, (x, y) in pos.items():
                        annotations.append(dict(
                            x=x, y=y,
                            text=node,
                            showarrow=False,
                            font=dict(size=10),
                            xanchor='center',
                            yanchor='bottom'
                        ))
                    
                    # Create the figure with all traces
                    fig = go.Figure(data=edge_traces + [node_trace],
                    layout=go.Layout(
                        title=f"Word Co-occurrence Network{' for '+selectedSentiment+' Sentiment' if selectedSentiment != 'All' else ''}",
                        title_font=dict(size=16),
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20,l=5,r=5,t=40),
                        annotations=annotations,
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        height=750  
                    ))

                    # Display with potential height override
                    st.plotly_chart(fig, use_container_width=True, height=800)
                    
                    # Network statistics
                    st.subheader("Network Statistics")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Number of Words (Nodes)", len(network_graph.nodes))
                    with col2:
                        st.metric("Number of Connections (Edges)", len(network_graph.edges))
                    with col3:
                        if len(network_graph.nodes) > 0:
                            density = nx.density(network_graph)
                            st.metric("Network Density", f"{density:.4f}")
                    
                    # Community detection for thematic clusters
                    if len(network_graph.nodes) >= 3:
                        try:
                            # Find communities
                            communities = nx.community.greedy_modularity_communities(network_graph)
                            
                            # Display community information
                            st.subheader("Thematic Clusters")
                            
                            for i, community in enumerate(communities[:5]):  # Show top 5 communities
                                community_words = list(community)
                                st.write(f"**Cluster {i+1}:** {', '.join(community_words[:20])}{' ...' if len(community_words) > 10 else ''}")
                        except:
                            st.info("Could not detect communities in this network.")
                    
                    st.subheader("Most Influential Words")
                    
                    try:
                        # Calculate centrality measures
                        centrality = nx.degree_centrality(network_graph)
                        betweenness = nx.betweenness_centrality(network_graph)
                        
                        # Combine centrality measures
                        influence_scores = {}
                        for node in network_graph.nodes():
                            influence_scores[node] = {
                                'Word': node,
                                'Degree': centrality.get(node, 0),
                                'Betweenness': betweenness.get(node, 0),
                                'Frequency': word_counts.get(node, 0)
                            }
                        
                        # Create DataFrame for display
                        influence_df = pd.DataFrame.from_dict(influence_scores, orient='index')
                        influence_df = influence_df.sort_values(by='Degree', ascending=False).head(10)
                        
                        # Display as table
                        st.dataframe(influence_df)
                    except:
                        st.info("Could not calculate centrality measures for this network.")
                else:
                    st.warning("Not enough connected words found for network visualization. Try reducing the minimum connection strength or including more words.")
            else:
                st.warning("No data available for the selected filters.")
                
            with st.expander("How to Interpret the Word Network"):
                st.markdown("""
                ## Understanding the Word Network Diagram
                
                This network visualization shows connections between words that frequently appear together in the same text.
                
                ### Key Components:
                
                - **Nodes:** Words from the dataset. The size of each node indicates how frequently the word appears.
                - **Edges:** Connections between words. Thicker edges indicate stronger connections (words appearing together more often).
                - **Clusters:** Groups of words that often appear together, potentially representing topics or themes.
                
                ### Metrics:
                
                - **Degree Centrality:** Measures how many connections a word has to other words. Words with high degree centrality are central to the conversation.
                - **Betweenness Centrality:** Measures how often a word acts as a bridge between other words. Words with high betweenness often connect different topics.
                """)
    else:
        st.warning("No uploaded data available.")