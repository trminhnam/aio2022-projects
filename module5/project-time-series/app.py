import os
import sys

# insert current directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))


import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from dataset import TelsaStockDataset
from model import ForecastModel
from utils import eval_fn, sliding_window, train_fn

st.set_page_config(
    layout="wide",
    page_title="Tesla Stock Analysis",
    page_icon="📈",
)


# @st.cache(allow_output_mutation=True)
def load_dataset():
    # Load the dataset
    df = pd.read_csv(
        os.path.join(os.path.dirname(__file__), "data", "tesla-stock-price.csv"),
        parse_dates=["date"],
    )
    # drop second row
    df.drop(df.index[0], inplace=True)
    return df


def introduction():
    st.title("📊 Introduction")
    st.write("👋 Welcome to the Tesla stock analysis app!")
    st.write(
        "🔍 This app allows you to explore the Tesla stock dataset, perform exploratory data analysis (EDA), and make predictions with a Long Short-Term Memory (LSTM) model."
    )
    st.write(
        "📈 The dataset contains daily stock prices for Tesla, and includes the following columns: date, close, volume, open, high, and low."
    )
    st.write("📑 The other pages in this app include:")
    st.write(
        "  - 📊 EDA Page: This page includes several visualizations to help you understand the dataset."
    )
    st.write(
        "  - 💰 Stock Prediction Page: This page uses an LSTM model to predict the closing price of the stock for a given date range."
    )
    st.write(
        "  - 🧑‍💻 About Page: This page provides information about the app and its creators."
    )
    st.write(
        "  - 📚 Reference Page: This page lists the references used in building this app."
    )


# Define a function for the EDA page
def eda():
    df = load_dataset()
    # Set the title of the app
    st.title("📊 Simple Tesla Stock Data EDA")

    st.subheader("Candlestick Chart")
    fig = go.Figure(
        data=[
            go.Candlestick(
                x=df["date"],
                open=df["open"],
                high=df["high"],
                low=df["low"],
                close=df["close"],
                increasing_line_color="green",
                decreasing_line_color="red",
            )
        ],
        layout=go.Layout(
            title="Tesla Stock Price",
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            xaxis_rangeslider_visible=True,
        ),
    )
    st.plotly_chart(fig)

    st.subheader("Line Chart")
    # Plot the closing price over time
    plot_column = st.selectbox(
        "Select a column", ["open", "close", "low", "high"], index=0
    )
    fig = px.line(
        df, x="date", y=plot_column, title=f"Tesla {plot_column} Price Over Time"
    )
    st.plotly_chart(fig)

    st.subheader("Distribution of Closing Price")
    # Plot the distribution of the closing price
    closing_price_hist = px.histogram(
        df, x="close", nbins=30, title="Distribution of Tesla Closing Price"
    )
    st.plotly_chart(closing_price_hist)

    st.subheader("Raw Data")
    st.write("You can see the raw data below.")
    # Display the dataset
    st.dataframe(df)


# Define a function for the Stock Prediction page
def stock_prediction():
    st.title("💰 Stock Prediction with LSTM Model")
    st.write(
        "This page uses an LSTM model to predict the closing price of the stock for a given date range."
    )

    # Add input items to tune hyperparameters
    st.sidebar.title("Hyperparameter Tuning")

    st.sidebar.subheader("Model Hyperparameters")
    hidden_size = st.sidebar.slider("Hidden Size", 32, 256, 128, 8)
    num_layers = st.sidebar.slider("Number of Layers", 1, 5, 2, 1)
    window_size = st.sidebar.slider("Window Size", 1, 30, 10, 1)

    st.sidebar.subheader("Training Hyperparameters")
    learning_rate = st.sidebar.number_input(
        "Learning Rate",
        min_value=1e-5,
        max_value=1e-2,
        value=1e-4,
        step=1e-5,
        format="%.5f",
    )
    batch_size = st.sidebar.slider("Batch Size", 16, 128, 64, 16)
    epochs = st.sidebar.slider("Epochs", 10, 100, 50, 10)
    train_rate = st.sidebar.slider("Train Set Ratio", 0.5, 0.9, 0.7, 0.1)
    test_rate = st.sidebar.slider("Test Set Ratio", 0.05, 0.4, 0.2, 0.05)
    val_rate = st.sidebar.slider("Validation Set Ratio", 0.05, 0.4, 0.1, 0.05)
    random_seed = st.sidebar.slider("Random Seed", 0, 100, 42, 1)

    # normalize train/validation/test rate to 1.0
    train_test_ratio = np.array([train_rate, test_rate, val_rate])
    train_test_ratio = train_test_ratio / train_test_ratio.sum()

    # Show the selected hyperparameters
    st.subheader("Selected Hyperparameters")
    st.write("**Model Hyperparameters:**")
    st.write(f"- Hidden Size: {hidden_size}")
    st.write(f"- Number of Layers: {num_layers}")
    st.write(f"- Window Size: {window_size}")
    st.write("**Training hyperparameters:**")
    st.write(f"- Learning Rate: {learning_rate}")
    st.write(f"- Batch Size: {batch_size}")
    st.write(f"- Epochs: {epochs}")
    st.write(
        f"- Train/Test/Validation Set Ratio: {train_test_ratio[0]:.2f}/{train_test_ratio[1]:.2f}/{train_test_ratio[2]:.2f}"
    )
    st.write(f"- Random Seed: {random_seed}")

    st.subheader("Training Model")
    # Train the model
    activate_train = st.button("Train Model")
    if activate_train:
        dataframe = load_dataset()
        if "date" in dataframe.columns:
            dataframe.drop(["date"], axis=1, inplace=True)
        target_idx = dataframe.columns.get_loc("close")

        for col in dataframe.columns:
            dataframe[col] = dataframe[col].replace(",", "", regex=True).astype(float)

        mean = dataframe.mean()
        std = dataframe.std()

        # normalize data
        dataframe = dataframe.astype(float)
        dataframe = (dataframe - mean) / std
        dataframe.head()

        data = sliding_window(dataframe, window_size + 1, 1)
        data = np.array(data)

        X = data[:, :-1, :]
        y = data[:, -1, target_idx : target_idx + 1]

        # train/val/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=train_test_ratio[1], random_state=random_seed
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train,
            y_train,
            test_size=train_test_ratio[2] / (train_test_ratio[0] + train_test_ratio[2]),
            random_state=random_seed,
        )

        train_dataset = TelsaStockDataset(X_train, y_train)
        val_dataset = TelsaStockDataset(X_val, y_val)
        test_dataset = TelsaStockDataset(X_test, y_test)

        train_dataloader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = ForecastModel(
            n_features=5,
            n_hidden=hidden_size,
            n_layers=num_layers,
            batch_first=True,
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()

        train_losses = []
        val_losses = []

        with st.spinner("Training model..."):
            for epoch in range(epochs):
                train_loss = train_fn(
                    model, train_dataloader, optimizer, criterion, device
                )
                val_loss = eval_fn(model, val_dataloader, criterion, device)

                train_losses.append(train_loss)
                val_losses.append(val_loss)

                st.write(
                    f"Epoch: {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
                )

            st.success("Model training complete!")

            # Plot the training and validation loss
            # st.subheader("Training and Validation Loss")
            # fig, ax = plt.subplots()
            # ax.plot(train_losses, label="Training Loss")
            # ax.plot(val_losses, label="Validation Loss")
            # ax.set_xlabel("Epoch")
            # ax.set_ylabel("Loss")
            # ax.legend()
            # st.pyplot(fig)

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=list(range(len(train_losses))),
                y=train_losses,
                mode="lines",
                name="Training Loss",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=list(range(len(val_losses))),
                y=val_losses,
                mode="lines",
                name="Validation Loss",
            )
        )
        fig.update_layout(
            title="Training and Validation Loss",
            xaxis_title="Epoch",
            yaxis_title="Loss",
        )
        st.plotly_chart(fig)

        # Make predictions
        st.subheader("Evaluation and visualization")
        model.eval()
        model = model.to(device)
        ground_truth = []
        predictions = []

        # plot the test true value and predicted value
        for batch_idx, (X_test, y_test) in enumerate(test_dataloader):
            X_test = X_test.to(device)
            y_test = y_test.to(device)

            with torch.no_grad():
                y_pred = model.predict(X_test)

            y_pred = y_pred.detach().cpu().numpy()
            y_test = y_test.detach().cpu().numpy()
            y_pred = y_pred * std[target_idx] + mean[target_idx]
            y_test = y_test * std[target_idx] + mean[target_idx]
            ground_truth.extend(y_test)
            predictions.extend(y_pred)

        ground_truth = np.array(ground_truth).reshape(-1).astype(np.int32)
        predictions = np.array(predictions).reshape(-1).astype(np.int32)
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=list(range(len(ground_truth))),
                y=ground_truth,
                mode="lines",
                name="Ground Truth",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=list(range(len(predictions))),
                y=predictions,
                mode="lines",
                name="Predictions",
            )
        )
        fig.update_layout(title="Stock Prices", xaxis_title="Time", yaxis_title="Price")
        st.plotly_chart(fig)

        # Save the model and download it
        st.subheader("Save and Download Model")
        model_path = "model.pt"
        torch.save(model.state_dict(), model_path)
        st.success("Model saved to disk.")
        st.download_button(
            label="Download Model",
            data=open(model_path, "rb"),
            file_name="model.pt",
            mime="application/octet-stream",
        )


# Define a function for the About page
def about():
    st.title("🧑‍💻 About")
    st.write(
        "This Streamlit app allows you to explore stock prices and make predictions using an LSTM model."
    )

    st.header("Author")
    st.write(
        "This app was developed by John Doe. You can contact the author at trminhnam20082002@gmail.com."
    )

    st.header("Data Sources")
    st.markdown(
        "The stock price data was sourced from [Yahoo Finance](https://finance.yahoo.com/quote/TSLA)."
    )

    st.header("Acknowledgments")
    st.write(
        "The author would like to thank Jane Smith for her invaluable feedback and support during the development of this app."
    )

    st.header("License")
    st.write(
        "This app is licensed under the MIT License. See LICENSE.txt for more information."
    )


def references():
    st.title("📚 References")
    st.header("References for Stock Prediction Model")

    st.subheader("1. 'Project for time-series data' by AI VIET NAM, et al.")
    st.write(
        "This organization provides a tutorial on how to build a stock price prediction model using LSTM in the AIO2022 course."
    )
    st.write("Link: https://www.facebook.com/aivietnam.edu.vn")

    st.subheader(
        "2. 'PyTorch LSTMs for time series forecasting of Indian Stocks' by Vinayak Nayak"
    )
    st.write(
        "This blog post describes how to build a stock price prediction model using LSTM, RNN and CNN-sliding window model."
    )
    st.write(
        "Link: https://medium.com/analytics-vidhya/pytorch-lstms-for-time-series-forecasting-of-indian-stocks-8a49157da8b9#b052"
    )

    st.header("References for Streamlit")

    st.subheader("1. Streamlit Documentation")
    st.write(
        "The official documentation for Streamlit provides detailed information about how to use the library and build Streamlit apps."
    )
    st.write("Link: https://docs.streamlit.io/")

    st.subheader("2. Streamlit Community")
    st.write(
        "The Streamlit community includes a forum and a GitHub repository with examples and resources for building Streamlit apps."
    )
    st.write(
        "Link: https://discuss.streamlit.io/ and https://github.com/streamlit/streamlit/"
    )


# Create the sidebar
st.sidebar.title("Menu")
pages = ["Introduction", "EDA", "Stock Prediction", "About", "References"]
selected_page = st.sidebar.radio("Go to", pages)

# Show the appropriate page based on the selection
if selected_page == "Introduction":
    introduction()
elif selected_page == "EDA":
    eda()
elif selected_page == "Stock Prediction":
    stock_prediction()
elif selected_page == "About":
    about()
elif selected_page == "References":
    references()
