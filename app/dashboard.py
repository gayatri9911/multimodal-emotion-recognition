import pandas as pd
import streamlit as st
import plotly.express as px


def show_dashboard():

    st.markdown("---")

    st.header("Emotion Analytics Dashboard")

    try:

        df = pd.read_csv(
            "../emotion_history.csv",
            on_bad_lines='skip'
        )

        df = df.dropna()

        if len(df) == 0:

            st.warning(
                "No emotion data available yet."
            )

            return

        emotion_counts = df[
            'emotion'
        ].value_counts()

        # Metrics
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Total Predictions",
                len(df)
            )

        with col2:
            st.metric(
                "Top Emotion",
                emotion_counts.idxmax()
            )

        with col3:
            st.metric(
                "Unique Emotions",
                len(emotion_counts)
            )

        st.markdown("---")

        # Bar Chart
        st.subheader("Emotion Frequency")

        bar_fig = px.bar(
            x=emotion_counts.index,
            y=emotion_counts.values,
            labels={
                'x': 'Emotion',
                'y': 'Count'
            },
            title='Emotion Frequency Distribution'
        )

        st.plotly_chart(
            bar_fig,
            use_container_width=True
        )

        # Pie Chart
        st.subheader("Emotion Distribution")

        pie_fig = px.pie(
            values=emotion_counts.values,
            names=emotion_counts.index,
            title='Emotion Distribution'
        )

        st.plotly_chart(
            pie_fig,
            use_container_width=True
        )

        # Recent History
        st.subheader("Recent Emotion History")

        st.dataframe(
            df.tail(10),
            use_container_width=True
        )

    except Exception as e:

        st.error(
            f"Dashboard Error: {e}"
        )