import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from nltk import bigrams
from sklearn.svm import LinearSVC
import joblib
import plotly.express as px

# ================================
# SETUP STREAMLIT CONFIG
# ================================
st.set_page_config(
    page_title="Dashboard Analisis Sentimen Komentar Platform X", layout="wide"
)

# HILANGKAN SIDEBAR
st.markdown(
    """
<style>
    section[data-testid="stSidebar"] {display: none;}
    .block-container {padding-top: 1rem;}
</style>
""",
    unsafe_allow_html=True,
)


# ================================
# LOAD MODEL
# ================================
# Pastikan file model Anda ada di folder yang sama:
# svm_tfidf_model.pkl

model_svm = joblib.load(
    "svm_tfidf_model_hyperparameter.pkl"
)


# ================================
# HEADER DASHBOARD
# ================================
st.title("Dashboard Analisis Sentimen Platform X (Pernikahan Dini)")
st.markdown("Analisis komentar menggunakan **NLP + TF-IDF + SVM**")
st.markdown("----")


# ================================
# INPUT DATASET
# ================================
st.subheader("📁 Upload Dataset Komentar (CSV)")

uploaded = st.file_uploader("Upload file CSV yang sudah dipreprocess", type=["csv"])

if uploaded is not None:
    df = pd.read_csv(uploaded)

    if "sentiment" not in df.columns:
        st.error(
            "Kolom 'sentiment' tidak ada. Pastikan Anda sudah melakukan labeling dengan RoBERTa / model lain."
        )
    else:
        st.success("Dataset berhasil dimuat!")

        # ================================
        # KPI SUMMARY
        # ================================

        st.markdown("----------")
        st.subheader("📌 Ringkasan Sentimen")

        total = len(df)
        pos = sum(df["sentiment"] == "positive")
        neg = sum(df["sentiment"] == "negative")
        neu = sum(df["sentiment"] == "neutral")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Komentar", total)
        col2.metric("Positif", f"{pos} ({pos/total*100:.1f}%)")
        col3.metric("Netral", f"{neu} ({neu/total*100:.1f}%)")
        col4.metric("Negatif", f"{neg} ({neg/total*100:.1f}%)")

        # ================================
        # DISTRIBUSI SENTIMEN
        # ================================
        st.markdown("----------")

        st.subheader("📊 Distribusi Sentimen")

        col1, col2 = st.columns(2)

        sentiment_counts = df["sentiment"].value_counts().reset_index()
        sentiment_counts.columns = ["sentiment", "count"]

        # -------- BAR CHART INTERAKTIF --------
        with col1:
            fig_bar = px.bar(
                sentiment_counts,
                x="sentiment",
                y="count",
                color="sentiment",
                text="count",
                title="Distribusi Sentimen (Bar)",
                color_discrete_map={
                    "positive": "#4CAF50",
                    "negative": "#F44336",
                    "neutral": "#FFC107",
                },
            )
            fig_bar.update_layout(
                height=350,
                showlegend=False,
                xaxis_title="Sentimen",
                yaxis_title="Jumlah",
            )
            fig_bar.update_traces(textposition="outside")
            st.plotly_chart(fig_bar, use_container_width=True)

        # -------- PIE CHART INTERAKTIF (DONUT) --------
        with col2:
            fig_pie = px.pie(
                sentiment_counts,
                names="sentiment",
                values="count",
                hole=0.40,
                title="Proporsi Sentimen (Pie)",
                color="sentiment",
                color_discrete_map={
                    "positive": "#4CAF50",
                    "negative": "#F44336",
                    "neutral": "#FFC107",
                },
            )
            fig_pie.update_traces(textposition="inside", textinfo="percent+label")

            fig_pie.update_layout(height=400)
            st.plotly_chart(fig_pie, use_container_width=True)

        # st.subheader("📊 Distribusi Sentimen")
        # fig, ax = plt.subplots(figsize=(6, 4))
        # sns.countplot(data=df, x="sentiment", palette="Set2", ax=ax)
        # ax.set_title("Distribusi Sentimen Komentar")
        # st.pyplot(fig)

        # ================================
        # WORDCLOUD FUNCTION
        # ================================
        st.markdown("----------")

        def show_wordcloud(text, title):
            if len(text.strip()) == 0:
                st.warning(f"Tidak ada kata untuk WordCloud {title}")
                return

            wc = WordCloud(
                width=1200, height=700, background_color="white", max_words=150
            ).generate(text)

            fig = plt.figure(figsize=(10, 6))
            plt.imshow(wc, interpolation="bilinear")
            plt.axis("off")
            plt.title(title)
            st.pyplot(fig)

        # ================================
        # WORDCLOUD PER SENTIMEN
        # ================================
        st.subheader("☁ WordCloud Sentimen")

        col1, col2, col3 = st.columns(3)

        with col1:
            positif_text = " ".join(
                df[df["sentiment"] == "positive"]["stopword removal"].astype(str)
            )
            show_wordcloud(positif_text, "WordCloud Positif")

        with col2:
            negatif_text = " ".join(
                df[df["sentiment"] == "negative"]["stopword removal"].astype(str)
            )
            show_wordcloud(negatif_text, "WordCloud Negatif")

        with col3:
            netral_text = " ".join(
                df[df["sentiment"] == "neutral"]["stopword removal"].astype(str)
            )
            show_wordcloud(netral_text, "WordCloud Netral")

        # ================================
        # TOP WORDS PER SENTIMEN
        # ===============================
        st.markdown("----------")

        def safe_join(series):
            if series is None:
                return ""
            return " ".join(series.dropna().astype(str))

        def get_top_words_df(series, top_n=20):
            text = safe_join(series)
            if not text.strip():
                return pd.DataFrame(columns=["word", "count"])
            words = text.split()
            counter = Counter(words)
            most = counter.most_common(top_n)
            df_top = pd.DataFrame(most, columns=["word", "count"])
            return df_top

        top_n = 10  # ubah kalau mau 10/15/30
        df_pos = get_top_words_df(
            df[df["sentiment"] == "positive"]["stopword removal"], top_n
        )
        df_neg = get_top_words_df(
            df[df["sentiment"] == "negative"]["stopword removal"], top_n
        )
        df_neu = get_top_words_df(
            df[df["sentiment"] == "neutral"]["stopword removal"], top_n
        )

        if df_pos.empty and df_neg.empty and df_neu.empty:
            st.warning(
                "Tidak ada teks untuk membuat top words. Periksa kolom 'stopword removal' dan isinya."
            )
        else:
            st.markdown("## 🔤 Top Kata Teratas per Sentimen (Interaktif)")

            c1, c2, c3 = st.columns(3)

            with c1:
                st.write("### Positif")
                if not df_pos.empty:
                    fig_pos = px.bar(
                        df_pos.sort_values("count"),
                        x="count",
                        y="word",
                        orientation="h",
                        labels={"count": "Frekuensi", "word": "Kata"},
                        title=f"Top {len(df_pos)} Kata - Positif",
                        height=420,
                    )
                    fig_pos.update_layout(
                        margin=dict(l=10, r=10, t=40, b=10), showlegend=False
                    )
                    st.plotly_chart(fig_pos, use_container_width=True)
                else:
                    st.info("Tidak ada kata untuk sentimen positif.")

            with c2:
                st.write("### Negatif")
                if not df_neg.empty:
                    fig_neg = px.bar(
                        df_neg.sort_values("count"),
                        x="count",
                        y="word",
                        orientation="h",
                        labels={"count": "Frekuensi", "word": "Kata"},
                        title=f"Top {len(df_neg)} Kata - Negatif",
                        height=420,
                    )
                    fig_neg.update_layout(
                        margin=dict(l=10, r=10, t=40, b=10), showlegend=False
                    )
                    st.plotly_chart(fig_neg, use_container_width=True)
                else:
                    st.info("Tidak ada kata untuk sentimen negatif.")

            with c3:
                st.write("### Netral")
                if not df_neu.empty:
                    fig_neu = px.bar(
                        df_neu.sort_values("count"),
                        x="count",
                        y="word",
                        orientation="h",
                        labels={"count": "Frekuensi", "word": "Kata"},
                        title=f"Top {len(df_neu)} Kata - Netral",
                        height=420,
                    )
                    fig_neu.update_layout(
                        margin=dict(l=10, r=10, t=40, b=10), showlegend=False
                    )
                    st.plotly_chart(fig_neu, use_container_width=True)
                else:
                    st.info("Tidak ada kata untuk sentimen netral.")

        st.markdown("----------")

        # ================================
        # BIGRAM ANALYSIS
        # ================================
        st.markdown("## 🧩 Bigram & TF-IDF Visualization")

        def plot_bigram_barchart(text_series, top_n=10):
            # Vectorizer bigram
            vectorizer = CountVectorizer(ngram_range=(2, 2))
            X = vectorizer.fit_transform(text_series.dropna())

            # Hitung frekuensi bigram
            bigram_counts = X.sum(axis=0).A1
            bigrams = vectorizer.get_feature_names_out()

            df_bigram = (
                pd.DataFrame({"bigram": bigrams, "count": bigram_counts})
                .sort_values("count", ascending=False)
                .head(top_n)
                .sort_values("count", ascending=True)
            )

            # Plotly horizontal bar chart
            fig = px.bar(
                df_bigram,
                x="count",
                y="bigram",
                orientation="h",
                title=f"Top {top_n} Bigram Paling Sering Muncul",
                text="count",
            )
            fig.update_layout(
                height=500,
                xaxis_title="Frekuensi",
                yaxis_title="Bigram",
                showlegend=False,
            )
            fig.update_traces(
                marker_color="rgba(0, 100, 255, 0.7)", textposition="outside"
            )

            return fig

        def plot_tfidf_bubble(text_series, top_n=10):
            vectorizer = TfidfVectorizer(max_features=5000)
            X = vectorizer.fit_transform(text_series.dropna())

            tfidf_scores = X.sum(axis=0).A1
            words = vectorizer.get_feature_names_out()

            df_tfidf = (
                pd.DataFrame({"word": words, "score": tfidf_scores})
                .sort_values("score", ascending=False)
                .head(top_n)
            )
            fig = px.scatter(
                df_tfidf,
                x="word",
                y="score",
                size="score",
                color="score",
                title=f"Top {top_n} TF-IDF Words",
                hover_name="word",
            )

            fig.update_layout(
                height=500,
                xaxis_title="Kata",
                yaxis_title="Nilai TF-IDF",
                showlegend=False,
            )
            return fig

        col1, col2 = st.columns(2)

        with col1:
            st.plotly_chart(
                plot_bigram_barchart(df["stopword removal"]), use_container_width=True
            )
        with col2:
            st.plotly_chart(
                plot_tfidf_bubble(df["stopword removal"]), use_container_width=True
            )

        st.markdown("----------")

        # ================================
        # TABEL KOMENTAR
        # ================================
        st.subheader("📋 Tabel Komentar")
        st.dataframe(df, height=300)

        # ================================
        # PREDIKSI SENTIMEN KOMENTAR BARU
        # ================================
        st.subheader("📝 Prediksi Sentimen Komentar Baru")

        input_text = st.text_area("Masukkan komentar baru:")

        if st.button("🔍 Prediksi Sentimen"):
            if input_text.strip() == "":
                st.warning("Masukkan komentar terlebih dahulu.")
            else:
                pred = model_svm.predict([input_text])[0]
                st.success(f"Sentimen komentar: **{pred.upper()}**")
else:
    st.info("Silakan upload dataset CSV untuk memulai analisis.")


