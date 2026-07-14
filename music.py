import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

# 1. Custom Page Configuration & Spotify Dark Theme Styling
st.set_page_config(page_title="Music Recommendation System", layout="wide", page_icon="🎵")

st.markdown("""
    <style>
    /* Global Background and Text styles */
    .stApp {
        background-color: #121212;
        color: #ffffff;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #000000 !important;
        border-right: 1px solid #282828;
    }
    
    /* Navigation Link Styles */
    .nav-item {
        padding: 10px;
        font-size: 16px;
        font-weight: bold;
        color: #b3b3b3;
        display: flex;
        align-items: center;
        cursor: pointer;
    }
    .nav-item-active {
        color: #1DB954;
        background-color: #282828;
        border-radius: 4px;
    }
    
    /* Card design for mixes */
    .mix-card {
        background: linear-gradient(180deg, #282828 0%, #181818 100%);
        padding: 15px;
        border-radius: 8px;
        text-align: center;
        transition: background 0.3s ease;
    }
    .mix-card:hover {
        background: #282828;
    }
    
    /* Tracks list styles */
    .track-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 10px;
        border-radius: 4px;
        margin-bottom: 5px;
    }
    .track-row:hover {
        background-color: #2a2a2a;
    }
    
    /* Player layout styling */
    .player-bar {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background-color: #181818;
        border-top: 1px solid #282828;
        padding: 15px;
        z-index: 9999;
    }
    </style>
""", unsafe_allow_index=True)

# 2. Mock Database generation (Simulating real tracks data)
@st.cache_data
def load_data():
    data = {
        'Song': ['Believer', 'Blinding Lights', 'Someone You Loved', 'Heat Waves', 'Perfect', 'Photograph', 'Memories', 'Thunder', 'Havana', 'Counting Stars'],
        'Artist': ['Imagine Dragons', 'The Weeknd', 'Lewis Capaldi', 'Glass Animals', 'Ed Sheeran', 'Ed Sheeran', 'Maroon 5', 'Imagine Dragons', 'Camila Cabello', 'OneRepublic'],
        'Album': ['Evolve', 'After Hours', 'Divinely Uninspired', 'Dreamland', '÷ (Deluxe)', 'x (Deluxe Edition)', 'JORDI (Deluxe)', 'Evolve', 'Camila', 'Native'],
        'Duration': ['3:24', '3:20', '3:02', '3:58', '4:23', '4:19', '3:09', '3:07', '3:36', '4:17'],
        'Genre': ['Indie Rock', 'Synthwave Pop', 'Pop Vocal', 'Indie Pop', 'Pop Romantic', 'Pop Acoustic', 'Pop Nostalgic', 'Indie Rock', 'Latin Pop', 'Pop Rock']
    }
    return pd.DataFrame(data)

df = load_data()

# 3. Recommendation Engine Algorithm
def get_recommendations(song_title, df):
    # Combine feature vectors to find exact similarities
    df['features'] = df['Artist'] + " " + df['Genre']
    cv = CountVectorizer()
    count_matrix = cv.fit_transform(df['features'])
    cosine_sim = cosine_similarity(count_matrix, count_matrix)
    
    try:
        idx = df[df['Song'] == song_title].index[0]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        # Fetch top 4 recommendations excluding the current playing song
        song_indices = [i[0] for i in sim_scores if i[0] != idx][:4]
        return df.iloc[song_indices]
    except:
        return df.head(4)

# --- SIDEBAR UI ---
with st.sidebar:
    st.title("🎵 Music Recs")
    st.markdown('<div class="nav-item nav-item-active">🏠 Home</div>', unsafe_allow_index=True)
    st.markdown('<div class="nav-item">🔍 Search</div>', unsafe_allow_index=True)
    st.markdown('<div class="nav-item">📚 Library</div>', unsafe_allow_index=True)
    
    st.write("---")
    st.subheader("YOUR LIBRARY")
    st.caption("💜 Liked Songs")
    st.caption("🕒 Recently Played")
    
    st.write("---")
    st.subheader("PLAYLISTS")
    st.caption("🎶 Chill Vibes")
    st.caption("⚡ Workout Mix")
    st.caption("😢 Sad Songs")

# --- MAIN PAGE UI ---
# Top Search Bar & Header
col1, col2 = st.columns([4, 1])
with col1:
    search_query = st.selectbox("Search for songs, artists, albums...", df['Song'].tolist())
with col2:
    st.markdown("🌐 **Suraj** ▾")

st.header("Good Evening, Suraj")

# Made For You Grid Section
st.subheader("Made for you")
mix_cols = st.columns(5)
mixes = ["Discover Weekly", "Daily Mix 1", "Chill Mix", "Workout Mix", "Romantic Mix"]
taglines = ["Your weekly mixtape", "Popular & trending", "Relax and unwind", "High energy beats", "Love and feelings"]

for idx, col in enumerate(mix_cols):
    with col:
        st.markdown(f"""
        <div class="mix-card">
            <h4 style="color:#1DB954;">{mixes[idx]}</h4>
            <p style="font-size:12px; color:#b3b3b3;">{taglines[idx]}</p>
        </div>
        """, unsafe_allow_index=True)

st.write("---")

# Recommended Table Section
st.subheader("Recommended For You")
recommended_df = get_recommendations(search_query, df)

# Simulating Table Layout
for index, row in recommended_df.iterrows():
    col_s, col_a, col_al, col_d = st.columns([3, 3, 3, 1])
    with col_s:
        st.markdown(f"▶️ **{row['Song']}**")
    with col_a:
        st.markdown(f"<span style='color:#b3b3b3'>{row['Artist']}</span>", unsafe_allow_index=True)
    with col_al:
        st.markdown(f"<span style='color:#b3b3b3'>{row['Album']}</span>", unsafe_allow_index=True)
    with col_d:
        st.markdown(f"⏱️ {row['Duration']}")

# --- NOW PLAYING BOTTOM CONTROL BAR ---
st.markdown("<br><br><br>", unsafe_allow_index=True)
st.markdown(f"""
<div class="player-bar">
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <div>
            <b style="color: #ffffff;">Now Playing: {search_query}</b><br>
            <small style="color: #b3b3b3;">Streaming Source Activated</small>
        </div>
        <div style="color: #1DB954; font-size: 20px;">
            ⏮️ ⏸️ ⏭️ 🔁
        </div>
        <div>
            <small style="color: #b3b3b3;">🔊 █ █ █ █ ░ ░</small>
        </div>
    </div>
</div>
""", unsafe_allow_index=True)
