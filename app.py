import streamlit as st
import pandas as pd
import joblib
import numpy as np
import io
import plotly.express as px
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="Movie Success Predictor",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for premium look
st.markdown("""
<style>
    /* Global styles */
    .main {
        background-color: #0E1117;
        color: #FAFAFA;
        font-family: 'Inter', sans-serif;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #FFFFFF;
        font-weight: 700;
    }
    
    /* Buttons */
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #FF416C 0%, #FF4B2B 100%);
        color: white;
        font-weight: bold;
        font-size: 18px;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(255, 75, 43, 0.4);
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(255, 75, 43, 0.6);
    }
    
    /* Prediction Boxes */
    .prediction-box {
        padding: 2.5rem;
        border-radius: 15px;
        text-align: center;
        margin: 2rem 0;
        animation: fadeIn 0.8s ease-out;
        color: white;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.5);
    }
    .hit { 
        background: linear-gradient(135deg, #11998e 0%, #2ecc71 100%); 
        box-shadow: 0 10px 30px rgba(46, 204, 113, 0.3);
    }
    .average { 
        background: linear-gradient(135deg, #f2994a 0%, #f2c94c 100%); 
        box-shadow: 0 10px 30px rgba(242, 201, 76, 0.3);
    }
    .flop { 
        background: linear-gradient(135deg, #cb2d3e 0%, #ef473a 100%); 
        box-shadow: 0 10px 30px rgba(239, 71, 58, 0.3);
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Custom Sidebar */
    [data-testid="stSidebar"] {
        background-color: #1A1C23;
        border-right: 1px solid #2D303E;
    }
</style>
""", unsafe_allow_html=True)

# Load models and data
@st.cache_resource
def load_models():
    model = joblib.load('models/best_movie_model_xgboost.pkl')
    ordinal_encoder = joblib.load('models/ordinal_encoder.pkl')
    scaler = joblib.load('models/standard_scaler.pkl')
    label_encoder = joblib.load('models/target_label_encoder.pkl')
    return model, ordinal_encoder, scaler, label_encoder

@st.cache_data
def load_data():
    df = pd.read_csv('cleaned_movie_data.csv')
    return df

try:
    model, ordinal_encoder, scaler, label_encoder = load_models()
    df = load_data()
    data_loaded = True
except Exception as e:
    st.error(f"Error loading models or data: {e}")
    data_loaded = False

if data_loaded:
    # Header
    col1, col2 = st.columns([1, 5])
    with col1:
        st.markdown("<h1 style='font-size: 4rem; text-align: center;'>🎬</h1>", unsafe_allow_html=True)
    with col2:
        st.title("Movie Success Predictor AI")
        st.markdown("Leverage advanced Machine Learning (XGBoost) to predict whether a movie will be a **Hit**, **Average**, or **Flop**. Analyze single movies or upload a dataset for batch predictions!")

    # Feature lists
    cat_cols = ['color', 'director_name', 'actor_2_name', 'genres', 'actor_1_name',
                'movie_title', 'actor_3_name', 'plot_keywords', 'language', 'country',
                'content_rating']
    
    num_cols = ['num_critic_for_reviews', 'duration', 'director_facebook_likes',
                'actor_3_facebook_likes', 'actor_1_facebook_likes', 'gross',
                'num_voted_users', 'facenumber_in_poster', 'num_user_for_reviews', 'budget',
                'title_year', 'actor_2_facebook_likes', 'aspect_ratio',
                'movie_facebook_likes']

    expected_features = ['color', 'director_name', 'num_critic_for_reviews', 'duration',
                         'director_facebook_likes', 'actor_3_facebook_likes', 'actor_2_name',
                         'actor_1_facebook_likes', 'gross', 'genres', 'actor_1_name', 'movie_title',
                         'num_voted_users', 'actor_3_name', 'facenumber_in_poster', 'plot_keywords',
                         'num_user_for_reviews', 'language', 'country', 'content_rating', 'budget',
                         'title_year', 'actor_2_facebook_likes', 'aspect_ratio',
                         'movie_facebook_likes']

    # Extract valid categories from encoder to prevent unknown category errors
    valid_categories = {}
    for i, col in enumerate(cat_cols):
        valid_categories[col] = ordinal_encoder.categories_[i]

    # Sidebar Navigation
    st.sidebar.markdown("<h2 style='text-align: center;'>Navigation</h2>", unsafe_allow_html=True)
    app_mode = st.sidebar.radio("Select Mode:", ["Single Movie Prediction", "Batch Dataset Analysis"])

    if app_mode == "Single Movie Prediction":
        st.sidebar.markdown("<h3 style='text-align: center;'>🎯 Select Movie</h3>", unsafe_allow_html=True)
        st.sidebar.markdown("Pick a movie from the dataset to load its features automatically. Then tweak the numbers!")
        
        movie_options = ["-- Select a Movie --"] + sorted(df['movie_title'].dropna().unique().tolist())
        selected_movie = st.sidebar.selectbox("Choose a Movie:", options=movie_options)

        if selected_movie != "-- Select a Movie --":
            movie_data = df[df['movie_title'] == selected_movie].iloc[0].to_dict()
        else:
            movie_data = df.iloc[0].to_dict()

        st.markdown("### 🔍 Single Prediction Engine")
        st.markdown("---")
        
        tab_engine, tab_context = st.tabs(["🚀 Prediction Engine", "📊 Dataset Context & Comparison"])
        
        with tab_engine:
            col1, col2, col3 = st.columns(3)
            input_data = {}

        with col1:
            st.markdown("#### 🎥 Production Details")
            def safe_selectbox(label, col_name, current_val):
                options = valid_categories[col_name].tolist()
                if current_val not in options:
                    options = [current_val] + options
                return st.selectbox(label, options, index=options.index(current_val))

            input_data['movie_title'] = safe_selectbox("Movie Title", 'movie_title', movie_data.get('movie_title', ''))
            input_data['director_name'] = safe_selectbox("Director Name", 'director_name', movie_data.get('director_name', ''))
            input_data['genres'] = safe_selectbox("Genres", 'genres', movie_data.get('genres', ''))
            input_data['plot_keywords'] = safe_selectbox("Plot Keywords", 'plot_keywords', movie_data.get('plot_keywords', ''))
            
            input_data['duration'] = st.number_input("Duration (minutes)", value=float(movie_data.get('duration', 120.0)), min_value=0.0)
            input_data['budget'] = st.number_input("Budget ($)", value=float(movie_data.get('budget', 50000000.0)), min_value=0.0)
            input_data['title_year'] = st.number_input("Title Year", value=float(movie_data.get('title_year', 2023.0)), min_value=1900.0, max_value=2100.0)

        with col2:
            st.markdown("#### ⭐ Cast & Crew")
            input_data['actor_1_name'] = safe_selectbox("Lead Actor", 'actor_1_name', movie_data.get('actor_1_name', ''))
            input_data['actor_1_facebook_likes'] = st.number_input("Lead Actor FB Likes", value=float(movie_data.get('actor_1_facebook_likes', 0.0)))
            
            input_data['actor_2_name'] = safe_selectbox("Supporting Actor 1", 'actor_2_name', movie_data.get('actor_2_name', ''))
            input_data['actor_2_facebook_likes'] = st.number_input("Supporting Actor 1 FB Likes", value=float(movie_data.get('actor_2_facebook_likes', 0.0)))
            
            input_data['actor_3_name'] = safe_selectbox("Supporting Actor 2", 'actor_3_name', movie_data.get('actor_3_name', ''))
            input_data['actor_3_facebook_likes'] = st.number_input("Supporting Actor 2 FB Likes", value=float(movie_data.get('actor_3_facebook_likes', 0.0)))

            input_data['director_facebook_likes'] = st.number_input("Director FB Likes", value=float(movie_data.get('director_facebook_likes', 0.0)))
            
        with col3:
            st.markdown("#### 🌍 Logistics & Reception")
            input_data['country'] = safe_selectbox("Country", 'country', movie_data.get('country', 'USA'))
            input_data['language'] = safe_selectbox("Language", 'language', movie_data.get('language', 'English'))
            input_data['content_rating'] = safe_selectbox("Content Rating", 'content_rating', movie_data.get('content_rating', 'PG-13'))
            input_data['color'] = safe_selectbox("Color/B&W", 'color', movie_data.get('color', 'Color'))
            
            input_data['num_critic_for_reviews'] = st.number_input("Critic Reviews Count", value=float(movie_data.get('num_critic_for_reviews', 0.0)))
            input_data['num_user_for_reviews'] = st.number_input("User Reviews Count", value=float(movie_data.get('num_user_for_reviews', 0.0)))
            input_data['num_voted_users'] = st.number_input("IMDB Voted Users", value=float(movie_data.get('num_voted_users', 0.0)))
            input_data['movie_facebook_likes'] = st.number_input("Movie FB Likes", value=float(movie_data.get('movie_facebook_likes', 0.0)))
            
            input_data['gross'] = st.number_input("Gross Revenue ($)", value=float(movie_data.get('gross', 0.0)))
            input_data['facenumber_in_poster'] = st.number_input("Faces in Poster", value=float(movie_data.get('facenumber_in_poster', 0.0)))
            input_data['aspect_ratio'] = st.number_input("Aspect Ratio", value=float(movie_data.get('aspect_ratio', 2.35)))

        st.markdown("---")

        _, center_col, _ = st.columns([1, 2, 1])
        with center_col:
            predict_button = st.button("🚀 Analyze & Predict Cinematic Success")

        if predict_button:
            input_df = pd.DataFrame([input_data])
            try:
                cat_input = input_df[cat_cols]
                encoded_cats = ordinal_encoder.transform(cat_input)
                
                num_input = input_df[num_cols]
                scaled_nums = scaler.transform(num_input)
                
                processed_df = pd.DataFrame(index=[0])
                for i, col in enumerate(cat_cols):
                    processed_df[col] = encoded_cats[0][i]
                for i, col in enumerate(num_cols):
                    processed_df[col] = scaled_nums[0][i]
                    
                processed_df = processed_df[expected_features]
                
                pred = model.predict(processed_df)
                pred_class = label_encoder.inverse_transform(pred)[0]
                
                probs = model.predict_proba(processed_df)[0] if hasattr(model, "predict_proba") else None

                if pred_class == 'Hit':
                    st.markdown(f"<div class='prediction-box hit'><h1 style='color: white;'>🎉 BOX OFFICE HIT!</h1><p style='font-size: 1.2rem; color: white;'>This movie is predicted to be highly successful and profitable.</p></div>", unsafe_allow_html=True)
                elif pred_class == 'Flop':
                    st.markdown(f"<div class='prediction-box flop'><h1 style='color: white;'>📉 BOX OFFICE FLOP</h1><p style='font-size: 1.2rem; color: white;'>This movie might struggle to recover its budget.</p></div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div class='prediction-box average'><h1 style='color: white;'>⚖️ AVERAGE PERFORMER</h1><p style='font-size: 1.2rem; color: white;'>This movie is predicted to have an average financial performance.</p></div>", unsafe_allow_html=True)
                
                if probs is not None:
                    st.markdown("<h3 style='text-align: center; margin-top: 2rem;'>Prediction Confidence Analysis</h3>", unsafe_allow_html=True)
                    classes = label_encoder.classes_
                    for idx, cls in enumerate(classes):
                        color = "#2ecc71" if cls == 'Hit' else "#f1c40f" if cls == 'Average' else "#e74c3c"
                        st.markdown(f"""
                        <div style="margin-bottom: 1rem;">
                            <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                                <span style="font-weight: bold; font-size: 1.1rem;">{cls}</span>
                                <span style="font-weight: bold; font-size: 1.1rem;">{probs[idx]*100:.1f}%</span>
                            </div>
                            <div style="width: 100%; background-color: #2D303E; border-radius: 10px; overflow: hidden; height: 20px;">
                                <div style="width: {probs[idx]*100}%; background-color: {color}; height: 100%; transition: width 1s ease-in-out;"></div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
            except Exception as e:
                st.error(f"Error during prediction: {e}")

        with tab_context:
            st.markdown("### 📚 Reference Dataset Explorer")
            st.markdown("Explore the `Cleaned Movie Dataset` used to train the model, and see how your currently selected movie stacks up against the rest of the industry!")
            
            # Show the dataset
            with st.expander("👁️ View Raw Cleaned Dataset", expanded=False):
                st.dataframe(df, use_container_width=True, height=400)
                
            st.markdown("### 📊 Market Positioning")
            
            # Plotly Visualizations comparing selected movie to the rest
            c_vis1, c_vis2 = st.columns(2)
            
            # Scatter Plot: Budget vs Gross
            with c_vis1:
                fig_scatter_ctx = px.scatter(
                    df, x='budget', y='gross', 
                    color='Classify', 
                    color_discrete_map={'Hit': '#2ecc71', 'Average': '#f1c40f', 'Flop': '#e74c3c'},
                    hover_data=['movie_title'],
                    title="Budget vs Gross Landscape",
                    opacity=0.3
                )
                
                # Highlight selected movie if one is selected
                if selected_movie != "-- Select a Movie --":
                    sel_b = input_data.get('budget', 0)
                    sel_g = input_data.get('gross', 0)
                    
                    fig_scatter_ctx.add_trace(go.Scatter(
                        x=[sel_b], y=[sel_g],
                        mode='markers+text',
                        marker=dict(color='white', size=15, symbol='star', line=dict(color='black', width=2)),
                        text=[f"⭐ {selected_movie}"],
                        textposition="top center",
                        name="Selected Movie"
                    ))
                    
                fig_scatter_ctx.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_scatter_ctx, use_container_width=True)
                
            with c_vis2:
                # Histogram: Distribution of Facebook Likes
                fig_hist_ctx = px.histogram(
                    df, x='movie_facebook_likes',
                    title="Distribution of Movie Facebook Likes",
                    nbins=50,
                    opacity=0.7,
                    color_discrete_sequence=['#3498db']
                )
                
                if selected_movie != "-- Select a Movie --":
                    sel_fb = input_data.get('movie_facebook_likes', 0)
                    fig_hist_ctx.add_vline(
                        x=sel_fb, line_width=4, line_dash="dash", line_color="white",
                        annotation_text=f"⭐ {selected_movie}", annotation_position="top right"
                    )
                
                fig_hist_ctx.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                # Cut off massive outliers in Facebook likes for a cleaner view
                fig_hist_ctx.update_xaxes(range=[0, df['movie_facebook_likes'].quantile(0.95)])
                st.plotly_chart(fig_hist_ctx, use_container_width=True)
            
            st.info("💡 **Context Tip:** The white star/dashed line represents your currently configured movie in the 'Prediction Engine' tab. If you tweak the budget or Facebook likes and return here, the star will move, allowing you to visually position your hypothetical movie in the real-world market!")

    else:
        st.markdown("### 📊 Batch Dataset Analysis")
        st.markdown("Upload a CSV file containing multiple movies to generate predictions in bulk. The AI will validate the dataset, handle missing data or unseen categories gracefully, and generate a downloadable report with success predictions.")
        
        uploaded_file = st.file_uploader("Upload your dataset (CSV format)", type=['csv'])
        
        if uploaded_file is not None:
            try:
                # Load the dataset
                batch_df = pd.read_csv(uploaded_file)
                st.success(f"Successfully loaded dataset with {len(batch_df)} rows and {len(batch_df.columns)} columns.")
                
                # Validation 1: Check required columns
                missing_cols = [col for col in expected_features if col not in batch_df.columns]
                
                if missing_cols:
                    st.error(f"Dataset Validation Failed. Missing {len(missing_cols)} required columns.")
                    st.warning(f"Missing columns: {', '.join(missing_cols)}")
                    st.info("Please upload a dataset containing all the features expected by the model.")
                else:
                    with st.spinner("Analyzing and predicting..."):
                        # Basic cleaning: remove rows with missing required features
                        batch_df = batch_df.dropna(subset=expected_features).copy()
                            
                        if len(batch_df) == 0:
                            st.error("The dataset was empty or contained only missing values in required columns. Please check your data.")
                        else:
                            # Handle unseen categories gracefully without warnings
                            for i, col in enumerate(cat_cols):
                                valid_cats = set(valid_categories[col])
                                fallback_cat = valid_categories[col][0] # use the first known category as fallback
                                
                                # Replace unseen categories
                                unseen_mask = ~batch_df[col].isin(valid_cats)
                                if unseen_mask.any():
                                    num_unseen = unseen_mask.sum()
                                    batch_df.loc[unseen_mask, col] = fallback_cat
                                    # Optional: st.info(f"Replaced {num_unseen} unseen categories in '{col}' with default.")

                            # Feature Engineering / Preparation
                            cat_input = batch_df[cat_cols]
                            encoded_cats = ordinal_encoder.transform(cat_input)
                            
                            num_input = batch_df[num_cols]
                            scaled_nums = scaler.transform(num_input)
                            
                            # Reconstruct dataframe for prediction in exact order
                            processed_df = pd.DataFrame(index=batch_df.index)
                            for i, col in enumerate(cat_cols):
                                processed_df[col] = encoded_cats[:, i]
                            for i, col in enumerate(num_cols):
                                processed_df[col] = scaled_nums[:, i]
                                
                            processed_df = processed_df[expected_features]
                            
                            # Prediction
                            preds = model.predict(processed_df)
                            pred_classes = label_encoder.inverse_transform(preds)
                            
                            batch_df['Predicted_Success'] = pred_classes
                            
                            if hasattr(model, "predict_proba"):
                                probs = model.predict_proba(processed_df)
                                classes = label_encoder.classes_
                                for idx, cls in enumerate(classes):
                                    batch_df[f'Probability_{cls}'] = probs[:, idx].round(4)
                            
                            # ===== DASHBOARD =====
                            st.markdown("### 🏆 Prediction Results Dashboard")
                            
                            # --- Top-level KPI metrics ---
                            hit_count = (batch_df['Predicted_Success'] == 'Hit').sum()
                            avg_count = (batch_df['Predicted_Success'] == 'Average').sum()
                            flop_count = (batch_df['Predicted_Success'] == 'Flop').sum()
                            total = len(batch_df)
                            avg_budget = batch_df['budget'].mean()
                            avg_gross = batch_df['gross'].mean()
                            
                            c1, c2, c3, c4, c5 = st.columns(5)
                            c1.metric("Total Movies", f"{total}")
                            c2.metric("Hits 🎉", f"{hit_count}", f"{(hit_count/total)*100:.1f}%")
                            c3.metric("Average ⚖️", f"{avg_count}", f"{(avg_count/total)*100:.1f}%")
                            c4.metric("Flops 📉", f"{flop_count}", f"{(flop_count/total)*100:.1f}%")
                            c5.metric("Avg Budget", f"${avg_budget/1e6:.1f}M")
                            
                            st.markdown("---")
                            
                            color_map = {'Hit': '#2ecc71', 'Average': '#f1c40f', 'Flop': '#e74c3c'}
                            
                            # --- Tabbed dashboard ---
                            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                                "📊 Overview", "💰 Financial Analysis", "🎭 Genre Breakdown",
                                "📈 Confidence Analysis", "📋 Full Data"
                            ])
                            
                            with tab1:
                                col_a, col_b = st.columns(2)
                                with col_a:
                                    # Donut chart of prediction distribution
                                    counts = batch_df['Predicted_Success'].value_counts()
                                    fig_donut = go.Figure(data=[go.Pie(
                                        labels=counts.index,
                                        values=counts.values,
                                        hole=0.55,
                                        marker_colors=[color_map.get(c, '#888') for c in counts.index],
                                        textinfo='label+percent',
                                        textfont_size=14
                                    )])
                                    fig_donut.update_layout(
                                        title_text="Prediction Distribution",
                                        template='plotly_dark',
                                        paper_bgcolor='rgba(0,0,0,0)',
                                        plot_bgcolor='rgba(0,0,0,0)',
                                        height=400,
                                        showlegend=False
                                    )
                                    st.plotly_chart(fig_donut, use_container_width=True)
                                
                                with col_b:
                                    # Year distribution of predictions
                                    if 'title_year' in batch_df.columns:
                                        year_df = batch_df.groupby(['title_year', 'Predicted_Success']).size().reset_index(name='count')
                                        fig_year = px.bar(
                                            year_df, x='title_year', y='count', color='Predicted_Success',
                                            color_discrete_map=color_map,
                                            title='Predictions by Release Year',
                                            labels={'title_year': 'Year', 'count': 'Number of Movies'}
                                        )
                                        fig_year.update_layout(
                                            template='plotly_dark',
                                            paper_bgcolor='rgba(0,0,0,0)',
                                            plot_bgcolor='rgba(0,0,0,0)',
                                            height=400,
                                            barmode='stack'
                                        )
                                        st.plotly_chart(fig_year, use_container_width=True)
                            
                            with tab2:
                                col_a, col_b = st.columns(2)
                                with col_a:
                                    # Budget vs Gross scatter
                                    fig_scatter = px.scatter(
                                        batch_df, x='budget', y='gross',
                                        color='Predicted_Success',
                                        color_discrete_map=color_map,
                                        hover_data=['movie_title'],
                                        title='Budget vs. Gross Revenue',
                                        labels={'budget': 'Budget ($)', 'gross': 'Gross ($)'},
                                        opacity=0.7
                                    )
                                    fig_scatter.update_layout(
                                        template='plotly_dark',
                                        paper_bgcolor='rgba(0,0,0,0)',
                                        plot_bgcolor='rgba(0,0,0,0)',
                                        height=450
                                    )
                                    st.plotly_chart(fig_scatter, use_container_width=True)
                                
                                with col_b:
                                    # Budget distribution by outcome (box plot)
                                    fig_box = px.box(
                                        batch_df, x='Predicted_Success', y='budget',
                                        color='Predicted_Success',
                                        color_discrete_map=color_map,
                                        title='Budget Distribution by Predicted Outcome',
                                        labels={'budget': 'Budget ($)', 'Predicted_Success': 'Outcome'}
                                    )
                                    fig_box.update_layout(
                                        template='plotly_dark',
                                        paper_bgcolor='rgba(0,0,0,0)',
                                        plot_bgcolor='rgba(0,0,0,0)',
                                        height=450,
                                        showlegend=False
                                    )
                                    st.plotly_chart(fig_box, use_container_width=True)
                                
                                # ROI Analysis
                                roi_df = batch_df[batch_df['budget'] > 0].copy()
                                if len(roi_df) > 0:
                                    roi_df['ROI'] = ((roi_df['gross'] - roi_df['budget']) / roi_df['budget'] * 100).round(1)
                                    fig_roi = px.histogram(
                                        roi_df, x='ROI', color='Predicted_Success',
                                        color_discrete_map=color_map,
                                        title='Return on Investment (ROI) Distribution',
                                        labels={'ROI': 'ROI (%)', 'count': 'Number of Movies'},
                                        nbins=40, barmode='overlay', opacity=0.7
                                    )
                                    fig_roi.update_layout(
                                        template='plotly_dark',
                                        paper_bgcolor='rgba(0,0,0,0)',
                                        plot_bgcolor='rgba(0,0,0,0)',
                                        height=350
                                    )
                                    st.plotly_chart(fig_roi, use_container_width=True)
                            
                            with tab3:
                                # Genre breakdown
                                if 'genres' in batch_df.columns:
                                    # Extract individual genres (they are pipe-separated)
                                    genre_rows = []
                                    for _, row in batch_df.iterrows():
                                        genres_str = str(row.get('genres', ''))
                                        for g in genres_str.split('|'):
                                            g = g.strip()
                                            if g:
                                                genre_rows.append({'genre': g, 'Predicted_Success': row['Predicted_Success']})
                                    
                                    if genre_rows:
                                        genre_df = pd.DataFrame(genre_rows)
                                        
                                        col_a, col_b = st.columns(2)
                                        with col_a:
                                            # Top genres by count
                                            top_genres = genre_df['genre'].value_counts().head(15)
                                            fig_genres = px.bar(
                                                x=top_genres.values, y=top_genres.index,
                                                orientation='h',
                                                title='Top 15 Genres in Dataset',
                                                labels={'x': 'Count', 'y': 'Genre'},
                                                color=top_genres.values,
                                                color_continuous_scale='Viridis'
                                            )
                                            fig_genres.update_layout(
                                                template='plotly_dark',
                                                paper_bgcolor='rgba(0,0,0,0)',
                                                plot_bgcolor='rgba(0,0,0,0)',
                                                height=500,
                                                yaxis=dict(autorange='reversed'),
                                                showlegend=False,
                                                coloraxis_showscale=False
                                            )
                                            st.plotly_chart(fig_genres, use_container_width=True)
                                        
                                        with col_b:
                                            # Genre success rate heatmap
                                            genre_success = genre_df.groupby(['genre', 'Predicted_Success']).size().unstack(fill_value=0)
                                            top_genre_names = top_genres.index.tolist()
                                            genre_success = genre_success.loc[genre_success.index.isin(top_genre_names)]
                                            
                                            # Normalize to percentages
                                            genre_pct = genre_success.div(genre_success.sum(axis=1), axis=0) * 100
                                            
                                            fig_heatmap = go.Figure(data=go.Heatmap(
                                                z=genre_pct.values,
                                                x=genre_pct.columns.tolist(),
                                                y=genre_pct.index.tolist(),
                                                colorscale='RdYlGn',
                                                text=genre_pct.values.round(1),
                                                texttemplate='%{text}%',
                                                textfont={"size": 12}
                                            ))
                                            fig_heatmap.update_layout(
                                                title='Genre × Outcome Success Rate (%)',
                                                template='plotly_dark',
                                                paper_bgcolor='rgba(0,0,0,0)',
                                                plot_bgcolor='rgba(0,0,0,0)',
                                                height=500
                                            )
                                            st.plotly_chart(fig_heatmap, use_container_width=True)
                                    else:
                                        st.info("No genre data could be parsed from this dataset.")
                                else:
                                    st.info("No 'genres' column found in the dataset.")
                            
                            with tab4:
                                if hasattr(model, "predict_proba"):
                                    classes = label_encoder.classes_
                                    prob_cols = [f'Probability_{cls}' for cls in classes]
                                    
                                    col_a, col_b = st.columns(2)
                                    with col_a:
                                        # Average confidence by outcome
                                        avg_probs = batch_df.groupby('Predicted_Success')[prob_cols].mean()
                                        fig_conf = go.Figure()
                                        for cls in classes:
                                            fig_conf.add_trace(go.Bar(
                                                name=cls,
                                                x=avg_probs.index,
                                                y=avg_probs[f'Probability_{cls}'],
                                                marker_color=color_map.get(cls, '#888')
                                            ))
                                        fig_conf.update_layout(
                                            title='Average Model Confidence by Predicted Outcome',
                                            template='plotly_dark',
                                            paper_bgcolor='rgba(0,0,0,0)',
                                            plot_bgcolor='rgba(0,0,0,0)',
                                            barmode='group',
                                            height=400,
                                            yaxis_title='Average Probability'
                                        )
                                        st.plotly_chart(fig_conf, use_container_width=True)
                                    
                                    with col_b:
                                        # Confidence distribution (violin)
                                        # Max probability = how sure the model is about its top pick
                                        batch_df['Max_Confidence'] = batch_df[prob_cols].max(axis=1)
                                        fig_violin = px.violin(
                                            batch_df, x='Predicted_Success', y='Max_Confidence',
                                            color='Predicted_Success',
                                            color_discrete_map=color_map,
                                            box=True, points='outliers',
                                            title='Model Confidence Distribution (Top-Class Probability)',
                                            labels={'Max_Confidence': 'Confidence', 'Predicted_Success': 'Outcome'}
                                        )
                                        fig_violin.update_layout(
                                            template='plotly_dark',
                                            paper_bgcolor='rgba(0,0,0,0)',
                                            plot_bgcolor='rgba(0,0,0,0)',
                                            height=400,
                                            showlegend=False
                                        )
                                        st.plotly_chart(fig_violin, use_container_width=True)
                                    
                                    # Low-confidence flagging
                                    low_conf = batch_df[batch_df['Max_Confidence'] < 0.5]
                                    if len(low_conf) > 0:
                                        st.warning(f"⚠️ {len(low_conf)} movie(s) have low model confidence (<50%). These predictions should be treated with caution.")
                                        st.dataframe(
                                            low_conf[['movie_title', 'Predicted_Success', 'Max_Confidence'] + prob_cols].sort_values('Max_Confidence'),
                                            use_container_width=True
                                        )
                                else:
                                    st.info("Probability analysis is not available for this model type.")
                            
                            with tab5:
                                st.markdown("#### Complete Predictions Table")
                                display_cols = ['movie_title', 'Predicted_Success'] + [c for c in batch_df.columns if 'Probability' in c]
                                # Add useful context columns if they exist
                                for extra in ['budget', 'gross', 'duration', 'genres', 'director_name', 'title_year']:
                                    if extra in batch_df.columns and extra not in display_cols:
                                        display_cols.append(extra)
                                
                                st.dataframe(batch_df[display_cols], use_container_width=True, height=500)
                                
                                # Provide Download Button
                                csv = batch_df.to_csv(index=False).encode('utf-8')
                                st.download_button(
                                    label="⬇️ Download Complete Predictions CSV",
                                    data=csv,
                                    file_name='movie_success_predictions.csv',
                                    mime='text/csv',
                                )

            except Exception as e:
                st.error(f"An error occurred while processing the file: {e}")
                st.exception(e)
