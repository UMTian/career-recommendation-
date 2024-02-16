import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from PIL import Image

# Load and Explore Dataset
data = pd.read_csv("C:\\Users\\LENOVO\\Desktop\\project\\Career recommendation\\dataset.csv")

# Combine Interests and Skills
data['combined_features'] = data['Interest'] + ' ' + data['Skills']

# TF-IDF Vectorization
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(data['combined_features'])

# Calculate Similarity Scores
content_similarity_matrix = linear_kernel(tfidf_matrix, tfidf_matrix)


# Function to Get Content-Based Recommendations (Top 3)
def content_based_recommendation(user_profile, content_similarity_matrix, num_recommendations=3):
    user_vector = vectorizer.transform([user_profile])
    scores = linear_kernel(user_vector, tfidf_matrix).flatten()

    # Find the indices of the top N similarity scores
    top_field_indices = scores.argsort()[-num_recommendations:][::-1]
    recommended_fields = data['Field'].iloc[top_field_indices].tolist()

    return recommended_fields


# Streamlit Web App
st.title("Content-Based Recommendation System")

# Input Boxes for User Profile
interests = st.text_input("Enter interests (e.g., 'Programming'):")
skills = st.text_input("Enter skills (e.g., 'c++, research'):")
additional_info = st.text_input("Extra info (e.g., '........'):")

# Combine User Inputs
user_profile = f"{interests} {skills} {additional_info}"

# Display Recommendations with Relevant Photos
if st.button("Get Recommendations"):
    recommended_fields = content_based_recommendation(user_profile, content_similarity_matrix, num_recommendations=3)
    st.subheader("Top 3 Recommended Fields:")
    st.write(recommended_fields)

    # Display Relevant Photos
    st.subheader("Relevant Photos:")
    for field in recommended_fields:
        # Replace the placeholder image URLs with your actual image paths
        field_to_image = {
            "data sciences" : "C:\\Users\\LENOVO\\Downloads\\artificial-intelligence.jpg",
            "robotics" : "C:\\Users\\LENOVO\\Downloads\\download.jpg",
            "fashion designing" :"C:\\Users\LENOVO\Pictures\\010-women-dressing-women-costume-institute-GalleryView-Agency.jpg",
            "math" :"C:\\Users\\LENOVO\\Pictures\\index.jpeg",
            "graphic design" :"C:\\Users\\LENOVO\\Pictures\\images (1).jpeg",
            "chemistry" :"C:\\Users\\LENOVO\\Pictures\\GettyImages-545286316-433dd345105e4c6ebe4cdd8d2317fdaa (1).jpg",
            "automotive engineering" :"C:\\Users\\LENOVO\\Pictures\\masters-in-automotive-engineering-usa.jpg",
            "electrical engineering" :"C:\\Users\\LENOVO\\Pictures\\index (1).jpeg",
            "physics" :"C:\\Users\\LENOVO\\Downloads\\download (8).jpeg",
            "chemical engineering" :"C:\\Users\\LENOVO\\Pictures\\Chemistry-Vs-Chemical-Engineering.png",
            "painting": "C:\\Users\\LENOVO\\Downloads\\download (9).jpeg",
            "civil engineering": "C:\\Users\\LENOVO\\Downloads\\download (3).jpeg",
            "drama": "C:\\Users\\LENOVO\\Pictures\\cover-1630324415438.png",
            "biology": "C:\\Users\\LENOVO\\Pictures\\What-is-Medical-Biology-Bolton-Uni__ResizedImageWzYwMCwzMzhd.jpg",
            "astronomy": "C:\\Users\\LENOVO\\Downloads\\download (5).jpeg",
            "earth sciences": "C:\\Users\\LENOVO\\Pictures\\41561_2012_Article_BFngeo1567_Fig1_HTML.jpg",
            "environmental sciences" :"C:\\Users\\LENOVO\\Pictures\\Environmental-Science-Ecosystem-600.jpg",
            "geography": "C:\\Users\\LENOVO\\Pictures\\human-geography.png",
            "software engineering": "C:\\Users\\LENOVO\\Pictures\\software-development-specialist.jpg",
            "cyber security": "C:\\Users\\LENOVO\\Pictures\\Cyber-Security-Icon-Concept-2-1.jpeg",
            "photography": "C:\\Users\\LENOVO\\Pictures\\images.jpeg",
            "english": "C:\\Users\\LENOVO\\Pictures\\english-british-england-language-education-concept-58368527.jpg",
            "Artificial Intelligence": "C:\\Users\\LENOVO\\Downloads\\download (1).jpeg",
        }

        if field in field_to_image:
            image_path = field_to_image[field]
            image = Image.open(image_path)
            st.image(image, caption=field, use_column_width=True)

# Display Dataset Preview
st.header("Dataset Preview")
st.dataframe(data.head())


