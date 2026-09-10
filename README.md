📡 SatQuery AI

Interactive Vision-Language Assistant for Satellite Image Analysis

SatQuery AI is an AI-powered web application that helps users understand and analyze satellite imagery using computer vision, vision-language models, and natural-language interaction.

The system allows users to upload satellite images, ask questions about them, estimate land-cover categories, and compare images from different dates to identify potential changes.

---

🎯 Problem Statement

Satellite images contain valuable information about urban areas, vegetation, agriculture, water bodies, and other land-cover features. However, analyzing these images manually can be difficult and time-consuming, especially for users without specialized remote-sensing knowledge.

SatQuery AI aims to provide an accessible AI-based interface where users can interact with satellite imagery using simple natural-language queries.

---

💡 Our Solution

SatQuery AI combines AI-based image understanding with an interactive dashboard.

Users can:

- Upload a satellite image
- Ask questions about the image
- Generate an AI description of the image
- Estimate major land-cover categories
- Compare two images from different dates
- Visualize potential changes
- Download the generated change-analysis overlay

---

✨ Key Features

🛰️ 1. AI Image Understanding

Upload a satellite image and interact with it using natural-language questions.

The application uses a Vision-Language Model to generate answers based on the uploaded image.

📝 2. Automatic Image Captioning

SatQuery AI can automatically generate a textual description of an uploaded image.

Example:

«"A view of a city from the air."»

🌍 3. Land-Cover Estimation

The system estimates the relevance of different land-cover categories using CLIP-based image-text similarity.

Current categories include:

- 🏙️ Urban / Built-up
- 🌳 Vegetation
- 💧 Water
- 🌾 Agriculture
- 🛣️ Roads

The results are presented as AI-estimated scores for easier interpretation.

🔄 4. Bi-Temporal Change Analysis

Users can upload two images representing the same area at different points in time.

The application compares the images and generates a visual representation of potential changes.

Users can also download the generated change overlay.

🤖 5. AI Agent

The AI Agent provides natural-language interaction with the satellite image.

It can route different types of requests, including:

- Image questions
- Image descriptions
- Land-cover analysis
- Full image analysis
- Change-analysis requests

⚠️ 6. AI Limitations & Disclaimer

The system clearly communicates that its results are AI-generated estimates and should not be treated as scientifically validated remote-sensing measurements.

---

🧠 AI Models

SatQuery AI currently uses the following pretrained models:

BLIP VQA

Salesforce/blip-vqa-base

Used for visual question answering.

BLIP Image Captioning

Salesforce/blip-image-captioning-base

Used for automatic image caption generation.

CLIP

openai/clip-vit-base-patch32

Used for image-text similarity and land-cover category estimation.

---

🏗️ System Architecture

                    ┌──────────────────────┐
                    │       User           │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │  Streamlit Web App   │
                    └──────────┬───────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
              ▼                ▼                ▼
       ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
       │    BLIP     │  │    BLIP     │  │    CLIP     │
       │     VQA     │  │  Captioning │  │ Land Cover  │
       └─────────────┘  └─────────────┘  └─────────────┘
              │                │                │
              └────────────────┼────────────────┘
                               ▼
                    ┌──────────────────────┐
                    │   AI Analysis &      │
                    │    Visualization     │
                    └──────────────────────┘

---

🛠️ Technology Stack

Technology| Purpose
Python| Core programming language
Streamlit| Web application interface
PyTorch| Deep learning framework
Hugging Face Transformers| AI model integration
BLIP| Visual question answering & captioning
CLIP| Land-cover estimation
PIL| Image processing
NumPy| Numerical operations
Git & GitHub| Version control

---

📂 Project Structure

SatQuery-AI/
│
├── app/
│   ├── main.py
│   ├── ai_model.py
│   ├── caption_model.py
│   ├── land_cover.py
│   ├── change_analysis.py
│   ├── inspect_dataset.py
│   ├── inspect_record.py
│   ├── test_dataset.py
│   ├── test_model.py
│   ├── test_vqa.py
│   └── main_backup.py
│
├── data/
│   └── satellite.png.png
│
├── .gitignore
│
└── README.md

---

🚀 Installation

1. Clone the repository

git clone https://github.com/vinutha53305/SatQuery-AI.git

2. Open the project

cd SatQuery-AI

3. Create a virtual environment

Windows:

python -m venv venv

4. Activate the environment

venv\Scripts\activate

5. Install dependencies

Install the required Python packages:

pip install streamlit torch transformers pillow numpy

6. Run the application

python -m streamlit run app/main.py

The application will open in your browser.

---

🖥️ Application Modules

Dashboard

Provides an overview of the SatQuery AI system and its major capabilities.

AI Agent

Allows users to interact with uploaded satellite imagery using natural-language questions.

Change Analysis

Compares two satellite images and visualizes potential changes between them.

Land Cover

Provides AI-estimated scores for major land-cover categories.

---

🔬 Example Use Cases

SatQuery AI can support exploratory analysis in areas such as:

- 🏙️ Urban development monitoring
- 🌳 Vegetation observation
- 🌾 Agricultural monitoring
- 💧 Water-body observation
- 🛣️ Infrastructure and road analysis
- 🛰️ Educational satellite-image exploration
- 🔄 Visual comparison of satellite imagery

---

⚠️ Limitations

SatQuery AI is currently a prototype.

The AI models used are general-purpose pretrained vision-language models and are not specialized scientific remote-sensing models.

Therefore:

- Land-cover values are AI-estimated scores.
- Results may contain inaccuracies.
- Change detection indicates potential visual differences rather than scientifically validated environmental change.
- Results should not be used as a replacement for professional remote-sensing analysis.

---

🔮 Future Scope

Future versions can include:

- 🌍 Integration with real satellite data APIs
- 🛰️ Sentinel-2 and Landsat-specific models
- 🧠 Fine-tuned remote-sensing foundation models
- 📊 More accurate land-cover segmentation
- 📍 Geographic coordinates and map integration
- 📅 Historical satellite-image retrieval
- 📈 Time-series analysis
- 🔥 Disaster and flood monitoring
- 🌳 Deforestation monitoring
- 🏗️ Urban expansion monitoring
- 👥 Multi-user collaboration
- ☁️ Cloud deployment

---

👥 Team

SatQuery AI — Smart India Hackathon 2026

Developed as a collaborative student project focused on applying Artificial Intelligence and Computer Vision to satellite-image analysis.

---

📌 Project Status

Current Status: 🚧 Prototype / Active Development

The project is being continuously improved with additional AI capabilities, better visualization, and remote-sensing functionality.

---

📜 License

This project is intended for educational, research, and hackathon purposes.

---

⭐ Acknowledgements

This project uses open-source technologies and pretrained models from the Hugging Face ecosystem and OpenAI CLIP.

Special thanks to the open-source AI and satellite-imagery communities.