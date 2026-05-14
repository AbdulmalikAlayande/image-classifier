import streamlit as st

st.set_page_config(page_title="About", page_icon="ℹ️")

st.title("About this project")

st.markdown(
    """
    ## What is this?

    A binary image classifier that distinguishes between **AI-generated images**
    and **real-world photographs**.

    ## Why does it matter?

    As AI image generation becomes more sophisticated, distinguishing real from
    synthetic media is increasingly important for:

    - **Combating misinformation** — verifying the authenticity of news images
    - **Protecting intellectual property** — detecting AI-generated content
    - **Maintaining media credibility** — ensuring trust in visual evidence

    ## How was it built?

    - **Architecture**: Sequential Convolutional Neural Network (CNN)
    - **Layers**: 5× Conv2D + MaxPooling blocks, followed by a Dense head with sigmoid output
    - **Input**: 300×300 RGB images, normalized to [0, 1]
    - **Training**: Data augmentation (horizontal flip, rotation, zoom) on a 70/15/15 train/val/test split
    - **Loss**: Binary cross-entropy
    - **Optimizer**: RMSprop (lr = 0.001)

    ## Team

    *Add your team info here.*
    """
)
