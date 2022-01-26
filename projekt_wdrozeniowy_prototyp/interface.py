import librosa
import numpy as np
import streamlit as st

from projekt_wdrozeniowy_prototyp.sound_generator import (
    ArraySource,
    SoundGenerator,
)

EMOTIONS = [
    'Anger',
    'Disgust',
    'Fear',
    'Happiness',
    'Neutral',
    'Sadness',
    'Surprise',
]

st.title('Real Time Speech Emotion Recognition')


uploaded_file = st.file_uploader('Select audio file')
if uploaded_file is not None:
    st.caption('Press play to play audio.')
    st.audio(uploaded_file)

    data, sr = librosa.load(uploaded_file, sr=48000)
    source = ArraySource(data, sr)
    generator = SoundGenerator(source)

    st.caption('Press analyze to analyze emotions in audio.')
    if st.button('Analyze'):
        placeholder = st.empty()
        for v in generator.generate():
            mlp = joblib.load('model/svm.joblib')
            predict = mlp.predict_proba(v.T)
            prob = np.mean(predict, axis=0)

            with placeholder.container():
                for emotion, value in zip(EMOTIONS, prob):
                    st.text(
                        f'{emotion:10}: {value:.3f} | {round(value * 10 * 4) * "#"}'
                    )
