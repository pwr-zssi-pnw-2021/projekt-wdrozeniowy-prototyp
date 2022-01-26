import joblib
import librosa
import streamlit as st

from projekt_wdrozeniowy_prototyp.sound_generator import (
    ArraySource,
    MFCCPreprocessor,
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
model = joblib.load('./svm.joblib')


uploaded_file = st.file_uploader('Select audio file')
if uploaded_file is not None:
    st.caption('Press play to play audio.')
    st.audio(uploaded_file)

    data, sr = librosa.load(uploaded_file, sr=48000)
    source = ArraySource(data, sr)
    generator = SoundGenerator(source, MFCCPreprocessor(sr))

    st.caption('Press analyze to analyze emotions in audio.')
    if st.button('Analyze'):
        placeholder = st.empty()
        for v in generator.generate():
            prob = model.predict_proba(v.T).mean(axis=0)

            with placeholder.container():
                for emotion, value in zip(EMOTIONS, prob):
                    st.text(
                        f'{emotion:10}: {value:.3f} | {round(value * 10 * 4) * "#"}'
                    )
