from PIL import Image as Image, ImageOps as ImagOps
import cv2
from keras.models import load_model
import numpy as np
import streamlit as st

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(
    page_title="Registro con Validación Facial",
    page_icon="👤",
    layout="centered",
)

# --- CARGA DEL MODELO ---
@st.cache_resource
def cargar_modelo():
    return load_model("keras_model.h5")


model = cargar_modelo()

# --- CABECERA E INTERFAZ ---
st.title("🔐 Registro de Usuario")
st.write(
    "Para iniciar el registro, primero debemos verificar tu identidad usando la cámara."
)

# Imagen decorativa (opcional, tu archivo OIG5.jpg)
try:
    image = Image.open("OIG5.jpg")
    st.image(image, width=350)
except Exception:
    pass

with st.sidebar:
    st.subheader("Verificación Biométrica")
    st.write(
        "Este sistema utiliza un modelo entrenado en Teachable Machine para validar tu presencia antes de habilitar el formulario."
    )

st.divider()

# --- PASO 1: VERIFICACIÓN CON CÁMARA ---
st.subheader("Paso 1: Validación de Identidad")
img_file_buffer = st.camera_input("Toma una foto para validar tu acceso")

persona_detectada = False

if img_file_buffer is not None:
    # --- TU CÓDIGO ORIGINAL DE DETECCIÓN (SIN CAMBIOS) ---
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    img = Image.open(img_file_buffer)

    newsize = (224, 224)
    img = img.resize(newsize)
    img_array = np.array(img)

    # Normalize the image
    normalized_image_array = (img_array.astype(np.float32) / 127.0) - 1
    data[0] = normalized_image_array

    # run the inference
    prediction = model.predict(data)
    print(prediction)

    # Evaluación según tus salidas de predicción
    prob_izquierda = prediction[0][0]
    prob_arriba = prediction[0][1]

    if prob_izquierda > 0.5:
        st.header("Izquierda, con Probabilidad: " + str(prob_izquierda))
        persona_detectada = True
    elif prob_arriba > 0.5:
        st.header("Arriba, con Probabilidad: " + str(prob_arriba))
        persona_detectada = True
    # if prediction[0][2]>0.5:
    #  st.header('Derecha, con Probabilidad: '+str( prediction[0][2]))

# --- PASO 2: FORMULARIO DE REGISTRO CONDICIONAL ---
st.divider()
st.subheader("Paso 2: Datos de Registro")

if persona_detectada:
    st.success("✅ Validación exitosa. Puedes completar tu registro.")

    with st.form("form_registro"):
        nombre = st.text_input("Nombre Completo")
        correo = st.text_input("Correo Electrónico")
        password = st.text_input("Contraseña", type="password")

        submit = st.form_submit_button("Completar Registro")

        if submit:
            if nombre and correo and password:
                st.balloons()
                st.success(f"¡Registro completado con éxito! Bienvenido, {nombre}.")
            else:
                st.error("Por favor completa todos los campos del formulario.")
else:
    st.warning(
        "⚠️ Debes tomarte una foto donde el sistema te detecte para habilitar el formulario de registro."
    )
