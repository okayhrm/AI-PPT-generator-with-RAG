import streamlit as st

st.set_page_config(layout="wide")
st.title("Mini Check")
st.write("If you see this, Streamlit is rendering correctly.")

name = st.text_input("Your name")

if st.button("Ping"):
    if name:
        st.success(f"Hello {name} 👋")
    else:
        st.success("Hello stranger 👋")
