import os
import streamlit as st

from file_checker import checkFile

st.title("MrExeSaver")

file = st.file_uploader("Upload a file to check for malwares:", accept_multiple_files=True)
option = st.selectbox(
    'Choose a Model',
    ('RandomForest', 'DecisionTree', 'KNeighbors', 'AdaBoost','SGD','ExtraTrees','GaussianNB'))

if st.button("Start Detection"):
    if len(file):
        if not option:
            st.error('Please select an option')
        else:
            with st.spinner("Checking..."):
                for i in file:
                    open('malwares/tempFile', 'wb').write(i.getvalue())
                    legitimate = checkFile("malwares/tempFile")
                    os.remove("malwares/tempFile")
                    if legitimate:
                        st.write(f"File {i.name} seems *LEGITIMATE*!")
                    else:
                        st.markdown(f"File {i.name} is probably a **MALWARE**!!!")
