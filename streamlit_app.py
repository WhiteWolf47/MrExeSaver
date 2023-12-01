import os
import streamlit as st

from file_checker import checkFile

# ----------- Sidebar
page = st.sidebar.selectbox('Page Navigation', ["MrExeSaver", "Model Description", "Model Performance"])

st.sidebar.markdown("""---""")
st.sidebar.write("Made with :heart: by [Anurag](https://github.com/WhiteWolf47) [Aditya](https://github.com/WhiteWolf47) [Naman](https://github.com/WhiteWolf47)")
if page=="MrExeSaver":
    file = st.file_uploader("Upload a file to check for malwares:", accept_multiple_files=True)
    model_name = st.selectbox(
        'Choose a Model',
        ('RandomForest', 'DecisionTree', 'KNeighbors', 'AdaBoost','SGD','ExtraTrees','GaussianNB'))
    temp = "malwares/tempFile"
    if st.button("Start Detection"):
        if len(file):
            if not model_name:
                st.error('Please select an option')
            else:
                with st.spinner("Checking..."):
                    for i in file:
                        open('malwares/tempFile', 'wb').write(i.getvalue())
                        legitimate = checkFile(temp, model_name)
                        os.remove("malwares/tempFile")
                        if legitimate:
                            st.write(f"File {i.name} seems *LEGITIMATE*!")
                        else:
                            st.markdown(f"File {i.name} is probably a **MALWARE**!!!")

elif page == "Model Description":
    st.header("About the Models")
    st.subheader("RandomForest")
    st.write("Random Forest is an ensemble learning method that constructs a multitude of decision trees during training. It provides high accuracy, handles large datasets well, and is less prone to overfitting.")
    st.subheader("DecisionTrees")
    st.write("A Decision Tree Classifier partitions the data into subsets based on feature values, making decisions in a tree-like structure. It is interpretable and useful for understanding the factors contributing to classification.")
    st.subheader("KNeighbors")
    st.write("K-Nearest Neighbors (KNN) is a simple and effective classification algorithm. It classifies data points based on the majority class of their k-nearest neighbors.")
    st.subheader("AdaBoostClassifier")
    st.write("AdaBoost is an ensemble learning method that combines multiple weak classifiers to create a strong classifier. It focuses on improving the accuracy of misclassified instances.")
    st.subheader("SGDClassifier")
    st.write("Stochastic Gradient Descent (SGD) is an iterative optimization algorithm that updates the model's parameters to minimize the loss function. SGDClassifier is particularly useful for large datasets.")
    st.subheader("ExtraTreesClassifier")
    st.write("Extra Trees, similar to Random Forest, is an ensemble learning algorithm. It constructs a forest of unpruned decision trees and further randomizes the decision-making process, enhancing diversity.")
    st.subheader("GaussianNB")
    st.write("Gaussian Naive Bayes is a probabilistic classifier based on the Bayes' theorem. It assumes that features are conditionally independent, making it computationally efficient and suitable for high-dimensional data.")

elif page == "Model Performance":
    st.header("Model Performance")
    st.subheader("For first Dataset:")
    st.image("d1.png")
    st.subheader("For second Dataset")
    st.image("d2.png")