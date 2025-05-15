# 🧠 Bottle Classification App

This is a Streamlit web application designed to classify bottles into two categories: **plastic** and **glass**.

The app is part of a **supervised learning project** led by **Nadia Urban** at **Shanghai Thomas School**, where students learn how to design and deploy machine learning models through hands-on, real-world problems.

---

## 🎓 Project Overview

This project follows five main stages:

1. **Model Design** – Defining the task of distinguishing between plastic and glass bottles  
2. **Data Collection** – Gathering image data of plastic and glass bottles  
3. **Model Training** – Using Google's Teachable Machine to train a CNN classifier  
4. **Model Assessment** – Evaluating model performance and class balance  
5. **Web App Design** – Deploying the model in an interactive, user-friendly Streamlit app

---

## 🛠️ App Description

The goal of this app is to **help people recognize whether a bottle is made of plastic or glass**, especially when it's not immediately obvious.

### 🧾 Model Information
- **Classes:**  
  1. Plastic  
  2. Glass  
- **Goal:** 🎯 We wanted to develop an AI model that can recognize plastic bottles from glass bottles  
- **Data Type:** 🖼️ Images of bottles in two categories (plastic and glass)  
- **Data Source:** 🌐 Collected online from **bing.com** and **kaggle.com**
- **Training:** 🏋️ Teachable Machine  
- **Model Type:** 🧠 Convolutional Neural Network (CNN)

---

## 🖼️ Training Data Samples

| Class   | Image Preview   | Number of Training Images |
|---------|------------------|----------------------------|
| Glass   | `example1.jpg`   | 196 photos                 |
| Plastic | `example2.jpg`   | 2,217 photos               |

(*These example images are included in the app for reference.*)

---

## 👩‍🔬 Model Authors

- **[Student Name 1]**
- **[Student Name 2]**

---

## ✨ Credits

This project was developed as part of the **AI & Machine Learning program** at **Shanghai Thomas School**, designed and taught by **Nadia Urban**.

---

## 🚀 Deployment

The app is deployed using [Streamlit Cloud](https://streamlit.io/cloud) and can be run locally by installing the required dependencies:

```bash
pip install streamlit tensorflow pillow numpy
streamlit run app.py
