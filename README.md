# PDF Q&A Chatbot

This project is a Streamlit web application that allows users to upload a PDF file and ask questions about its content. The application extracts text from the PDF, splits the text into manageable chunks, and uses the Google Generative AI model (through LangChain) to generate answers to user questions based on the PDF content.

## --------- Features --------- 

- Upload PDF files and extract text from them.
- Ask questions about the content of the PDF.
- Get answers generated by Google Generative AI.
- View conversation history within the app.

## -------- Installation ------

To run this project, you need to have Python installed on your system. You also need to install the required Python modules. Follow the steps below to set up the project:

1. Clone the repository:

2. Create and activate a virtual environment (optional but recommended):
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. Install the required Python modules:
    ```sh
    pip install -r requirements.txt
    ```

4. Set up your Google API key:
    - Replace `"your_google_api_key_here"` in `hello.py` with your actual Google API key.

## ----------- Usage -------- 

To run the Streamlit application, use the following command:

```sh
streamlit run hello.py
```

## ------  How It Works ------

1. Upload PDF: Upload a PDF file using the file uploader.
2. Extract Text: The application extracts text from the uploaded PDF file.
3. Ask Questions: Enter a question about the PDF content.
4. Get Answers: The application uses Google Generative AI to generate answers based on the extracted text.
5. View History: The conversation history is displayed within the app.

## -------- License ------ 
This project is licensed under the MIT License. See the LICENSE file for details.

## --------- Acknowledgments -------
  1. PyMuPDF for PDF text extraction.
  2. LangChain for the Google Generative AI integration.
  3. Streamlit for creating the web application.
