# BERT Relevancy webUI
- Consists of a flask api that calculates how similar a link is to the provided context, at the endpoint `/calculate_relevance`
- example usage :
  - `~$ python3 api_server.py` (Starts the server on localhost : 5000)
  - `~$ curl -XPOST -H "Content-type: application/json" -d '{"webpage": "https://example.com", "context_string": "your context string"}' 'localhost:5000/calculate_relevance'` (Returns a similarity metric)
   ![CleanShot 2023-07-11 at 09 09 34@2x](https://github.com/internetarchive/tarb_gsoc23_content_drift/assets/63366288/4986a970-1096-498e-b542-30d03baa0138)

  -  To run the webUI, `~$ streamlit run app.py`

 ![UI_Preview](https://github.com/internetarchive/tarb_gsoc23_content_drift/assets/63366288/1c2ffc4e-56d0-4725-ae30-a3a5fbda7173)
